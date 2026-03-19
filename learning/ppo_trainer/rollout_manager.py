"""
Rollout collection manager.

This file manages:
- multiple parallel Catan environments
- per-environment hidden states
- trajectory collection for PPO
- active-player self-play style sampling

This is the single-process version.
A distributed wrapper will call into this manager from workers.
"""

import copy
import random
import torch

from environment.catan_env import CatanEnv
from core.constants import PlayerId


class RolloutManager:
    def __init__(self, num_envs, num_steps, model_builder, seed=None, device="cpu"):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.model_builder = model_builder
        self.seed = seed
        self.device = device

        self.policies = [model_builder() for _ in range(4)]
        self.envs = [CatanEnv(seed=None if seed is None else seed + i) for i in range(num_envs)]

        self.use_lstm = getattr(self.policies[0], "use_lstm", False)
        self.lstm_dim = getattr(self.policies[0], "lstm_dim", 256)

        self.policy_maps = []
        self.active_player_ids = []

        self._initialize_policy_assignments()
        self.reset()

    def _initialize_policy_assignments(self):
        """
        For each environment, shuffle player-policy assignment.

        policy 0 is always the learning policy we care about.
        The remaining policies are opponents from the current pool.
        """
        self.policy_maps = []
        self.active_player_ids = []

        for _ in range(self.num_envs):
            order = [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]
            random.shuffle(order)

            policy_map = {}
            for i, player_id in enumerate(order):
                policy_map[player_id] = self.policies[i]

            self.policy_maps.append(policy_map)
            self.active_player_ids.append(order[0])

    def reset(self):
        """
        Reset rollout buffers and env-local state.
        """
        self.observations = [[] for _ in range(self.num_envs)]
        self.hidden_states = [[] for _ in range(self.num_envs)]
        self.rewards = [[] for _ in range(self.num_envs)]
        self.actions = [[] for _ in range(self.num_envs)]
        self.action_masks = [[] for _ in range(self.num_envs)]
        self.action_log_probs = [[] for _ in range(self.num_envs)]
        self.done_masks = [[] for _ in range(self.num_envs)]

        self.current_hidden_states = [{} for _ in range(self.num_envs)]
        self.current_observations = [{} for _ in range(self.num_envs)]

        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            obs = self._obs_to_model_input(obs)

            current_player = self.envs[env_idx].get_current_player_id()
            self.current_observations[env_idx][current_player] = obs

            self.done_masks[env_idx].append(1.0)

            for player_id in [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]:
                self.current_hidden_states[env_idx][player_id] = self._zero_hidden_state()

            if current_player == self.active_player_ids[env_idx]:
                self.observations[env_idx].append(obs)
                self.hidden_states[env_idx].append(self.current_hidden_states[env_idx][current_player])

    def _zero_hidden_state(self):
        if not self.use_lstm:
            return None

        h = torch.zeros(1, 1, self.lstm_dim, device=self.device)
        c = torch.zeros(1, 1, self.lstm_dim, device=self.device)
        return (h, c)

    def update_policy(self, state_dict, policy_id=0):
        """
        Update one or all policy copies with new weights.
        """
        self.policies[policy_id].load_state_dict(state_dict)

    def gather_rollouts(self):
        """
        Collect one rollout batch from all environments.

        Returns:
            (
                observations,
                hidden_states,
                rewards,
                actions,
                action_masks,
                action_log_probs,
                done_masks,
            )
        """
        for policy in self.policies:
            policy.eval()

        with torch.no_grad():
            for env_idx in range(self.num_envs):
                done_mask_tensor = torch.tensor(
                    self.done_masks[env_idx][0],
                    dtype=torch.float32,
                    device=self.device,
                ).view(1, 1)

                cumulative_rewards = {
                    player_id: 0.0
                    for player_id in [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]
                }

                done_since_last_active_turn = False

                while len(self.observations[env_idx]) < self.num_steps + 1:
                    env = self.envs[env_idx]
                    current_player = env.get_current_player_id()

                    obs = self.current_observations[env_idx][current_player]
                    hidden_state = self.current_hidden_states[env_idx][current_player]

                    action_mask = self._build_action_mask(env)

                    policy = self.policy_maps[env_idx][current_player]

                    values, action_dict, action_log_prob, next_hidden_state, _ = policy.act(
                        obs=obs,
                        action_masks=action_mask,
                        hidden_state=hidden_state,
                        done_mask=done_mask_tensor,
                        deterministic=False,
                    )

                    self.current_hidden_states[env_idx][current_player] = next_hidden_state

                    env_action = self._decode_action(env, action_dict)

                    next_obs, reward, done, _ = env.step(env_action)
                    next_obs = self._obs_to_model_input(next_obs)

                    for player_id in cumulative_rewards:
                        if player_id == current_player:
                            cumulative_rewards[player_id] += float(reward)

                    done_mask_tensor = 1.0 - torch.tensor(
                        done,
                        dtype=torch.float32,
                        device=self.device,
                    ).view(1, 1)

                    next_player = env.get_current_player_id()

                    reward_was_committed = False

                    if current_player == self.active_player_ids[env_idx]:
                        self.actions[env_idx].append(action_dict)
                        self.action_masks[env_idx].append(action_mask)
                        self.action_log_probs[env_idx].append(action_log_prob)

                    if next_player == self.active_player_ids[env_idx] and len(self.actions[env_idx]) > 0:
                        if not done_since_last_active_turn:
                            self.rewards[env_idx].append(cumulative_rewards[self.active_player_ids[env_idx]])
                            cumulative_rewards[self.active_player_ids[env_idx]] = 0.0
                            reward_was_committed = True

                    if done:
                        next_obs = env.reset()
                        next_obs = self._obs_to_model_input(next_obs)

                        self.done_masks[env_idx].append(0.0)
                        done_since_last_active_turn = False

                        if not reward_was_committed and len(self.actions[env_idx]) > 0:
                            self.rewards[env_idx].append(cumulative_rewards[self.active_player_ids[env_idx]])

                        for player_id in cumulative_rewards:
                            cumulative_rewards[player_id] = 0.0
                            self.current_hidden_states[env_idx][player_id] = self._zero_hidden_state()

                        next_player = env.get_current_player_id()

                    self.current_observations[env_idx][next_player] = next_obs

                    if next_player == self.active_player_ids[env_idx]:
                        if not done and not done_since_last_active_turn:
                            self.done_masks[env_idx].append(1.0)

                        done_since_last_active_turn = False
                        self.observations[env_idx].append(next_obs)
                        self.hidden_states[env_idx].append(self.current_hidden_states[env_idx][next_player])
                    else:
                        if done:
                            done_since_last_active_turn = True

        result = copy.deepcopy(
            (
                self.observations,
                self.hidden_states,
                self.rewards,
                self.actions,
                self.action_masks,
                self.action_log_probs,
                self.done_masks,
            )
        )

        self._after_rollouts()
        return result

    def _after_rollouts(self):
        """
        Keep final observation/hidden state and clear the rest.
        """
        for env_idx in range(self.num_envs):
            self.observations[env_idx] = [self.observations[env_idx][-1]]
            self.hidden_states[env_idx] = [self.hidden_states[env_idx][-1]]
            self.done_masks[env_idx] = [self.done_masks[env_idx][-1]]

            self.rewards[env_idx] = []
            self.actions[env_idx] = []
            self.action_masks[env_idx] = []
            self.action_log_probs[env_idx] = []

    def _obs_to_model_input(self, obs):
        """
        Convert env observation dict into model-ready tensors.

        This keeps the training code isolated from the raw env format.
        """
        device = self.device

        # placeholder/simple tensorization
        model_obs = {
            "tile_features": torch.tensor(
                self._extract_tile_features(obs),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "current_player_main": torch.tensor(
                self._extract_current_player_features(obs),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "current_player_hidden_dev": [torch.tensor([0], dtype=torch.long, device=device)],
            "current_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
            "next_player_main": torch.tensor(
                self._extract_other_player_features(obs, 0),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
            "next_next_player_main": torch.tensor(
                self._extract_other_player_features(obs, 1),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "next_next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
            "next_next_next_player_main": torch.tensor(
                self._extract_other_player_features(obs, 2),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0),
            "next_next_next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
        }

        return model_obs

    def _extract_tile_features(self, obs):
        """
        Turn board tile dicts into fixed-size numeric features.

        Current shape:
            (19, 16)
        """
        tile_features = []

        for tile in obs["board"]["tiles"]:
            resource = tile["resource"]
            number = tile["number"] if tile["number"] is not None else 0
            has_robber = 1.0 if tile["has_robber"] else 0.0

            resource_one_hot = [0.0] * 6
            if resource is not None:
                resource_idx = int(resource)
                if 0 <= resource_idx < len(resource_one_hot):
                    resource_one_hot[resource_idx] = 1.0

            base = resource_one_hot + [float(number), has_robber]
            padding = [0.0] * (16 - len(base))
            tile_features.append(base + padding)

        return tile_features

    def _extract_current_player_features(self, obs):
        """
        Build current player feature vector.

        Current size:
            32
        """
        player = obs["player"]
        resources = player["resources"]

        vec = [
            float(resources.get(PlayerId, 0)) if False else 0.0,  # kept intentionally dead/simple
            float(resources.get(0, 0)),
            float(resources.get(1, 0)),
            float(resources.get(2, 0)),
            float(resources.get(3, 0)),
            float(resources.get(4, 0)),
            float(resources.get(5, 0)),
            float(player["victory_points"]),
            float(len(player["roads"])),
            float(len(player["dev_cards"])),
            float(player["dev_victory_points"]),
        ]

        while len(vec) < 32:
            vec.append(0.0)

        return vec[:32]

    def _extract_other_player_features(self, obs, offset):
        """
        Build one opponent feature vector.

        Current size:
            24
        """
        current_player = obs["game"]["current_player"]
        order = [PlayerId.White, PlayerId.Blue, PlayerId.Orange, PlayerId.Red]

        try:
            idx = order.index(current_player)
        except ValueError:
            idx = 0

        target = order[(idx + offset + 1) % 4]
        summary = obs["players"][target]

        vec = [
            float(summary["victory_points"]),
            float(len(summary["roads"])),
            float(summary["dev_card_count"]),
            float(summary["dev_victory_points"]),
            float(summary["building_count"]),
        ]

        while len(vec) < 24:
            vec.append(0.0)

        return vec[:24]

    def _build_action_mask(self, env):
        """
        Convert legal actions into per-head masks.
        """
        legal_actions = env.get_legal_actions()

        action_type_mask = torch.zeros(1, 9, dtype=torch.float32, device=self.device)
        settlement_mask = torch.ones(1, 54, dtype=torch.float32, device=self.device) * 1e-8
        road_mask = torch.ones(1, 72, dtype=torch.float32, device=self.device) * 1e-8
        city_mask = torch.ones(1, 54, dtype=torch.float32, device=self.device) * 1e-8
        robber_mask = torch.ones(1, 19, dtype=torch.float32, device=self.device) * 1e-8
        trade_mask = torch.ones(1, 2, dtype=torch.float32, device=self.device)

        for action in legal_actions:
            action_type = action["type"]
            action_type_mask[0, int(action_type)] = 1.0

            if action_type.name == "BuildSettlement":
                settlement_mask[0, action["vertex_id"]] = 1.0
            elif action_type.name == "BuildRoad":
                road_mask[0, action["connection_id"]] = 1.0
            elif action_type.name == "BuildCity":
                city_mask[0, action["vertex_id"]] = 1.0
            elif action_type.name == "MoveRobber":
                robber_mask[0, action["tile_id"]] = 1.0

        return {
            "action_type": action_type_mask,
            "settlement": settlement_mask,
            "road": road_mask,
            "city": city_mask,
            "robber": robber_mask,
            "trade": trade_mask,
        }

    def _decode_action(self, env, action_dict):
        """
        Convert network action heads into one env action dict.

        This is the current simple decoder.
        """
        action_type_idx = int(action_dict["action_type"].squeeze().cpu().item())
        legal_actions = env.get_legal_actions()

        # try to pick a compatible legal action based on predicted type
        candidates = [a for a in legal_actions if int(a["type"]) == action_type_idx]

        if len(candidates) == 0:
            return legal_actions[-1]

        chosen = candidates[0]
        action_name = chosen["type"].name

        if action_name == "BuildSettlement":
            vertex_id = int(action_dict["settlement"].squeeze().cpu().item())
            for c in candidates:
                if c.get("vertex_id") == vertex_id:
                    return c
            return chosen

        if action_name == "BuildRoad":
            connection_id = int(action_dict["road"].squeeze().cpu().item())
            for c in candidates:
                if c.get("connection_id") == connection_id:
                    return c
            return chosen

        if action_name == "BuildCity":
            vertex_id = int(action_dict["city"].squeeze().cpu().item())
            for c in candidates:
                if c.get("vertex_id") == vertex_id:
                    return c
            return chosen

        if action_name == "MoveRobber":
            tile_id = int(action_dict["robber"].squeeze().cpu().item())
            for c in candidates:
                if c.get("tile_id") == tile_id:
                    return c
            return chosen

        return chosen