from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from core.constants import PlayerId, Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv


class HybridRolloutManager:
    def __init__(
        self,
        num_envs: int,
        env_config: Dict[str, Any],
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.device = device

        self.envs: List[CatanEnv] = [
            CatanEnv(**env_config) for _ in range(num_envs)
        ]

        self.obs = [env.reset() for env in self.envs]

    def _route_phase(self, env: CatanEnv) -> str:
        phase = env.get_phase()

        if phase in (TurnPhase.MAIN_ACTION, TurnPhase.END_TURN):
            return "gameplay"

        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            return "trade"

        return "auto"

    def _build_gameplay_obs(self, env: CatanEnv) -> Dict[str, torch.Tensor]:
        """Build gameplay observation dictionary for DQN policy."""
        # Get raw observation from environment
        raw_obs = env.build_gameplay_observation()

        # Convert to tensor format expected by DQN policy
        obs = {
            "board": torch.tensor(raw_obs["board_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "self": torch.tensor(raw_obs["self_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "opponent": torch.tensor(raw_obs["opponent_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "gameplay_candidates": torch.tensor(raw_obs["gameplay_candidates"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "gameplay_mask": torch.tensor(raw_obs["gameplay_mask"], dtype=torch.float32, device=self.device).unsqueeze(0),
            # Add dummy trade_history for consistent structure
            "trade_history": {
                "proposer_ids": torch.zeros((1, 4), dtype=torch.long, device=self.device),
                "target_ids": torch.zeros((1, 4), dtype=torch.long, device=self.device),
                "response_types": torch.zeros((1, 4), dtype=torch.long, device=self.device),
                "offers": torch.zeros((1, 4, 5), dtype=torch.float32, device=self.device),
                "requests": torch.zeros((1, 4, 5), dtype=torch.float32, device=self.device),
                "accepted_flags": torch.zeros((1, 4), dtype=torch.float32, device=self.device),
                "turn_numbers": torch.zeros((1, 4), dtype=torch.float32, device=self.device),
            },
        }

        return obs

    def _build_trade_obs(self, env: CatanEnv) -> Dict[str, torch.Tensor]:
        """Build trade observation dictionary for PPO policy."""
        # Get raw observation from environment
        raw_obs = env.build_trade_observation()

        # Convert to tensor format - ensure same keys as gameplay for batching
        obs = {
            "board": torch.tensor(raw_obs["board_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "self": torch.tensor(raw_obs["self_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "opponent": torch.tensor(raw_obs["opponent_state"], dtype=torch.float32, device=self.device).unsqueeze(0),
            # Add dummy values for gameplay keys that trade obs doesn't have
            "gameplay_candidates": torch.zeros((1, 10, 40), dtype=torch.float32, device=self.device),  # dummy
            "gameplay_mask": torch.zeros((1, 10), dtype=torch.float32, device=self.device),  # dummy
            # Trade-specific data
            "trade_history": {
                "proposer_ids": torch.tensor(raw_obs["trade_history"]["proposer_ids"], dtype=torch.long, device=self.device).unsqueeze(0),
                "target_ids": torch.tensor(raw_obs["trade_history"]["target_ids"], dtype=torch.long, device=self.device).unsqueeze(0),
                "response_types": torch.tensor(raw_obs["trade_history"]["response_types"], dtype=torch.long, device=self.device).unsqueeze(0),
                "offers": torch.tensor(raw_obs["trade_history"]["offers"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "requests": torch.tensor(raw_obs["trade_history"]["requests"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "accepted_flags": torch.tensor(raw_obs["trade_history"]["accepted_flags"], dtype=torch.float32, device=self.device).unsqueeze(0),
                "turn_numbers": torch.tensor(raw_obs["trade_history"]["turn_numbers"], dtype=torch.float32, device=self.device).unsqueeze(0),
            },
        }

        return obs

    def _default_empty_trade_vector(self) -> Dict[Resource, int]:
        return {
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        }

    def _one_hot_trade_vector(self, idx: int) -> Dict[Resource, int]:
        vector = self._default_empty_trade_vector()
        resources = [
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ]
        if 0 <= idx < len(resources):
            vector[resources[idx]] = 1
        return vector

    def _decode_gameplay_action(self, action_idx: int, env: CatanEnv) -> dict:
        phase = env.get_phase()

        if phase == TurnPhase.END_TURN:
            return {"type": "end_turn"}

        if action_idx == 0:
            return {"type": "build_road"}
        if action_idx == 1:
            return {"type": "build_settlement"}
        if action_idx == 2:
            return {"type": "build_city"}

        return {"type": "end_main_action"}

    def _decode_trade_action(self, action_dict: Dict[str, torch.Tensor], env: CatanEnv) -> dict:
        phase = env.get_phase()

        if phase == TurnPhase.TRADE_PROPOSE:
            action_idx = int(action_dict["action_type"].item())
            target_idx = int(action_dict["target"].item())

            players = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]
            current_player = env.get_current_player_id()
            legal_targets = [p for p in players if p != current_player]

            if len(legal_targets) == 0:
                return {"type": "skip_trade"}

            target = legal_targets[target_idx % len(legal_targets)]

            offer_idx = int(torch.argmax(action_dict["offer"], dim=-1).item())
            request_idx = int(torch.argmax(action_dict["request"], dim=-1).item())

            if action_idx == 0:
                return {
                    "type": "propose_trade",
                    "target": target,
                    "offer": self._one_hot_trade_vector(offer_idx),
                    "request": self._one_hot_trade_vector(request_idx),
                }

            return {"type": "skip_trade"}

        if phase == TurnPhase.TRADE_RESPOND:
            action_idx = int(action_dict["action_type"].item())
            offer_idx = int(torch.argmax(action_dict["offer"], dim=-1).item())
            request_idx = int(torch.argmax(action_dict["request"], dim=-1).item())

            if action_idx == 1:
                return {"type": "accept_trade", "response_type": "accept"}

            if action_idx == 2:
                return {"type": "reject_trade", "response_type": "reject"}

            if action_idx == 3:
                return {
                    "type": "counter_trade",
                    "response_type": "counter",
                    "counter_offer": self._one_hot_trade_vector(offer_idx),
                    "counter_request": self._one_hot_trade_vector(request_idx),
                }

            return {"type": "reject_trade", "response_type": "reject"}

        return {"type": "skip_trade"}

    def _maybe_reset_env(self, env: CatanEnv, done: bool):
        if done:
            return env.reset()
        return env.get_observation()

    def collect(
        self,
        dqn_trainer,
        epsilon: float,
        trade_policy,
        rollout_storage,
        steps: int = 128,
    ) -> Tuple[List[Dict], Any]:
        gameplay_transitions: List[Dict] = []
        rollout_storage.reset()

        for _ in range(steps):
            for i, env in enumerate(self.envs):
                route = self._route_phase(env)

                if route == "gameplay":
                    obs_tensor = self._build_gameplay_obs(env)

                    action_mask = torch.ones(
                        dqn_trainer.q_network.num_actions,
                        dtype=torch.float32,
                        device=self.device,
                    )

                    action_idx = dqn_trainer.act(
                        obs_tensor,
                        action_mask.unsqueeze(0),
                        epsilon,
                    )

                    env_action = self._decode_gameplay_action(action_idx, env)
                    _, reward, done, _ = env.step(env_action)

                    next_obs_tensor = self._build_gameplay_obs(env)
                    next_action_mask = torch.ones(
                        dqn_trainer.q_network.num_actions,
                        dtype=torch.float32,
                        device=self.device,
                    )

                    gameplay_transitions.append(
                        {
                            "obs": obs_tensor.detach().cpu(),
                            "action": int(action_idx),
                            "reward": float(reward),
                            "next_obs": next_obs_tensor.detach().cpu(),
                            "done": bool(done),
                            "action_mask": action_mask.detach().cpu(),
                            "next_action_mask": next_action_mask.detach().cpu(),
                        }
                    )

                    self.obs[i] = self._maybe_reset_env(env, done)
                    continue

                if route == "trade":
                    obs_dict = self._build_trade_obs(env)

                    with torch.no_grad():
                        value, action_dict, log_prob, _, tom = trade_policy.act(
                            obs_dict,
                            deterministic=False,
                        )

                    env_action = self._decode_trade_action(action_dict, env)
                    _, reward, done, _ = env.step(env_action)

                    rollout_storage.add(
                        obs=obs_dict,
                        action=action_dict,
                        reward=reward,
                        value=value,
                        log_prob=log_prob,
                        done=done,
                    )

                    self.obs[i] = self._maybe_reset_env(env, done)
                    continue

                _, reward, done, _ = env.step(None)
                self.obs[i] = self._maybe_reset_env(env, done)

        return gameplay_transitions, rollout_storage