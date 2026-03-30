import random
import torch

from environment.catan_env import CatanEnv
from core.constants import ActionType, PlayerId
from core.constants import Resource


class Evaluator:
    def __init__(self, model_builder, seed=None, device="cpu"):
        self.model_builder = model_builder
        self.device = device
        self.seed = seed

        self.policies = [model_builder() for _ in range(4)]
        self.env = CatanEnv(seed=seed)
        self.use_lstm = getattr(self.policies[0], "use_lstm", False)
        self.lstm_dim = getattr(self.policies[0], "lstm_dim", 256)

    def reset(self):
        self.current_hidden_states = {}
        self.current_observations = {}

        for player_id in [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]:
            self.current_hidden_states[player_id] = self._zero_hidden_state()

        self.order = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]
        random.shuffle(self.order)

        self.policy_map = {}
        for i, player_id in enumerate(self.order):
            self.policy_map[player_id] = i

        obs = self.env.reset()
        obs = self._obs_to_model_input(obs)
        current_player = self.env.get_current_player_id()
        self.current_observations[current_player] = obs

    def run_evaluation_game(self, deterministic=False, max_steps=1000):
        for policy in self.policies:
            policy.eval()

        self.reset()

        done = False
        total_steps = 0
        policy_decisions = 0

        while not done and total_steps < max_steps:
            current_player = self.env.get_current_player_id()

            obs = self.current_observations[current_player]
            hidden_state = self.current_hidden_states[current_player]
            action_mask = self._build_action_mask()

            _, action_dict, _, next_hidden_state, _ = self.policies[
                self.policy_map[current_player]
            ].act(
                obs=obs,
                action_masks=action_mask,
                hidden_state=hidden_state,
                done_mask=torch.ones(1, 1, device=self.device),
                deterministic=deterministic,
            )

            if self.policy_map[current_player] == 0:
                policy_decisions += 1

            self.current_hidden_states[current_player] = next_hidden_state

            env_action = self._decode_action(action_dict)

            next_obs, _, done, _ = self.env.step(env_action)
            next_obs = self._obs_to_model_input(next_obs)

            next_player = self.env.get_current_player_id()
            self.current_observations[next_player] = next_obs

            total_steps += 1

        winner = self.order.index(self.env.engine.winner) if self.env.engine.winner is not None else -1
        active_player = self.order[0]
        victory_points = self.env.engine.players[active_player].victory_points

        return winner, victory_points, total_steps, policy_decisions

    def update_policies(self, state_dicts):
        for i in range(4):
            self.policies[i].load_state_dict(state_dicts[i])

    def _zero_hidden_state(self):
        if not self.use_lstm:
            return None

        h = torch.zeros(1, 1, self.lstm_dim, device=self.device)
        c = torch.zeros(1, 1, self.lstm_dim, device=self.device)
        return (h, c)

    def _obs_to_model_input(self, obs):
        model_obs = {
            "tile_features": torch.tensor(
                self._extract_tile_features(obs),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "current_player_main": torch.tensor(
                self._extract_current_player_features(obs),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "current_player_hidden_dev": [torch.tensor([0], dtype=torch.long, device=self.device)],
            "current_player_played_dev": [torch.tensor([0], dtype=torch.long, device=self.device)],
            "next_player_main": torch.tensor(
                self._extract_other_player_features(obs, 0),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=self.device)],
            "next_next_player_main": torch.tensor(
                self._extract_other_player_features(obs, 1),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "next_next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=self.device)],
            "next_next_next_player_main": torch.tensor(
                self._extract_other_player_features(obs, 2),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0),
            "next_next_next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=self.device)],
        }

        return model_obs

    def _extract_tile_features(self, obs):
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
        player = obs["player"]
        resources = player["resources"]

        vec = [
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
        current_player = obs["game"]["current_player"]
        order = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]

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

    def _build_action_mask(self):
        legal_actions = self.env.get_legal_actions()

        action_type_mask = torch.zeros(1, 10, dtype=torch.float32, device=self.device)
        settlement_mask = torch.ones(1, 54, dtype=torch.float32, device=self.device) * 1e-8
        road_mask = torch.ones(1, 72, dtype=torch.float32, device=self.device) * 1e-8
        city_mask = torch.ones(1, 54, dtype=torch.float32, device=self.device) * 1e-8
        robber_mask = torch.ones(1, 19, dtype=torch.float32, device=self.device) * 1e-8
        trade_mask = torch.ones(1, 2, dtype=torch.float32, device=self.device)
        discard_wood_mask = torch.ones(1, 8, dtype=torch.float32, device=self.device) * 1e-8
        discard_brick_mask = torch.ones(1, 8, dtype=torch.float32, device=self.device) * 1e-8
        discard_sheep_mask = torch.ones(1, 8, dtype=torch.float32, device=self.device) * 1e-8
        discard_wheat_mask = torch.ones(1, 8, dtype=torch.float32, device=self.device) * 1e-8
        discard_ore_mask = torch.ones(1, 8, dtype=torch.float32, device=self.device) * 1e-8

        string_to_action_type = {
            "end_turn": ActionType.END_TURN,
            "build_road": ActionType.BUILD_ROAD,
            "build_settlement": ActionType.BUILD_SETTLEMENT,
            "build_city": ActionType.BUILD_CITY,
            "buy_dev_card": ActionType.BUY_DEV_CARD,
            "play_dev_card": ActionType.PLAY_DEV_CARD,
            "move_robber": ActionType.MOVE_ROBBER,
            "bank_trade": ActionType.TRADE_BANK,
            "trade_player": ActionType.TRADE_PLAYER,
            "discard_cards": ActionType.DISCARD_CARDS,
        }

        for action in legal_actions:
            raw_type = action["type"]
            if isinstance(raw_type, str):
                action_type = string_to_action_type.get(raw_type, None)
            elif isinstance(raw_type, int):
                action_type = ActionType(raw_type)
            else:
                action_type = raw_type

            if action_type is None:
                continue
            action_type_mask[0, int(action_type)] = 1.0

            if action_type == ActionType.BUILD_SETTLEMENT:
                settlement_mask[0, action["vertex_id"]] = 1.0
            elif action_type == ActionType.BUILD_ROAD:
                road_mask[0, action["connection_id"]] = 1.0
            elif action_type == ActionType.BUILD_CITY:
                city_mask[0, action["vertex_id"]] = 1.0
            elif action_type == ActionType.MOVE_ROBBER:
                robber_mask[0, action["tile_id"]] = 1.0
            elif action_type == ActionType.DISCARD_CARDS:
                pass

        return {
            "action_type": action_type_mask,
            "settlement": settlement_mask,
            "road": road_mask,
            "city": city_mask,
            "robber": robber_mask,
            "trade": trade_mask,
            "discard_wood": discard_wood_mask,
            "discard_brick": discard_brick_mask,
            "discard_sheep": discard_sheep_mask,
            "discard_wheat": discard_wheat_mask,
            "discard_ore": discard_ore_mask,
        }

    def _decode_action(self, action_dict):
        legal_actions = self.env.get_legal_actions()
        if len(legal_actions) == 0:
            return None

        action_type_idx = int(action_dict["action_type"].squeeze().cpu().item())
        candidates = [action for action in legal_actions if int(action["type"]) == action_type_idx]

        if len(candidates) == 0:
            return legal_actions[-1]

        chosen = candidates[0]
        action_type = chosen["type"]

        if action_type == ActionType.BUILD_SETTLEMENT:
            vertex_id = int(action_dict["settlement"].squeeze().cpu().item())
            for candidate in candidates:
                if candidate.get("vertex_id") == vertex_id:
                    return candidate
            return chosen

        if action_type == ActionType.BUILD_ROAD:
            connection_id = int(action_dict["road"].squeeze().cpu().item())
            for candidate in candidates:
                if candidate.get("connection_id") == connection_id:
                    return candidate
            return chosen

        if action_type == ActionType.BUILD_CITY:
            vertex_id = int(action_dict["city"].squeeze().cpu().item())
            for candidate in candidates:
                if candidate.get("vertex_id") == vertex_id:
                    return candidate
            return chosen

        if action_type == ActionType.MOVE_ROBBER:
            tile_id = int(action_dict["robber"].squeeze().cpu().item())
            for candidate in candidates:
                if candidate.get("tile_id") == tile_id:
                    return candidate
            return chosen

        if action_type == ActionType.DISCARD_CARDS:
            current_player = self.env.get_current_player_id()
            required = self.env.engine.robber_discard_required.get(current_player, 0)
            player_resources = self.env.engine.players[current_player].resources

            chosen_discard = {
                Resource.WOOD: int(action_dict["discard_wood"].squeeze().cpu().item()),
                Resource.BRICK: int(action_dict["discard_brick"].squeeze().cpu().item()),
                Resource.SHEEP: int(action_dict["discard_sheep"].squeeze().cpu().item()),
                Resource.WHEAT: int(action_dict["discard_wheat"].squeeze().cpu().item()),
                Resource.ORE: int(action_dict["discard_ore"].squeeze().cpu().item()),
            }

            for res in chosen_discard:
                chosen_discard[res] = min(chosen_discard[res], int(player_resources.get(res, 0)))

            total = sum(chosen_discard.values())
            if total > required:
                while total > required:
                    reducible = [r for r, v in chosen_discard.items() if v > 0]
                    if not reducible:
                        break
                    r = max(reducible, key=lambda x: chosen_discard[x])
                    chosen_discard[r] -= 1
                    total -= 1
            elif total < required:
                for res in [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]:
                    available = int(player_resources.get(res, 0))
                    while total < required and chosen_discard[res] < available:
                        chosen_discard[res] += 1
                        total += 1
                    if total == required:
                        break

            if total != required:
                candidate = {r: 0 for r in chosen_discard}
                remaining = required
                for res in [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]:
                    available = int(player_resources.get(res, 0))
                    if remaining <= 0:
                        break
                    allocate = min(available, remaining)
                    candidate[res] = allocate
                    remaining -= allocate
                chosen_discard = candidate

            return {
                "type": "discard_cards",
                "resources": chosen_discard,
            }

        return chosen