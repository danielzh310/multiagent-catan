from __future__ import annotations

from typing import Any, Dict, List

import torch

from core.constants import PlayerId, Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv


class UnifiedRolloutManager:
    MAX_GAMEPLAY_ACTIONS = 256
    GAMEPLAY_FEATURE_DIM = 40
    MAX_TRADE_ACTIONS = 128
    TRADE_FEATURE_DIM = 32

    def __init__(self, num_envs: int, device: str = "cpu", enable_trading: bool = True):
        self.num_envs = num_envs
        self.device = device
        self.enable_trading = enable_trading
        self.envs = [CatanEnv(enable_trading=enable_trading) for _ in range(num_envs)]
        self.obs = [env.reset() for env in self.envs]

    def _resource_slot(self, resource_value: Any) -> int:
        if resource_value is None:
            return -1
        try:
            return int(Resource(resource_value))
        except (ValueError, TypeError):
            return -1

    def _dev_card_slot(self, card_value: Any) -> int:
        if card_value is None:
            return -1
        try:
            return int(card_value)
        except (ValueError, TypeError):
            return -1

    def _encode_gameplay_action(self, action: Dict[str, Any], env: CatanEnv) -> List[float]:
        features = [0.0] * self.GAMEPLAY_FEATURE_DIM
        action_type = action.get("type", "")
        action_types = {
            "build_settlement": 0,
            "build_road": 1,
            "build_city": 2,
            "buy_dev_card": 3,
            "play_dev_card": 4,
            "bank_trade": 5,
            "move_robber": 6,
            "discard_cards": 7,
            "end_main_action": 8,
            "end_turn": 9,
            "roll": 10,
            "skip_trade": 11,
        }
        action_type_idx = action_types.get(action_type)
        if action_type_idx is not None and action_type_idx < 12:
            features[action_type_idx] = 1.0

        current_player = env.get_current_player_id()
        player = env.engine.players[current_player]
        victory_points = float(player.update_victory_points())

        features[12] = victory_points / 10.0
        features[13] = float(player.n_settlements) / 5.0
        features[14] = float(player.n_cities) / 4.0
        features[15] = float(player.n_roads) / 15.0

        if "vertex" in action:
            features[16] = float(action["vertex"]) / 64.0
        if "connection" in action:
            features[17] = float(action["connection"]) / 128.0
        if "tile" in action:
            features[18] = float(action["tile"]) / 19.0
        if "connection_1" in action and action["connection_1"] is not None:
            features[19] = float(action["connection_1"]) / 128.0
        if "connection_2" in action and action["connection_2"] is not None:
            features[20] = float(action["connection_2"]) / 128.0

        give_slot = self._resource_slot(action.get("give"))
        receive_slot = self._resource_slot(action.get("receive"))
        resource_slot = self._resource_slot(action.get("resource"))
        resource_1_slot = self._resource_slot(action.get("resource_1"))
        resource_2_slot = self._resource_slot(action.get("resource_2"))
        card_slot = self._dev_card_slot(action.get("card"))
        rate = action.get("rate")
        required = action.get("required")
        resources_to_discard = action.get("resources")

        if give_slot >= 0:
            features[21 + give_slot] = 1.0
        if receive_slot >= 0:
            features[26 + receive_slot] = 1.0
        if resource_slot >= 0:
            features[31] = float(resource_slot + 1) / 5.0
        if resource_1_slot >= 0:
            features[32] = float(resource_1_slot + 1) / 5.0
        if resource_2_slot >= 0:
            features[33] = float(resource_2_slot + 1) / 5.0
        if card_slot >= 0:
            features[34] = float(card_slot + 1) / 5.0
        if rate is not None:
            features[35] = float(rate) / 4.0
        if required is not None:
            features[36] = float(required) / 8.0
        if isinstance(resources_to_discard, dict):
            total_discard = 0.0
            for amount in resources_to_discard.values():
                total_discard += float(amount)
            features[37] = total_discard / 8.0
            non_zero = sum(1 for amount in resources_to_discard.values() if int(amount) > 0)
            features[38] = float(non_zero) / 5.0
        if action.get("type") == "play_dev_card":
            features[39] = 1.0

        return features

    def _build_gameplay_candidates(self, env: CatanEnv, legal_actions: List[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = torch.zeros(
            (1, self.MAX_GAMEPLAY_ACTIONS, self.GAMEPLAY_FEATURE_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros((1, self.MAX_GAMEPLAY_ACTIONS), dtype=torch.bool, device=self.device)

        capped_actions = legal_actions[: self.MAX_GAMEPLAY_ACTIONS]
        for idx, action in enumerate(capped_actions):
            candidates[0, idx] = torch.tensor(
                self._encode_gameplay_action(action, env),
                dtype=torch.float32,
                device=self.device,
            )
            mask[0, idx] = True

        if not capped_actions:
            mask[0, 0] = True

        return candidates, mask

    def _encode_trade_action(self, action: Dict[str, Any], env: CatanEnv) -> List[float]:
        features = [0.0] * self.TRADE_FEATURE_DIM
        action_type = action.get("type", "")
        action_types = {
            "skip_trade": 0,
            "propose_trade": 1,
            "accept_trade": 2,
            "reject_trade": 3,
            "counter_trade": 4,
        }
        action_type_idx = action_types.get(action_type)
        if action_type_idx is not None:
            features[action_type_idx] = 1.0

        current_player = env.get_current_player_id()
        player = env.engine.players[current_player]
        features[5] = float(player.update_victory_points()) / 10.0
        features[6] = float(sum(int(v) for v in player.resources.values())) / 20.0

        pending = env.engine.trade_manager.get_pending_trade()
        if pending is not None:
            features[7] = 1.0
            features[8] = float(pending.counter_count) / 3.0
            features[9] = float(int(pending.proposer)) / 3.0
            features[10] = float(int(pending.target)) / 3.0

        target = action.get("target")
        if target is not None:
            features[11 + int(target)] = 1.0

        offer = action.get("offer") or action.get("counter_offer") or {}
        request = action.get("request") or action.get("counter_request") or {}

        for resource, amount in offer.items():
            slot = self._resource_slot(resource)
            if slot >= 0:
                features[15 + slot] = float(amount)

        for resource, amount in request.items():
            slot = self._resource_slot(resource)
            if slot >= 0:
                features[20 + slot] = float(amount)

        features[25] = float(sum(int(v) for v in offer.values())) / 4.0
        features[26] = float(sum(int(v) for v in request.values())) / 4.0

        response_type = action.get("response_type", "")
        if response_type == "accept":
            features[27] = 1.0
        elif response_type == "reject":
            features[28] = 1.0
        elif response_type == "counter":
            features[29] = 1.0

        if action_type == "counter_trade":
            features[30] = 1.0
        if action_type == "propose_trade":
            features[31] = 1.0

        return features

    def _build_trade_candidates(self, env: CatanEnv, legal_actions: List[Dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        candidates = torch.zeros(
            (1, self.MAX_TRADE_ACTIONS, self.TRADE_FEATURE_DIM),
            dtype=torch.float32,
            device=self.device,
        )
        mask = torch.zeros((1, self.MAX_TRADE_ACTIONS), dtype=torch.bool, device=self.device)

        capped_actions = legal_actions[: self.MAX_TRADE_ACTIONS]
        for idx, action in enumerate(capped_actions):
            candidates[0, idx] = torch.tensor(
                self._encode_trade_action(action, env),
                dtype=torch.float32,
                device=self.device,
            )
            mask[0, idx] = True

        if not capped_actions:
            mask[0, 0] = True

        return candidates, mask

    def _resolve_discard_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        required = int(action.get("required", 0))
        available = action.get("available", {})

        ordered_resources = sorted(
            available.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )

        resources_to_discard: Dict[Resource, int] = {
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        }

        remaining = required
        for resource_name, count in ordered_resources:
            if remaining <= 0:
                break
            take = min(int(count), remaining)
            if take <= 0:
                continue
            try:
                resource = Resource[resource_name]
            except KeyError:
                continue
            resources_to_discard[resource] = take
            remaining -= take

        if remaining > 0:
            return action

        resolved = dict(action)
        resolved["resources"] = resources_to_discard
        return resolved

    def _phase_name(self, env: CatanEnv) -> str:
        phase = env.get_phase()
        if phase in (TurnPhase.SETUP, TurnPhase.MAIN_ACTION, TurnPhase.END_TURN):
            return "gameplay"
        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            return "trade"
        return "auto"

    def _build_obs(self, env: CatanEnv) -> Dict[str, torch.Tensor]:
        raw = env.get_observation()
        player = raw["player"]
        other_players = [v for k, v in raw["players"].items() if k != raw["game"]["current_player"]]

        def to_vec(state: dict) -> List[float]:
            resources = state.get("resources", {})
            roads_val = state.get("roads", 0)
            if isinstance(roads_val, list):
                roads_val = len(roads_val)

            dev_cards_val = state.get("dev_cards", 0)
            if isinstance(dev_cards_val, list):
                dev_cards_val = len(dev_cards_val)

            vec = [
                float(resources.get("WOOD", 0)),
                float(resources.get("BRICK", 0)),
                float(resources.get("SHEEP", 0)),
                float(resources.get("WHEAT", 0)),
                float(resources.get("ORE", 0)),
                float(state.get("victory_points", 0)),
                float(state.get("num_settlements", 0)),
                float(state.get("num_cities", 0)),
                float(roads_val),
                float(state.get("bonus_vp", 0)),
                float(state.get("dev_victory_points", 0)),
                float(dev_cards_val),
                float(state.get("played_knights", 0)),
                float(state.get("revealed_vp_cards", 0)),
            ]
            vec += [0.0] * (64 - len(vec))
            return vec[:64]

        self_vec = torch.tensor([to_vec(player)], dtype=torch.float32, device=self.device)

        op_vec = [0.0] * 64
        if other_players:
            nums = len(other_players)
            sum_vec = [0.0] * 64
            for opp in other_players:
                opp_v = to_vec(opp)
                for i in range(64):
                    sum_vec[i] += opp_v[i]
            op_vec = [x / nums for x in sum_vec]

        board_vec = torch.zeros((1, 64), dtype=torch.float32, device=self.device)

        game = raw.get("game", {})
        board_vec[0, 0] = float(game.get("turn_number", 0))
        board_vec[0, 1] = float(int(env.get_current_player_id()))
        board_vec[0, 2] = float(env.get_phase().value)
        board_vec[0, 3] = 1.0 if game.get("enable_trading", True) else 0.0

        last_roll = game.get("last_roll")
        board_vec[0, 4] = float(last_roll if last_roll is not None else 0.0)
        board_vec[0, 5] = 1.0 if game.get("robber_pending", False) else 0.0

        pending_trade = raw.get("trade")
        if pending_trade is not None:
            board_vec[0, 6] = 1.0
            board_vec[0, 7] = float(pending_trade.counter_count)

        robber_event = game.get("last_robber_event")
        if robber_event is not None:
            board_vec[0, 8] = 1.0 if robber_event.get("rolled_seven", False) else 0.0
            board_vec[0, 9] = 1.0 if robber_event.get("stolen_from") is not None else 0.0

            discarded = robber_event.get("discarded", {})
            total_discarded = 0.0
            for _, res_map in discarded.items():
                total_discarded += float(sum(res_map.values()))
            board_vec[0, 10] = total_discarded

        legal_actions = raw.get("legal_actions", [])
        board_vec[0, 11] = float(len(legal_actions))

        board_vec[0, 12] = 1.0 if game.get("initial_placement_phase", False) else 0.0
        board_vec[0, 13] = float(game.get("initial_placement_index", 0))

        stage = game.get("initial_placement_stage")
        board_vec[0, 14] = 1.0 if stage == "settlement" else 0.0
        board_vec[0, 15] = 1.0 if stage == "road" else 0.0
        board_vec[0, 16] = float(game.get("dev_card_deck_size", 0))
        board_vec[0, 17] = 1.0 if game.get("longest_road_owner") == env.get_current_player_id() else 0.0
        board_vec[0, 18] = 1.0 if game.get("largest_army_owner") == env.get_current_player_id() else 0.0
        board_vec[0, 19] = float(len(env.engine.robber_discard_queue))
        board_vec[0, 20] = float(env.engine.robber_discard_required.get(env.get_current_player_id(), 0))

        gameplay_candidates, gameplay_mask = self._build_gameplay_candidates(env, legal_actions)
        trade_candidates, trade_mask = self._build_trade_candidates(env, legal_actions)

        return {
            "board": board_vec,
            "self": self_vec,
            "opponent": torch.tensor([op_vec], dtype=torch.float32, device=self.device),
            "gameplay_candidates": gameplay_candidates,
            "gameplay_mask": gameplay_mask,
            "trade_candidates": trade_candidates,
            "trade_mask": trade_mask,
        }

    def _one_hot_trade_vector(self, idx: int) -> Dict[Resource, int]:
        resources = [
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ]
        out = {r: 0 for r in resources}
        if 0 <= idx < len(resources):
            out[resources[idx]] = 1
        return out

    def _decode_gameplay(self, action_idx: int, env: CatanEnv) -> dict:
        legal_actions = env.get_legal_actions()
        phase = env.get_phase()

        if phase == TurnPhase.END_TURN:
            return {"type": "end_turn"}

        if not legal_actions:
            return {"type": "end_main_action"}

        if phase == TurnPhase.SETUP:
            mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
            return legal_actions[mapped_idx]

        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        chosen_action = legal_actions[mapped_idx]
        if chosen_action.get("type") == "discard_cards" and "resources" not in chosen_action:
            return self._resolve_discard_action(chosen_action)
        return chosen_action

    def _decode_trade(self, action_idx: int, env: CatanEnv) -> dict:
        if not self.enable_trading:
            phase = env.get_phase()
            if phase == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            if env.get_phase() == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}

        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        return legal_actions[mapped_idx]

    def collect(self, policy, steps: int = 128) -> List[Dict[str, Any]]:
        storage: List[Dict[str, Any]] = []

        for _ in range(steps):
            for i, env in enumerate(self.envs):
                phase_name = self._phase_name(env)

                if phase_name == "auto":
                    _, reward, done, info = env.step(None)
                    if done:
                        self.obs[i] = env.reset()
                    else:
                        self.obs[i] = env.get_observation()
                    continue

                obs = self._build_obs(env)
                value, action_dict, log_prob_dict, tom_outputs = policy.act(
                    obs=obs,
                    phase=phase_name,
                    deterministic=False,
                )

                if phase_name == "gameplay":
                    env_action = self._decode_gameplay(int(action_dict["gameplay_action"].item()), env)
                else:
                    env_action = self._decode_trade(int(action_dict["trade_action"].item()), env)

                _, reward, done, info = env.step(env_action)

                storage.append(
                    {
                        "obs": {k: v.detach().cpu().clone() for k, v in obs.items()},
                        "phase": phase_name,
                        "action": {k: v.detach().cpu().clone() for k, v in action_dict.items()},
                        "log_prob": {k: v.detach().cpu().clone() for k, v in log_prob_dict.items()},
                        "value": value.detach().cpu().clone(),
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                        "env_action": env_action,
                    }
                )

                if done:
                    self.obs[i] = env.reset()
                else:
                    self.obs[i] = env.get_observation()

        return storage
