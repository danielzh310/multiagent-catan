from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.constants import PlayerId, Resource, DevCard, COST_BUILD_ROAD, COST_BUILD_SETTLEMENT, COST_BUILD_CITY, COST_BUY_DEV_CARD, cost_to_dict
from core.engine import CatanEngine
from core.phase_router import ControllerType, TurnPhase

_ACTION_TYPE_TO_IDX: Dict[str, int] = {
    "build_settlement": 0, "build_road": 1, "build_city": 2,
    "buy_dev_card": 3, "play_dev_card": 4, "bank_trade": 5,
    "move_robber": 6, "discard_cards": 7, "end_main_action": 8,
    "end_turn": 9, "roll": 10, "skip_trade": 11,
}

_TRADE_ACTION_TYPE_TO_IDX: Dict[str, int] = {
    "skip_trade": 0, "propose_trade": 1, "accept_trade": 2,
    "reject_trade": 3, "counter_trade": 4,
}

_RESOURCE_NAMES = ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
_RESOURCE_TO_SLOT = {r: i for i, r in enumerate(_RESOURCE_NAMES)}


class CatanEnv:
    MAX_DISCARD_ACTIONS = 64
    MAX_GAMEPLAY_ACTIONS = 256
    GAMEPLAY_FEATURE_DIM = 40
    MAX_TRADE_ACTIONS = 128
    TRADE_FEATURE_DIM = 32

    def __init__(
        self,
        seed: Optional[int] = None,
        enable_trading: bool = True,
        max_steps: int | None = None,
        opponent_policies: Optional[List[Any]] = None,
        training_player_id: PlayerId = PlayerId.WHITE,
    ):
        self.engine = CatanEngine(seed=seed, enable_trading=enable_trading)
        self.max_steps = max_steps
        self.step_count = 0
        self.opponent_policies = opponent_policies or []
        self.training_player_id = training_player_id
        self.opponent_policy_map: Dict[PlayerId, Any] = {}

    def reset(self) -> dict:
        self.engine.reset()
        self.step_count = 0
        self._assign_opponent_policies()
        return self.get_observation()

    def update_opponent_policies(self, new_policies: List[Any]):
        """Updates the list of opponent policies and re-assigns them."""
        self.opponent_policies = new_policies or []
        self._assign_opponent_policies()

    def step(self, action: Optional[dict]):
        self.step_count += 1
        acting_player_id = self.engine.get_current_player_id()

        if action is None and acting_player_id != self.training_player_id:
            action = self._select_opponent_action()

        obs, reward, done, info = self.engine.step(action)

        # Convert opponent rewards into the training player's perspective.
        if acting_player_id != self.training_player_id:
            if done:
                reward = 1.0 if self.engine.winner == self.training_player_id else -1.0
            else:
                reward = 0.0

        # Check for truncation. The game is 'done' if there's a winner or it's truncated.
        is_truncated = False
        if self.max_steps is not None and self.step_count >= self.max_steps:
            is_truncated = True

        final_done = done or is_truncated
        info["truncated"] = is_truncated

        decision = self.engine.phase_router.get_controller(self.engine)
        obs["controller"] = {
            "phase": decision.phase,
            "controller": decision.controller,
            "acting_player": decision.acting_player,
            "target_player": decision.target_player,
        }
        obs["legal_actions"] = self.get_legal_actions()

        return obs, reward, final_done, info

    def get_observation(self) -> dict:
        obs = self.engine.get_observation()
        decision = self.engine.phase_router.get_controller(self.engine)

        obs["controller"] = {
            "phase": decision.phase,
            "controller": decision.controller,
            "acting_player": decision.acting_player,
            "target_player": decision.target_player,
        }

        obs["legal_actions"] = self.get_legal_actions()
        return obs

    def _assign_opponent_policies(self) -> None:
        non_training_players = [pid for pid in self.engine.player_order if pid != self.training_player_id]
        self.opponent_policy_map = {}
        if not self.opponent_policies:
            return

        for index, player_id in enumerate(non_training_players):
            self.opponent_policy_map[player_id] = self.opponent_policies[index % len(self.opponent_policies)]

    def _phase_name(self, phase: TurnPhase) -> str:
        if phase in (TurnPhase.SETUP, TurnPhase.MAIN_ACTION, TurnPhase.END_TURN):
            return "gameplay"
        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            return "trade"
        return "auto"

    def _to_vec(self, state: dict) -> np.ndarray:
        resources = state.get("resources", {})
        roads_val = state.get("roads", 0)
        if isinstance(roads_val, list):
            roads_val = len(roads_val)
        dev_cards_val = state.get("dev_cards", 0)
        if isinstance(dev_cards_val, list):
            dev_cards_val = len(dev_cards_val)
        vec = np.array([
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
        ], dtype=np.float32)
        padded = np.zeros(64, dtype=np.float32)
        padded[: len(vec)] = vec
        return padded

    def _build_policy_obs(self) -> Dict[str, torch.Tensor]:
        obs = self.get_observation()
        current_player = self.engine.get_current_player_id()
        player = self.engine.players[current_player]
        other_players = [v for k, v in obs["players"].items() if k != current_player]
        
        # Construct player_stats dictionary in the same format as RolloutManagers
        player_stats = {
            "vp": float(player.update_victory_points()),
            "n_settlements": float(player.n_settlements),
            "n_cities": float(player.n_cities),
            "n_roads": float(player.n_roads),
            "resources": {str(k): float(v) for k, v in player.resources.items()},
            "resource_total": float(sum(int(v) for v in player.resources.values())),
        }

        self_vec = self._to_vec(obs["player"])
        if other_players:
            opponent_vec = np.mean([self._to_vec(opp) for opp in other_players], axis=0).astype(np.float32)
        else:
            opponent_vec = np.zeros(64, dtype=np.float32)

        resource_bank = obs["game"].get("resource_bank", {})
        bank_vec = np.zeros(5, dtype=np.float32)
        for idx, res in enumerate(["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]):
            bank_vec[idx] = float(resource_bank.get(res, 0)) / 19.0

        pending = self.get_pending_trade()
        pending_vec = np.zeros(4, dtype=np.float32)
        if pending is not None:
            pending_vec[0] = 1.0
            pending_vec[1] = float(pending.counter_count) / 3.0
            pending_vec[2] = float(int(pending.proposer)) / 3.0
            pending_vec[3] = float(int(pending.target)) / 3.0

        opponent_flat = np.zeros(192, dtype=np.float32)
        opponent_vectors = []
        for opp in other_players[:3]:
            opponent_vectors.append(self._to_vec(opp))
        while len(opponent_vectors) < 3:
            opponent_vectors.append(np.zeros(64, dtype=np.float32))
        opponent_flat = np.concatenate(opponent_vectors, axis=0)

        global_state = np.concatenate([self_vec, opponent_flat, bank_vec, pending_vec], axis=0)

        board_np = np.zeros(64, dtype=np.float32)
        board_np[0] = float(obs["game"].get("turn_number", 0))
        board_np[1] = float(int(obs["game"].get("current_player", 0)))
        board_np[2] = float(obs["game"].get("phase", 0).value if hasattr(obs["game"].get("phase", 0), "value") else obs["game"].get("phase", 0))
        board_np[3] = 1.0 if obs["game"].get("enable_trading", True) else 0.0
        board_np[4] = float(obs["game"].get("last_roll", 0) or 0)
        board_np[5] = 1.0 if obs["game"].get("robber_pending", False) else 0.0
        board_np[6] = 1.0 if obs["game"].get("initial_placement_phase", False) else 0.0
        board_np[7] = float(obs["game"].get("initial_placement_index", 0))
        board_np[8] = 1.0 if obs["game"].get("initial_placement_stage") == "settlement" else 0.0
        board_np[9] = 1.0 if obs["game"].get("initial_placement_stage") == "road" else 0.0
        board_np[10] = float(obs["game"].get("dev_card_deck_size", 0))
        board_np[11] = float(obs["game"].get("longest_road_owner") == current_player)
        board_np[12] = float(obs["game"].get("largest_army_owner") == current_player)
        board_np[13] = float(len(obs.get("robber_discard_queue", [])))
        board_np[14] = float(obs.get("robber_discard_required", {}).get(current_player, 0))

        gameplay_candidates, gameplay_mask = self._encode_gameplay_actions(
            obs["legal_actions"], player, pending, self.MAX_GAMEPLAY_ACTIONS, self.GAMEPLAY_FEATURE_DIM
        )
        trade_candidates, trade_mask = self._encode_trade_actions(
            obs["legal_actions"], player, pending, self.MAX_TRADE_ACTIONS, self.TRADE_FEATURE_DIM
        )

        return {
            "board": torch.from_numpy(board_np).unsqueeze(0),
            "self": torch.from_numpy(self_vec).unsqueeze(0),
            "opponent": torch.from_numpy(opponent_vec).unsqueeze(0),
            "global_state": torch.from_numpy(global_state).unsqueeze(0),
            "gameplay_candidates": torch.from_numpy(gameplay_candidates).unsqueeze(0),
            "gameplay_mask": torch.from_numpy(gameplay_mask).unsqueeze(0),
            "trade_candidates": torch.from_numpy(trade_candidates).unsqueeze(0),
            "trade_mask": torch.from_numpy(trade_mask).unsqueeze(0),
        }

    def _encode_gameplay_actions(
        self,
        actions: List[dict], # Renamed argument from 'player' to 'player_stats'
        player_stats: Dict, # Changed type hint from Any to Dict
        pending_info: Optional[Any],
        max_actions: int,
        feature_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros((max_actions, feature_dim), dtype=np.float32)
        mask = np.zeros(max_actions, dtype=bool)

        if not actions:
            mask[0] = True
            return out, mask

        vp = player_stats["vp"]
        n_settlements = player_stats["n_settlements"]
        n_cities = player_stats["n_cities"]
        n_roads = player_stats["n_roads"]

        for i, action in enumerate(actions[:max_actions]):
            f = out[i]
            f[ _ACTION_TYPE_TO_IDX.get(action.get("type", ""), 0) ] = 1.0
            f[12] = vp / 10.0
            f[13] = n_settlements / 5.0
            f[14] = n_cities / 4.0
            f[15] = n_roads / 15.0

            vertex = action.get("vertex")
            if vertex is not None:
                f[16] = float(vertex) / 64.0
            connection = action.get("connection")
            if connection is not None:
                f[17] = float(connection) / 128.0
            tile = action.get("tile")
            if tile is not None:
                f[18] = float(tile) / 19.0
            action_1 = action.get("connection_1")
            if action_1 is not None:
                f[19] = float(action_1) / 128.0
            action_2 = action.get("connection_2")
            if action_2 is not None:
                f[20] = float(action_2) / 128.0

            give_slot = self._resource_slot(action.get("give"))
            recv_slot = self._resource_slot(action.get("receive"))
            res_slot = self._resource_slot(action.get("resource"))
            res1_slot = self._resource_slot(action.get("resource_1"))
            res2_slot = self._resource_slot(action.get("resource_2"))

            if give_slot >= 0:
                f[21 + give_slot] = 1.0
            if recv_slot >= 0:
                f[26 + recv_slot] = 1.0
            if res_slot >= 0:
                f[31] = float(res_slot + 1) / 5.0
            if res1_slot >= 0:
                f[32] = float(res1_slot + 1) / 5.0
            if res2_slot >= 0:
                f[33] = float(res2_slot + 1) / 5.0

            card = action.get("card")
            if card is not None:
                try:
                    f[34] = float(int(card) + 1) / 5.0
                except (ValueError, TypeError):
                    pass

            rate = action.get("rate")
            if rate is not None:
                f[35] = float(rate) / 4.0
            required = action.get("required")
            if required is not None:
                f[36] = float(required) / 8.0

            discard = action.get("resources")
            if isinstance(discard, dict):
                vals = list(discard.values())
                total = float(sum(int(v) for v in vals))
                non_zero = float(sum(1 for v in vals if int(v) > 0))
                f[37] = total / 8.0
                f[38] = non_zero / 5.0

            if action.get("type") == "play_dev_card":
                f[39] = 1.0

            mask[i] = True

        return out, mask

    def _encode_trade_actions(
        self,
        actions: List[dict], # Renamed argument from 'player' to 'player_stats'
        player_stats: Dict, # Changed type hint from Any to Dict
        pending_info: Optional[Any],
        max_actions: int,
        feature_dim: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros((max_actions, feature_dim), dtype=np.float32)
        mask = np.zeros(max_actions, dtype=bool)

        if not actions:
            mask[0] = True
            return out, mask

        vp = player_stats["vp"]
        res_total = player_stats["resource_total"]
        has_pending = 1.0 if pending_info is not None else 0.0
        p_counter = float(pending_info["counter_count"]) if pending_info else 0.0
        p_proposer = float(pending_info["proposer"]) if pending_info else 0.0
        p_target = float(pending_info["target"]) if pending_info else 0.0

        for i, action in enumerate(actions[:max_actions]):
            f = out[i]
            f[ _TRADE_ACTION_TYPE_TO_IDX.get(action.get("type", ""), 0) ] = 1.0
            f[5] = vp / 10.0
            f[6] = res_total / 20.0
            f[7] = has_pending
            if has_pending:
                f[8] = p_counter / 3.0
                f[9] = p_proposer / 3.0
                f[10] = p_target / 3.0

            target = action.get("target")
            if target is not None:
                try:
                    f[11 + int(target)] = 1.0
                except (ValueError, TypeError, IndexError):
                    pass

            offer = action.get("offer") or action.get("counter_offer") or {}
            request = action.get("request") or action.get("counter_request") or {}
            offer_total = 0.0
            for resource, amount in offer.items():
                slot = self._resource_slot(resource)
                if slot >= 0:
                    v = float(amount)
                    f[15 + slot] = v
                    offer_total += v
            request_total = 0.0
            for resource, amount in request.items():
                slot = self._resource_slot(resource)
                if slot >= 0:
                    v = float(amount)
                    f[20 + slot] = v
                    request_total += v

            f[25] = offer_total / 4.0
            f[26] = request_total / 4.0

            if action.get("response_type") == "accept":
                f[27] = 1.0
            elif action.get("response_type") == "reject":
                f[28] = 1.0
            elif action.get("response_type") == "counter":
                f[29] = 1.0

            if action.get("type") == "counter_trade":
                f[30] = 1.0
            if action.get("type") == "propose_trade":
                f[31] = 1.0

            mask[i] = True

        return out, mask

    def _resource_slot(self, resource_value: Any) -> int:
        try:
            name = resource_value.name if hasattr(resource_value, "name") else str(resource_value)
            return _RESOURCE_TO_SLOT.get(name, -1)
        except Exception:
            return -1

    def _unwrap_action_dict(self, action_dict: Dict[str, Any]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for key, value in action_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
                if value.size == 0:
                    continue
                result[key] = int(value.reshape(-1)[0])
            elif isinstance(value, list):
                result[key] = int(value[0]) if value else -1
            else:
                result[key] = int(value)
        return result

    def _select_opponent_action(self) -> Optional[dict]:
        phase = self.engine.phase_router.get_phase()
        if phase == TurnPhase.ROLL:
            return None

        current_player = self.engine.get_current_player_id()
        policy = self.opponent_policy_map.get(current_player)
        legal_actions = self.get_legal_actions()
        if not legal_actions:
            return None

        if policy is None:
            return random.choice(legal_actions)

        obs = self._build_policy_obs()
        policy_phase = self._phase_name(phase)
        try:
            result = policy.act(obs, policy_phase, deterministic=True)
        except TypeError:
            result = policy.act(obs, policy_phase)

        if isinstance(result, tuple) and len(result) > 1:
            action_dict = result[1]
        elif isinstance(result, dict):
            action_dict = result
        else:
            action_dict = None

        if not isinstance(action_dict, dict):
            return random.choice(legal_actions)

        action_dict = self._unwrap_action_dict(action_dict)
        if policy_phase == "gameplay":
            action_idx = action_dict.get("gameplay_action", action_dict.get("action", -1))
            return self._decode_gameplay(int(action_idx), legal_actions, phase)
        return self._decode_trade(int(action_dict.get("trade_action", -1)), legal_actions, phase)

    def _decode_gameplay(self, action_idx: int, legal_actions: List[Dict], phase: TurnPhase) -> dict:
        if phase == TurnPhase.END_TURN:
            return {"type": "end_turn"}
        if not legal_actions:
            return {"type": "end_main_action"}
        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        chosen_action = legal_actions[mapped_idx]
        if chosen_action.get("type") == "discard_cards" and "resources" not in chosen_action:
            return self._resolve_discard_action(chosen_action)
        return chosen_action

    def _decode_trade(self, action_idx: int, legal_actions: List[Dict], phase: TurnPhase) -> dict:
        if not self.engine.enable_trading:
            if phase == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}
        if not legal_actions:
            if phase == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}
        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        return legal_actions[mapped_idx]

    def get_current_player_id(self) -> PlayerId:
        return self.engine.get_current_player_id()

    def get_step_count(self) -> int:
        return self.step_count

    def get_phase(self) -> TurnPhase:
        return self.engine.phase_router.get_phase()

    def get_controller_type(self) -> ControllerType:
        decision = self.engine.phase_router.get_controller(self.engine)
        return decision.controller

    def get_trade_history(self) -> dict:
        return self.engine.trade_history.build_sequence_tensor_dict()

    def get_pending_trade(self):
        return self.engine.trade_manager.get_pending_trade()

    def get_last_roll(self):
        return self.engine.last_roll

    def get_last_robber_event(self):
        return self.engine.last_robber_event

    def get_legal_actions(self) -> List[dict]:
        phase = self.engine.phase_router.get_phase()
        current_player = self.engine.get_current_player_id()

        if self.engine.winner is not None:
            return []

        if phase == TurnPhase.SETUP:
            player_id = current_player
            if self.engine.initial_placement_stage == "settlement":
                valid_settlements = self.engine.get_valid_settlement_vertices(player_id, require_road=False)
                if valid_settlements:
                    return [{"type": "build_settlement", "vertex": v} for v in valid_settlements]
                return []
            else:
                valid_roads = self.engine.get_valid_road_connections(player_id)
                if valid_roads:
                    return [{"type": "build_road", "connection": c} for c in valid_roads]
                return []

        if phase == TurnPhase.ROLL:
            return [{"type": "roll"}]

        if phase == TurnPhase.MAIN_ACTION:
            if self.engine.robber_pending:
                if self.engine.robber_discard_queue:
                    return self._get_legal_discard_actions(current_player)
                current_robber_tile = next((t.id for t in self.engine.board.tiles if t.has_robber), None)
                moves = []
                for t in self.engine.board.tiles:
                    if t.id != current_robber_tile:
                        moves.append({"type": "move_robber", "tile": t.id})
                return moves
            return self._get_legal_gameplay_actions(current_player)

        if phase == TurnPhase.TRADE_PROPOSE:
            if not self.engine.enable_trading:
                return [{"type": "skip_trade"}]
            return self._get_legal_trade_proposals(current_player)

        if phase == TurnPhase.TRADE_RESPOND:
            if not self.engine.enable_trading:
                return [{"type": "reject_trade", "response_type": "reject"}]
            return self._get_legal_trade_responses(current_player)

        if phase == TurnPhase.END_TURN:
            return [{"type": "end_turn"}]

        return []

    def _get_legal_gameplay_actions(self, player_id: PlayerId) -> List[dict]:
        player = self.engine.players[player_id]
        actions: List[dict] = []

        valid_settlements = self.engine.get_valid_settlement_vertices(player_id)
        if (
            player.n_settlements < 5
            and player.can_pay_cost(COST_BUILD_SETTLEMENT)
            and valid_settlements
        ):
            for vertex_id in valid_settlements:
                actions.append({"type": "build_settlement", "vertex": vertex_id})

        valid_roads = self.engine.get_valid_road_connections(player_id)
        if (
            player.n_settlements > 0
            and player.n_roads < 15
            and player.can_pay_cost(COST_BUILD_ROAD)
            and valid_roads
        ):
            for conn_id in valid_roads:
                actions.append({"type": "build_road", "connection": conn_id})

        if player.n_settlements > 0 and player.n_cities < 4 and player.can_pay_cost(COST_BUILD_CITY):
            for vertex_id in self.engine.settlement_positions[player_id]:
                actions.append({"type": "build_city", "vertex": vertex_id})

        if player.can_pay_cost(COST_BUY_DEV_CARD) and self.engine.dev_card_deck:
            actions.append({"type": "buy_dev_card"})

        if player.can_play_dev_card(DevCard.KNIGHT):
            current_robber_tile = next((t.id for t in self.engine.board.tiles if t.has_robber), None)
            for tile in self.engine.board.tiles:
                if tile.id != current_robber_tile:
                    actions.append({"type": "play_dev_card", "card": int(DevCard.KNIGHT), "tile": tile.id})
        if player.can_play_dev_card(DevCard.ROAD_BUILDING):
            actions.extend(self._get_road_building_actions(player_id))
        if player.can_play_dev_card(DevCard.INVENTION):
            resources = [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]
            for resource_1 in resources:
                for resource_2 in resources:
                    actions.append({
                        "type": "play_dev_card",
                        "card": int(DevCard.INVENTION),
                        "resource_1": int(resource_1),
                        "resource_2": int(resource_2),
                    })
        if player.can_play_dev_card(DevCard.MONOPOLY):
            for resource in (Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE):
                actions.append({
                    "type": "play_dev_card",
                    "card": int(DevCard.MONOPOLY),
                    "resource": int(resource),
                })

        actions.extend(self._get_legal_bank_trades(player_id))
        actions.append({"type": "end_main_action"})
        return actions

    def _get_legal_bank_trades(self, player_id: PlayerId) -> List[dict]:
        player = self.engine.players[player_id]
        actions: List[dict] = []

        resources = [
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ]

        for give in resources:
            rate = self.engine._best_maritime_rate(player_id, give)
            if player.resources.get(give, 0) < rate:
                continue
            for receive in resources:
                if give == receive:
                    continue
                actions.append({
                    "type": "bank_trade",
                    "give": give,
                    "receive": receive,
                    "rate": rate,
                })

        return actions

    def _get_road_building_actions(self, player_id: PlayerId) -> List[dict]:
        actions: List[dict] = []
        first_roads = self.engine.get_valid_road_connections(player_id)
        if not first_roads:
            return actions

        seen = set()
        for connection_1 in first_roads:
            action_single = {
                "type": "play_dev_card",
                "card": int(DevCard.ROAD_BUILDING),
                "connection_1": connection_1,
                "connection_2": None,
            }
            key_single = (connection_1, None)
            if key_single not in seen:
                actions.append(action_single)
                seen.add(key_single)

            simulated_roads = {
                pid: set(roads)
                for pid, roads in self.engine.road_positions.items()
            }
            simulated_roads[player_id].add(connection_1)

            second_roads = self.engine.board.get_valid_road_connections(
                player_id,
                self.engine.settlement_positions,
                simulated_roads,
                city_positions=self.engine.city_positions,
            )

            for connection_2 in second_roads:
                if connection_2 == connection_1:
                    continue
                key_double = (connection_1, connection_2)
                if key_double in seen:
                    continue
                actions.append({
                    "type": "play_dev_card",
                    "card": int(DevCard.ROAD_BUILDING),
                    "connection_1": connection_1,
                    "connection_2": connection_2,
                })
                seen.add(key_double)

        return actions

    def _get_legal_discard_actions(self, player_id: PlayerId) -> List[dict]:
        required = int(self.engine.robber_discard_required.get(player_id, 0))
        if required <= 0:
            return []

        player = self.engine.players[player_id]
        resources = [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE]
        available = {resource: int(player.resources.get(resource, 0)) for resource in resources}
        combinations: List[dict] = []

        def build_combo(index: int, remaining: int, current: dict) -> None:
            if len(combinations) >= self.MAX_DISCARD_ACTIONS:
                return
            if index == len(resources):
                if remaining == 0:
                    combinations.append({
                        "type": "discard_cards",
                        "required": required,
                        "available": {resource.name: available[resource] for resource in resources},
                        "resources": dict(current),
                    })
                return

            resource = resources[index]
            max_take = min(available[resource], remaining)
            for amount in range(max_take, -1, -1):
                current[resource] = amount
                build_combo(index + 1, remaining - amount, current)
                if len(combinations) >= self.MAX_DISCARD_ACTIONS:
                    return
            current.pop(resource, None)

        build_combo(0, required, {})

        if combinations:
            return combinations

        return [{
            "type": "discard_cards",
            "required": required,
            "available": {resource.name: available[resource] for resource in resources},
        }]

    def _can_afford_resources(self, resources: dict, cost: tuple) -> bool:
        cost_dict = cost_to_dict(cost)
        for resource, amount in cost_dict.items():
            if resources.get(resource, 0) < amount:
                return False
        return True

    def _get_legal_trade_proposals(self, player_id: PlayerId) -> List[dict]:
        actions: List[dict] = [{"type": "skip_trade"}]

        player_ids = list(self.engine.players.keys())
        targets = self.engine.trade_manager.legal_trade_targets(player_id, player_ids)

        trade_templates = self._default_trade_templates()

        for target in targets:
            for offer, request in trade_templates:
                proposer_state = self.engine.players[player_id]
                if self.engine.trade_manager.can_player_afford(proposer_state, offer):
                    actions.append({
                        "type": "propose_trade",
                        "target": target,
                        "offer": offer,
                        "request": request,
                    })

        return actions

    def _get_legal_trade_responses(self, player_id: PlayerId) -> List[dict]:
        pending = self.engine.trade_manager.get_pending_trade()
        if pending is None:
            return []

        actions: List[dict] = [
            {"type": "reject_trade", "response_type": "reject"},
            {"type": "accept_trade", "response_type": "accept"},
        ]

        responder_state = self.engine.players[player_id]
        response_templates = self._default_trade_templates()

        for counter_offer, counter_request in response_templates:
            if self.engine.trade_manager.can_player_afford(responder_state, counter_offer):
                actions.append({
                    "type": "counter_trade",
                    "response_type": "counter",
                    "counter_offer": counter_offer,
                    "counter_request": counter_request,
                })

        return actions

    def _default_trade_templates(self):
        singles: List[dict] = [
            {Resource.WOOD: 1, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 1, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 1, Resource.WHEAT: 0, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 1, Resource.ORE: 0},
            {Resource.WOOD: 0, Resource.BRICK: 0, Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 1},
        ]

        templates = []
        for offer in singles:
            for request in singles:
                if offer != request:
                    templates.append((offer, request))

        return templates

    def build_gameplay_observation(self) -> dict:
        obs = self.get_observation()
        current_player = self.get_current_player_id()

        return {
            "turn_number": obs["game"]["turn_number"],
            "phase": obs["game"]["phase"],
            "current_player": current_player,
            "initial_placement_phase": obs["game"].get("initial_placement_phase", False),
            "initial_placement_index": obs["game"].get("initial_placement_index", 0),
            "initial_placement_stage": obs["game"].get("initial_placement_stage"),
            "last_roll": obs["game"].get("last_roll"),
            "robber_pending": obs["game"].get("robber_pending", False),
            "last_robber_event": obs["game"].get("last_robber_event"),
            "self_state": obs["player"],
            "all_players": obs["players"],
            "trade_history": obs["trade_history"],
            "legal_actions": obs["legal_actions"],
        }

    def build_trade_observation(self) -> dict:
        obs = self.get_observation()
        current_player = self.get_current_player_id()

        return {
            "turn_number": obs["game"]["turn_number"],
            "phase": obs["game"]["phase"],
            "current_player": current_player,
            "initial_placement_phase": obs["game"].get("initial_placement_phase", False),
            "initial_placement_index": obs["game"].get("initial_placement_index", 0),
            "initial_placement_stage": obs["game"].get("initial_placement_stage"),
            "last_roll": obs["game"].get("last_roll"),
            "robber_pending": obs["game"].get("robber_pending", False),
            "last_robber_event": obs["game"].get("last_robber_event"),
            "self_state": obs["player"],
            "all_players": obs["players"],
            "pending_trade": obs["trade"],
            "trade_history": obs["trade_history"],
            "legal_actions": obs["legal_actions"],
        }

    def get_active_model_name(self) -> str:
        controller = self.get_controller_type()

        if controller == ControllerType.GAMEPLAY:
            return "gameplay"
        if controller == ControllerType.TRADE:
            return "trade"
        return "none"
