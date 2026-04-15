import argparse
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np

from core.constants import Resource, DevCard
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv
from learning.ppo.ppo_policy import PPOPolicy
from learning.unified.unified_policy import UnifiedPolicy
from learning.tom_dqn.tom_dqn_policy import ToMEnhancedDQNPolicy
from learning.dqn.dqn_policy import DQNBaselinePolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Play one game from a checkpoint and print actions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy deterministic actions")
    parser.add_argument("--max-steps", type=int, default=2000, help="Maximum simulation steps")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for environment")
    parser.add_argument("--show-obs", action="store_true", help="Print raw observation vectors each step")
    parser.add_argument("--show-board", action="store_true", help="Print board state each step")
    parser.add_argument("--gameplay-only", action="store_true", help="Disable trading and only evaluate gameplay")
    parser.add_argument("--policy-type", type=str, default="unified", choices=["unified", "dqn", "ppo", "tom_dqn"], help="Type of policy to use")
    return parser.parse_args()


# Constants for observation/action feature dimensions
MAX_GAMEPLAY_ACTIONS = 256
GAMEPLAY_FEATURE_DIM = 40
MAX_TRADE_ACTIONS = 128
TRADE_FEATURE_DIM = 32


def _resource_slot(resource_value: Any) -> int:
    if resource_value is None:
        return -1
    try:
        return int(Resource(resource_value))
    except (ValueError, TypeError):
        return -1


def _dev_card_slot(card_value: Any) -> int:
    if card_value is None:
        return -1
    try:
        return int(DevCard(card_value))
    except (ValueError, TypeError):
        return -1


# =============================================================================
# Action encoding helpers (from unified rollout manager)
# =============================================================================

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

def _res_slot(r: Any) -> int:
    if r is None:
        return -1
    try:
        name = r.name if hasattr(r, "name") else str(r)
        return _RESOURCE_TO_SLOT.get(name, -1)
    except Exception:
        return -1


def _encode_gameplay_actions_batch(
    actions: List[Dict[str, Any]],
    player_stats: Dict[str, Any],
    max_actions: int,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(actions), max_actions)
    out = np.zeros((max_actions, feature_dim), dtype=np.float32)
    mask = np.zeros(max_actions, dtype=bool)

    if n == 0:
        mask[0] = True
        return out, mask

    vp = player_stats["vp"]
    n_settlements = player_stats.get("n_settlements", 0)
    n_cities = player_stats.get("n_cities", 0)
    n_roads = player_stats.get("n_roads", 0)

    for i in range(n):
        a = actions[i]
        f = out[i]
        atype = a.get("type", ""); tidx = _ACTION_TYPE_TO_IDX.get(atype)
        if tidx is not None: f[tidx] = 1.0
        f[12] = vp / 10.0; f[13] = n_settlements / 5.0; f[14] = n_cities / 4.0; f[15] = n_roads / 15.0
        v = a.get("vertex"); f[16] = float(v) / 64.0 if v is not None else 0.0
        c = a.get("connection"); f[17] = float(c) / 128.0 if c is not None else 0.0
        t = a.get("tile"); f[18] = float(t) / 19.0 if t is not None else 0.0
        c1 = a.get("connection_1"); f[19] = float(c1) / 128.0 if c1 is not None else 0.0
        c2 = a.get("connection_2"); f[20] = float(c2) / 128.0 if c2 is not None else 0.0
        give_slot = _res_slot(a.get("give")); recv_slot = _res_slot(a.get("receive"))
        res_slot  = _res_slot(a.get("resource")); res1_slot = _res_slot(a.get("resource_1")); res2_slot = _res_slot(a.get("resource_2"))
        if give_slot >= 0: f[21 + give_slot] = 1.0
        if recv_slot >= 0: f[26 + recv_slot] = 1.0
        if res_slot >= 0: f[31] = float(res_slot + 1) / 5.0
        if res1_slot >= 0: f[32] = float(res1_slot + 1) / 5.0
        if res2_slot >= 0: f[33] = float(res2_slot + 1) / 5.0
        card = a.get("card")
        if card is not None:
            try: f[34] = float(int(card) + 1) / 5.0
            except (ValueError, TypeError): pass
        rate = a.get("rate"); f[35] = float(rate) / 4.0 if rate is not None else 0.0
        req = a.get("required"); f[36] = float(req) / 8.0 if req is not None else 0.0
        discard = a.get("resources")
        if isinstance(discard, dict):
            vals = list(discard.values()); total = float(sum(int(v) for v in vals)); non_zero = float(sum(1 for v in vals if int(v) > 0))
            f[37] = total / 8.0; f[38] = non_zero / 5.0
        if atype == "play_dev_card": f[39] = 1.0
        mask[i] = True
    return out, mask

def _encode_trade_actions_batch(
    actions: List[Dict[str, Any]], player_stats: Dict[str, Any], pending_info: Optional[Dict], max_actions: int, feature_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(actions), max_actions)
    out = np.zeros((max_actions, feature_dim), dtype=np.float32); mask = np.zeros(max_actions, dtype=bool)
    if n == 0: mask[0] = True; return out, mask
    vp = player_stats["vp"]; res_total = player_stats["resource_total"]
    p_counter = float(pending_info["counter_count"]) if pending_info else 0.0; p_proposer = float(pending_info["proposer"]) if pending_info else 0.0
    p_target = float(pending_info["target"]) if pending_info else 0.0; has_pending = 1.0 if pending_info is not None else 0.0
    for i in range(n):
        a = actions[i]; f = out[i]; atype = a.get("type", ""); tidx = _TRADE_ACTION_TYPE_TO_IDX.get(atype)
        if tidx is not None: f[tidx] = 1.0
        f[5] = vp / 10.0; f[6] = res_total / 20.0; f[7] = has_pending
        if has_pending: f[8] = p_counter / 3.0; f[9] = p_proposer / 3.0; f[10] = p_target / 3.0
        target = a.get("target")
        if target is not None:
            try: f[11 + int(target)] = 1.0
            except (ValueError, TypeError, IndexError): pass
        offer = a.get("offer") or a.get("counter_offer") or {}; request = a.get("request") or a.get("counter_request") or {}
        offer_total = 0.0
        for r, amount in offer.items():
            if (slot := _res_slot(r)) >= 0: v = float(amount); f[15 + slot] = v; offer_total += v
        request_total = 0.0
        for r, amount in request.items():
            if (slot := _res_slot(r)) >= 0: v = float(amount); f[20 + slot] = v; request_total += v
        f[25] = offer_total / 4.0; f[26] = request_total / 4.0
        response_type = a.get("response_type", "")
        if response_type == "accept": f[27] = 1.0
        elif response_type == "reject": f[28] = 1.0
        elif response_type == "counter": f[29] = 1.0
        if atype == "counter_trade": f[30] = 1.0
        if atype == "propose_trade": f[31] = 1.0
        mask[i] = True
    return out, mask

# =============================================================================
# Observation/Action builders for Unified/DQN policies
# =============================================================================

# NOTE: The following functions are for the 'unified' and 'dqn' policy types.
# PPO-specific functions are defined separately below.
def build_policy_obs(env: CatanEnv, raw_obs: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    player_state = raw_obs["player"]
    other_players = [v for k, v in raw_obs["players"].items() if k != raw_obs["game"]["current_player"]]
    legal_actions = raw_obs.get("legal_actions", [])

    # Get player stats and pending info for action encoding
    current_player_id = env.get_current_player_id()
    player_engine_state = env.engine.players[current_player_id]
    player_stats = {
        "vp": float(player_engine_state.update_victory_points()),
        "n_settlements": float(player_engine_state.n_settlements),
        "n_cities": float(player_engine_state.n_cities),
        "n_roads": float(player_engine_state.n_roads),
        "resource_total": float(sum(int(v) for v in player_engine_state.resources.values())),
    }
    pending = env.engine.trade_manager.get_pending_trade()
    pending_info = None
    if pending is not None:
        pending_info = {
            "counter_count": float(pending.counter_count),
            "proposer": float(int(pending.proposer)),
            "target": float(int(pending.target)),
        }

    def to_vec(state: dict) -> List[float]:
        roads_val = state.get("roads", 0)
        if isinstance(roads_val, list):
            roads_val = len(roads_val)

        dev_cards_val = state.get("dev_cards", 0)
        if isinstance(dev_cards_val, list):
            dev_cards_val = len(dev_cards_val)

        vec = [
            float(state.get("resources", {}).get("WOOD", 0)), float(state.get("resources", {}).get("BRICK", 0)),
            float(state.get("resources", {}).get("SHEEP", 0)), float(state.get("resources", {}).get("WHEAT", 0)),
            float(state.get("resources", {}).get("ORE", 0)), float(state.get("victory_points", 0)),
            float(state.get("num_settlements", 0)), float(state.get("num_cities", 0)),
            float(roads_val), float(state.get("bonus_vp", 0)),
            float(state.get("dev_victory_points", 0)), float(dev_cards_val),
            float(state.get("played_knights", 0)), float(state.get("revealed_vp_cards", 0)),
        ]
        return (vec + [0.0] * 64)[:64]

    self_vec = torch.tensor([to_vec(player_state)], dtype=torch.float32, device=device)

    op_vec = [0.0] * 64
    if other_players:
        sum_vec = [sum(col) for col in zip(*(to_vec(opp) for opp in other_players))]
        op_vec = [x / len(other_players) for x in sum_vec]

    board_vec = torch.zeros((1, 64), dtype=torch.float32, device=device)
    game = raw_obs.get("game", {})
    board_vec[0, 0] = float(game.get("turn_number", 0))
    board_vec[0, 1] = float(int(env.get_current_player_id()))
    board_vec[0, 2] = float(env.get_phase().value)
    board_vec[0, 3] = 1.0 if game.get("enable_trading", True) else 0.0
    last_roll = game.get("last_roll")
    board_vec[0, 4] = float(last_roll if last_roll is not None else 0.0)
    board_vec[0, 5] = 1.0 if game.get("robber_pending", False) else 0.0
    if (pending_trade := raw_obs.get("trade")) is not None:
        board_vec[0, 6] = 1.0
        board_vec[0, 7] = float(pending_trade.counter_count)
    if (robber_event := game.get("last_robber_event")) is not None:
        board_vec[0, 8] = 1.0 if robber_event.get("rolled_seven", False) else 0.0
        board_vec[0, 9] = 1.0 if robber_event.get("stolen_from") is not None else 0.0
        board_vec[0, 10] = sum(sum(res.values()) for _, res in robber_event.get("discarded", {}).items())

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

    def build_global_state() -> List[float]:
        opponent_vectors = []
        for opp in other_players[:3]:
            opponent_vectors.append(to_vec(opp))
        while len(opponent_vectors) < 3:
            opponent_vectors.append([0.0] * 64)
        opponent_flat = [item for sublist in opponent_vectors for item in sublist]

        resource_bank = game.get("resource_bank", {})
        bank_vec = [float(resource_bank.get(res, 0)) / 19.0 for res in ["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]]

        pending_vec = [0.0] * 4
        if (pending_trade := raw_obs.get("trade")) is not None:
            pending_vec[0] = 1.0
            pending_vec[1] = float(pending_trade.counter_count) / 3.0
            pending_vec[2] = float(int(pending_trade.proposer)) / 3.0
            pending_vec[3] = float(int(pending_trade.target)) / 3.0

        return to_vec(player_state) + opponent_flat + bank_vec + pending_vec

    global_vec = torch.tensor([build_global_state()], dtype=torch.float32, device=device)

    gameplay_candidates_np, gameplay_mask_np = _encode_gameplay_actions_batch(
        legal_actions, player_stats, MAX_GAMEPLAY_ACTIONS, GAMEPLAY_FEATURE_DIM
    )
    trade_candidates_np, trade_mask_np = _encode_trade_actions_batch(
        legal_actions, player_stats, pending_info, MAX_TRADE_ACTIONS, TRADE_FEATURE_DIM
    )

    gameplay_candidates = torch.from_numpy(gameplay_candidates_np).unsqueeze(0).to(device)
    gameplay_mask = torch.from_numpy(gameplay_mask_np).unsqueeze(0).to(device)
    trade_candidates = torch.from_numpy(trade_candidates_np).unsqueeze(0).to(device)
    trade_mask = torch.from_numpy(trade_mask_np).unsqueeze(0).to(device)

    return {
        "board": board_vec,
        "self": self_vec,
        "opponent": torch.tensor([op_vec], dtype=torch.float32, device=device),
        "global_state": global_vec,
        "gameplay_candidates": gameplay_candidates,
        "gameplay_mask": gameplay_mask,
        "trade_candidates": trade_candidates,
        "trade_mask": trade_mask,
    }


# =============================================================================
# Observation/Action builders for PPO policy
# (Copied from learning/ppo/ppo_rollout_manager.py for consistency)
# =============================================================================

def _encode_ppo_gameplay_action(action: Dict[str, Any], phase: TurnPhase, player_stats: Dict) -> List[float]:
    features = [0.0] * GAMEPLAY_FEATURE_DIM
    action_type = action.get("type", "")
    action_types = {
        "build_settlement": 0, "build_road": 1, "build_city": 2,
        "buy_dev_card": 3, "play_dev_card": 4, "bank_trade": 5,
        "move_robber": 6, "discard_cards": 7, "end_main_action": 8, "end_turn": 9
    }
    if (action_type_idx := action_types.get(action_type)) is not None:
        features[action_type_idx] = 1.0

    if "connection" in action: features[10] = float(action["connection"]) / 128.0
    if "tile" in action: features[11] = float(action["tile"]) / 19.0
    if "connection_1" in action and action["connection_1"] is not None: features[12] = float(action["connection_1"]) / 128.0
    if "connection_2" in action and action["connection_2"] is not None: features[13] = float(action["connection_2"]) / 128.0

    if (give_slot := _resource_slot(action.get("give"))) >= 0: features[14 + give_slot] = 1.0
    if (receive_slot := _resource_slot(action.get("receive"))) >= 0: features[19 + receive_slot] = 1.0
    if (resource_slot := _resource_slot(action.get("resource"))) >= 0: features[24] = float(resource_slot + 1) / 5.0
    if (resource_1_slot := _resource_slot(action.get("resource_1"))) >= 0: features[25] = float(resource_1_slot + 1) / 5.0
    if (resource_2_slot := _resource_slot(action.get("resource_2"))) >= 0: features[26] = float(resource_2_slot + 1) / 5.0
    if (card_slot := _dev_card_slot(action.get("card"))) >= 0: features[27] = float(card_slot + 1) / 5.0
    if (rate := action.get("rate")) is not None: features[28] = float(rate) / 4.0
    if (required := action.get("required")) is not None: features[29] = float(required) / 8.0
    if isinstance(resources_to_discard := action.get("resources"), dict):
        features[30] = sum(float(v) for v in resources_to_discard.values()) / 8.0
        features[31] = sum(1 for v in resources_to_discard.values() if int(v) > 0) / 5.0
    if action.get("type") == "play_dev_card": features[32] = 1.0

    features[33] = float(phase.value) / 10.0
    features[34] = 1.0 if phase == TurnPhase.SETUP else 0.0
    features[35] = 1.0 if phase == TurnPhase.MAIN_ACTION else 0.0
    features[36] = 1.0 if phase == TurnPhase.TRADE_PROPOSE else 0.0
    features[37] = 1.0 if phase == TurnPhase.TRADE_RESPOND else 0.0
    features[38] = float(player_stats["vp"]) / 10.0
    features[39] = float(player_stats["resource_total"]) / 20.0
    return features

def _encode_ppo_trade_action(action: Dict[str, Any], player_stats: Dict, pending_info: Optional[Dict]) -> List[float]:
    features = [0.0] * TRADE_FEATURE_DIM
    action_type = action.get("type", "")
    action_types = {"skip_trade": 0, "propose_trade": 1, "accept_trade": 2, "reject_trade": 3, "counter_trade": 4}
    if (action_type_idx := action_types.get(action_type)) is not None:
        features[action_type_idx] = 1.0

    features[5] = float(player_stats["vp"]) / 10.0
    features[6] = float(player_stats["resource_total"]) / 20.0

    if pending_info is not None:
        features[7] = 1.0
        features[8] = float(pending_info.get("counter_count", 0)) / 3.0
        features[9] = float(pending_info.get("proposer", 0)) / 3.0
        features[10] = float(pending_info.get("target", 0)) / 3.0

    if (target := action.get("target")) is not None:
        try:
            features[11 + int(target)] = 1.0
        except (ValueError, TypeError, IndexError): pass

    offer = action.get("offer") or action.get("counter_offer") or {}
    for i, res in enumerate(["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]):
        features[15 + i] = float(offer.get(res, 0)) / 5.0

    request = action.get("request") or action.get("counter_request") or {}
    for i, res in enumerate(["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]):
        features[20 + i] = float(request.get(res, 0)) / 5.0

    if (response_type := action.get("response_type")) is not None:
        response_types = {"accept": 25, "reject": 26, "counter": 27}
        if (response_idx := response_types.get(response_type)) is not None:
            features[response_idx] = 1.0
    return features

def _encode_ppo_gameplay_actions_batch(legal_actions: List[Dict], player_stats: Dict, phase: TurnPhase) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.zeros((MAX_GAMEPLAY_ACTIONS, GAMEPLAY_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(MAX_GAMEPLAY_ACTIONS, dtype=bool)
    capped_actions = legal_actions[:MAX_GAMEPLAY_ACTIONS]
    for idx, action in enumerate(capped_actions):
        candidates[idx] = np.array(_encode_ppo_gameplay_action(action, phase, player_stats), dtype=np.float32)
        mask[idx] = True
    if not capped_actions: mask[0] = True
    return candidates, mask

def _encode_ppo_trade_actions_batch(legal_actions: List[Dict], player_stats: Dict, pending_info: Optional[Dict]) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.zeros((MAX_TRADE_ACTIONS, TRADE_FEATURE_DIM), dtype=np.float32)
    mask = np.zeros(MAX_TRADE_ACTIONS, dtype=bool)
    capped_actions = legal_actions[:MAX_TRADE_ACTIONS]
    for idx, action in enumerate(capped_actions):
        candidates[idx] = np.array(_encode_ppo_trade_action(action, player_stats, pending_info), dtype=np.float32)
        mask[idx] = True
    if not capped_actions: mask[0] = True
    return candidates, mask

def build_ppo_obs(env: CatanEnv, raw_obs: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    player_state = raw_obs["player"]
    other_players = [v for k, v in raw_obs["players"].items() if k != raw_obs["game"]["current_player"]]
    phase = env.get_phase()
    legal_actions = raw_obs.get("legal_actions", [])

    current_player_id = env.get_current_player_id()
    player_engine_state = env.engine.players[current_player_id]
    player_stats = {
        "vp": float(player_engine_state.update_victory_points()),
        "resource_total": float(sum(int(v) for v in player_engine_state.resources.values())),
    }
    pending = env.engine.trade_manager.get_pending_trade()
    pending_info = None
    if pending is not None:
        pending_info = {
            "counter_count": float(pending.counter_count),
            "proposer": float(int(pending.proposer)),
            "target": float(int(pending.target)),
        }

    def to_vec(state: dict) -> np.ndarray:
        resources = state.get("resources", {})
        roads_val = state.get("roads", 0)
        if isinstance(roads_val, list): roads_val = len(roads_val)
        dev_cards_val = state.get("dev_cards", 0)
        if isinstance(dev_cards_val, list): dev_cards_val = len(dev_cards_val)
        vec = np.array([
            float(resources.get("WOOD", 0)), float(resources.get("BRICK", 0)), float(resources.get("SHEEP", 0)),
            float(resources.get("WHEAT", 0)), float(resources.get("ORE", 0)), float(state.get("victory_points", 0)),
            float(state.get("num_settlements", 0)), float(state.get("num_cities", 0)), float(roads_val),
            float(state.get("bonus_vp", 0)), float(state.get("dev_victory_points", 0)), float(dev_cards_val),
            float(state.get("played_knights", 0)), float(state.get("revealed_vp_cards", 0)),
        ], dtype=np.float32)
        padded = np.zeros(64, dtype=np.float32)
        padded[:len(vec)] = vec
        return padded

    self_np = to_vec(player_state)
    op_np = np.mean([to_vec(opp) for opp in other_players], axis=0).astype(np.float32) if other_players else np.zeros(64, dtype=np.float32)

    board_np = np.zeros(64, dtype=np.float32)
    game = raw_obs.get("game", {})
    board_np[0] = float(game.get("turn_number", 0))
    board_np[1] = float(int(raw_obs["game"].get("current_player", 0)))
    board_np[2] = float(phase.value)
    board_np[3] = 1.0 if game.get("enable_trading", True) else 0.0
    last_roll = game.get("last_roll")
    board_np[4] = float(last_roll if last_roll is not None else 0.0)
    board_np[5] = 1.0 if game.get("robber_pending", False) else 0.0
    if pending_info is not None:
        board_np[6] = 1.0
        board_np[7] = pending_info["counter_count"]
    robber_event = game.get("last_robber_event")
    if robber_event is not None:
        board_np[8] = 1.0 if robber_event.get("rolled_seven", False) else 0.0
        board_np[9] = 1.0 if robber_event.get("stolen_from") is not None else 0.0
        discarded = robber_event.get("discarded", {})
        total_discarded = sum(float(sum(res_map.values())) for res_map in discarded.values())
        board_np[10] = total_discarded
    board_np[11] = float(len(legal_actions))
    board_np[19] = float(len(raw_obs.get("robber_discard_queue", [])))
    board_np[20] = float(raw_obs.get("robber_discard_required", {}).get(current_player_id, 0))

    gameplay_np, gameplay_mask_np = _encode_ppo_gameplay_actions_batch(legal_actions, player_stats, phase)
    trade_np, trade_mask_np = _encode_ppo_trade_actions_batch(legal_actions, player_stats, pending_info)

    np_obs = {
        "board": board_np, "self": self_np, "opponent": op_np,
        "gameplay_candidates": gameplay_np, "gameplay_mask": gameplay_mask_np,
        "trade_candidates": trade_np, "trade_mask": trade_mask_np,
    }
    return {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in np_obs.items()}


def _resolve_discard_action(action: Dict[str, Any]) -> Dict[str, Any]:
    required = int(action.get("required", 0))
    available = action.get("available", {})
    ordered_resources = sorted(available.items(), key=lambda item: (-int(item[1]), item[0]))
    resources_to_discard: Dict[Resource, int] = {r: 0 for r in Resource}
    remaining = required
    for resource_name, count in ordered_resources:
        if remaining <= 0: break
        take = min(int(count), remaining)
        if take > 0:
            resources_to_discard[Resource[resource_name]] = take
            remaining -= take
    if remaining > 0: return action
    resolved = dict(action)
    resolved["resources"] = resources_to_discard
    return resolved


def decode_gameplay_action(action_idx: int, env: CatanEnv) -> dict:
    legal_actions = env.get_legal_actions()
    if env.get_phase() == TurnPhase.END_TURN: return {"type": "end_turn"}
    if not legal_actions: return {"type": "end_main_action"}
    mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
    chosen_action = legal_actions[mapped_idx]
    if chosen_action.get("type") == "discard_cards" and "resources" not in chosen_action:
        return _resolve_discard_action(chosen_action)
    return chosen_action


def decode_trade_action(action_idx: int, env: CatanEnv) -> dict:
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        return {"type": "reject_trade", "response_type": "reject"} if env.get_phase() == TurnPhase.TRADE_RESPOND else {"type": "skip_trade"}
    mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
    return legal_actions[mapped_idx]


def snapshot_resources(env):
    return {
        str(p): {k.name: int(v) for k, v in env.engine.players[p].resources.items()}
        for p in env.engine.players
    }


def snapshot_vp(env):
    return {
        str(p): int(env.engine.players[p].update_victory_points())
        for p in env.engine.players
    }


def diff_resources(before, after):
    out = {}
    for player, res_map in after.items():
        delta_map = {}
        for resource, value_after in res_map.items():
            value_before = before[player][resource]
            delta = int(value_after) - int(value_before)
            if delta != 0:
                delta_map[resource] = delta
        if delta_map:
            out[player] = delta_map
    return out


def format_ascii_board(env):
    def fmt_tile(t_id):
        tile = env.engine.board.tiles[t_id]
        res = tile.resource.name[:2]
        rob = "*" if getattr(tile, "has_robber", False) else " "
        num = str(tile.number) if tile.number is not None else ""
        return f"[{res}{rob}{num:>2}]"

    r0 = [7, 12, 16]
    r1 = [3, 8, 13, 17]
    r2 = [0, 4, 9, 14, 18]
    r3 = [1, 5, 10, 15]
    r4 = [2, 6, 11]

    lines = [
        "ASCII BOARD:",
        " " * 10 + "   ".join(fmt_tile(i) for i in r0),
        " " * 5 + "   ".join(fmt_tile(i) for i in r1),
        "" + "   ".join(fmt_tile(i) for i in r2),
        " " * 5 + "   ".join(fmt_tile(i) for i in r3),
        " " * 10 + "   ".join(fmt_tile(i) for i in r4),
    ]
    return "\n".join(lines)


def format_board_state(env):
    lines = []
    lines.append("=== CURRENT BOARD STATE ===")
    lines.append(format_ascii_board(env))
    lines.append("")
    lines.append(f"phase={env.get_phase().name}")
    lines.append(f"current_player={env.get_current_player_id()}")
    lines.append(f"last_roll={env.get_last_roll()}")
    lines.append(f"last_robber_event={env.get_last_robber_event()}")
    lines.append(f"board: {len(env.engine.board.tiles)} tiles, {len(env.engine.board.vertices)} vertices, {len(env.engine.board.connections)} connections")
    lines.append("")

    # Show tiles
    lines.append("TILES:")
    for i, tile in enumerate(env.engine.board.tiles):
        robber_str = " (ROBBER)" if tile.has_robber else ""
        lines.append(f"  tile_{i}: {tile.resource.name}@{tile.number}{robber_str}")
    lines.append("")

    # Show tile vertex mapping
    lines.append("TILE VERTICES:")
    lines.append("  (Corner order: [BottomRight, TopRight, Top, TopLeft, BottomLeft, Bottom])")
    lines.append("       v2(Top)      ")
    lines.append("   v3          v1   ")
    lines.append("       [Tile]       ")
    lines.append("   v4          v0   ")
    lines.append("      v5(Bottom)    ")
    for i, tile in enumerate(env.engine.board.tiles):
        v_ids = [f"{v.id:02d}" for v in tile.vertices]
        lines.append(f"  tile_{i:>2} ({tile.resource.name[:2]:>2}): [{', '.join(v_ids)}]")
    lines.append("")

    # Show ports
    lines.append("PORTS:")
    for port in env.engine.board.ports:
        v1, v2 = port.vertices
        shared_tiles = set(t.id for t in v1.tiles) & set(t.id for t in v2.tiles)
        tile_id = list(shared_tiles)[0] if shared_tiles else None
        res_str = "3:1" if port.resource is None else f"{port.resource.name} 2:1"
        lines.append(f"  P{port.id}: {res_str} at V{v1.id}-V{v2.id} (Tile {tile_id})")
    lines.append("")

    # Show vertices and their buildings
    lines.append("VERTICES (settlements/cities):")
    occupied_vertices = []
    for i in range(len(env.engine.board.vertices)):  # Show all vertices
        vertex = env.engine.board.vertices[i]
        owner = vertex.owner()
        building = "empty"
        if owner is not None and vertex.building is not None:
            if vertex.building.type.name == "SETTLEMENT":
                building = f"settlement({owner.name})"
                occupied_vertices.append(f"V{i}: {building}")
            elif vertex.building.type.name == "CITY":
                building = f"city({owner.name})"
                occupied_vertices.append(f"V{i}: {building}")
    if occupied_vertices:
        for line in occupied_vertices:
            lines.append(f"  {line}")
    else:
        lines.append("  (no buildings placed)")
    lines.append("")

    # Show connections and roads
    lines.append("CONNECTIONS (roads):")
    roads = []
    for i in range(len(env.engine.board.connections)):  # Show all connections
        conn = env.engine.board.connections[i]
        if conn.owner is not None:
            v1_id = env.engine.board.vertices.index(conn.v1)
            v2_id = env.engine.board.vertices.index(conn.v2)
            roads.append(f"C{i} (V{v1_id}-V{v2_id}): {conn.owner.name}")
    if roads:
        for line in roads:
            lines.append(f"  {line}")
    else:
        lines.append("  (no roads placed)")
    lines.append("")

    # Show player states
    lines.append("PLAYERS:")
    for player in env.engine.player_order:
        state = env.engine.players[player]
        resources = (
            f"WOOD:{state.resources[Resource.WOOD]}, "
            f"BRICK:{state.resources[Resource.BRICK]}, "
            f"SHEEP:{state.resources[Resource.SHEEP]}, "
            f"WHEAT:{state.resources[Resource.WHEAT]}, "
            f"ORE:{state.resources[Resource.ORE]}"
        )
        lines.append(
            f"  {player.name}: settlements={state.n_settlements}, "
            f"cities={state.n_cities}, roads={state.n_roads}, "
            f"vp={state.update_victory_points()}, resources={{ {resources} }}"
        )

    return "\n".join(lines)


def load_state_dict_from_checkpoint(checkpoint_path, device):
    raw = torch.load(checkpoint_path, map_location=device)

    if isinstance(raw, dict):
        if "state_dict" in raw:
            return raw["state_dict"]
        if "policy_state_dict" in raw:
            return raw["policy_state_dict"]
        if "model_state_dict" in raw:
            return raw["model_state_dict"]
        if "policy" in raw:
            return raw["policy"]
        if "models" in raw and isinstance(raw["models"], dict):
            if "model" in raw["models"]:
                return raw["models"]["model"]
        candidate = raw
        if all(isinstance(k, str) for k in candidate.keys()):
            return candidate

    raise ValueError(f"Unable to locate state_dict in checkpoint: {checkpoint_path}")


def infer_policy_type_from_state_dict(state_dict: Dict[str, Any]) -> Optional[str]:
    """Infers policy type by looking for unique layer names in the state_dict."""
    keys = set(state_dict.keys())

    if any(k.startswith("gameplay_actor.") for k in keys) or any(k.startswith("trade_actor.") for k in keys):
        return "ppo"

    if any(k.startswith("action_heads.") for k in keys):
        return "unified"

    # ToMEnhancedDQNPolicy has a global_encoder, which DQNBaselinePolicy does not.
    # This is the primary differentiator.
    if any(k.startswith("global_encoder.") for k in keys):
        return "tom_dqn"

    # DQNBaselinePolicy has 'gameplay_head' and 'trade_head' (like ToM-DQN, but without global_encoder).
    # The absence of 'global_encoder' is the key.
    if any(k.startswith("gameplay_head.") for k in keys) or any(k.startswith("trade_head.") for k in keys):
        return "dqn"

    return None


def run_single_game(
    model,
    policy_type,
    args,
    checkpoint_path,
    device,
    deterministic=False,
    max_steps=2000,
    show_obs=False,
    show_board=False,
    gameplay_only=False,
):
    model.eval()
    env = CatanEnv(seed=args.seed, enable_trading=not gameplay_only)
    env.reset()

    report = []
    total_steps = 0
    done = False

    while not done and total_steps < max_steps:
        phase = env.get_phase()
        phase_name = (
            "gameplay"
            if phase.name in ("SETUP", "MAIN_ACTION", "END_TURN")
            else "trade"
            if phase.name in ("TRADE_PROPOSE", "TRADE_RESPOND")
            else "auto"
        )

        resources_before = snapshot_resources(env)
        vp_before = snapshot_vp(env)
        obs_before = env.get_observation()
        legal_before = obs_before.get("legal_actions", [])

        if phase_name == "auto":
            next_obs, reward, done, info = env.step(None)

            last_roll = next_obs["game"].get("last_roll")
            robber_event = next_obs["game"].get("last_robber_event")

            print(
                f"step={total_steps} phase={phase.name} player={env.get_current_player_id()} "
                f"action=auto roll={last_roll} reward={reward:.3f} done={done}"
            )
            if robber_event is not None:
                print(f" robber_event={robber_event}")

            resources_after = snapshot_resources(env)
            vp_after = snapshot_vp(env)
            resource_delta = diff_resources(resources_before, resources_after)

            if resource_delta:
                print(f" resource_delta={resource_delta}")
            if vp_after != vp_before:
                print(f" vp={vp_after}")

            report.append(
                {
                    "step": total_steps,
                    "phase": phase.name,
                    "action": None,
                    "legal_actions_before": legal_before,
                    "roll": last_roll,
                    "robber_event": robber_event,
                    "reward": reward,
                    "done": done,
                    "info": info,
                    "vp_before": vp_before,
                    "vp_after": vp_after,
                    "resources_before": resources_before,
                    "resources_after": resources_after,
                    "resource_delta": resource_delta,
                }
            )
            total_steps += 1
            continue

        current_player = env.get_current_player_id()

        if policy_type == 'ppo':
            obs = build_ppo_obs(env, obs_before, device)
        else:
            # Legacy observation builder for unified/dqn
            obs = build_policy_obs(env, obs_before, device)

        if show_board:
            print(format_board_state(env))
            print()

        if policy_type == "unified":
            value, action_dict, _, tom_out = model.act(
                obs=obs,
                phase=phase_name,
                deterministic=deterministic,
            )
            value = value.item()
        elif policy_type == "ppo":
            value, action_dict, _ = model.act(
                obs=obs,
                phase=phase_name,
                deterministic=deterministic,
            )
            value = value.item()
            tom_out = None
        else:  # dqn or tom_dqn
            # The act method for DQN policies just returns the action dict
            action_dict = model.act(
                obs=obs,
                phase=phase_name,
                epsilon=0.0,  # Use greedy action for evaluation
            )
            value = model.get_value(obs, phase_name).item() if hasattr(model, 'get_value') else None

            # For ToM-enhanced DQN, we can extract the need prediction for inspection
            if policy_type == 'tom_dqn' and hasattr(model, 'encode'):
                _, _, need_pred = model.encode(obs)
                tom_out = {'need_pred': need_pred}
            else:
                tom_out = None

        if gameplay_only and phase_name == "trade":
            env_action = (
                {"type": "reject_trade", "response_type": "reject"}
                if phase.name == "TRADE_RESPOND"
                else {"type": "skip_trade"}
            )
        else:
            if phase_name == "gameplay":
                env_action = decode_gameplay_action(int(action_dict["gameplay_action"].item()), env)
            else:  # trade phase
                env_action = decode_trade_action(int(action_dict["trade_action"].item()), env)

        next_obs, reward, done, info = env.step(env_action)

        last_roll = next_obs["game"].get("last_roll")
        robber_event = next_obs["game"].get("last_robber_event")
        setup_stage = next_obs["game"].get("initial_placement_stage")
        setup_phase = next_obs["game"].get("initial_placement_phase")

        resources_after = snapshot_resources(env)
        vp_after = snapshot_vp(env)
        resource_delta = diff_resources(resources_before, resources_after)

        print(f"step={total_steps} phase={phase.name} player={current_player}")
        print(f" legal_actions_before={legal_before}")
        print(f" env_action={env_action}")
        print(
            " action_dict={"
            + ", ".join(f"{k}:{v.detach().cpu().numpy().tolist()}" for k, v in action_dict.items())
            + "}"
        )
        print(f" roll={last_roll}")
        print(f" setup_phase={setup_phase} setup_stage={setup_stage}")
        if robber_event is not None:
            print(f" robber_event={robber_event}")
        print(f" reward={reward:.3f} done={done}")
        print(f" value={value}")
        print(f" vp_before={vp_before}")
        print(f" vp_after={vp_after}")
        if resource_delta:
            print(f" resource_delta={resource_delta}")
        print(f" resources_after={resources_after}\n")

        report.append(
            {
                "step": total_steps,
                "phase": phase.name,
                "player": str(current_player),
                "legal_actions_before": legal_before,
                "action_dict": {k: v.detach().cpu().numpy().tolist() for k, v in action_dict.items()},
                "env_action": env_action,
                "roll": last_roll,
                "setup_phase": setup_phase,
                "setup_stage": setup_stage,
                "robber_event": robber_event,
                "reward": reward,
                "done": done,
                "value": value,
                "tom": {k: v.detach().cpu().numpy().tolist() for k, v in tom_out.items()} if tom_out is not None else None,
                "vp_before": vp_before,
                "vp_after": vp_after,
                "resources_before": resources_before,
                "resources_after": resources_after,
                "resource_delta": resource_delta,
            }
        )

        total_steps += 1

    winner = env.engine.winner
    stats = {
        "winner": str(winner) if winner is not None else None,
        "victory_points": snapshot_vp(env),
        "total_steps": total_steps,
        "done": done,
        "report": report,
        "final_board_state": format_board_state(env),
    }
    return stats


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    state_dict = load_state_dict_from_checkpoint(args.checkpoint, device)

    # Infer policy type from state_dict keys, but allow user to override.
    # If the user-provided type (or default) mismatches the inferred one,
    # prioritize the inferred type to prevent loading errors.
    policy_type = args.policy_type
    inferred_type = infer_policy_type_from_state_dict(state_dict)

    if inferred_type and policy_type != inferred_type:
        print(f"Info: Checkpoint keys suggest policy type '{inferred_type}', overriding specified/default '{policy_type}'.")
        policy_type = inferred_type
    elif inferred_type is None:
        print(f"Warning: Could not infer policy type from checkpoint. Using specified type: '{policy_type}'")

    if policy_type == "unified":
        model = UnifiedPolicy().to(device)
    elif policy_type == "tom_dqn":
        model = ToMEnhancedDQNPolicy().to(device)
    elif policy_type == "ppo":
        model = PPOPolicy(
            gameplay_feature_dim=GAMEPLAY_FEATURE_DIM,
            trade_feature_dim=TRADE_FEATURE_DIM
        ).to(device)
    else:
        model = DQNBaselinePolicy().to(device)

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict

        stripped = OrderedDict()
        for k, v in state_dict.items():
            stripped[k.replace("module.", "")] = v
        model.load_state_dict(stripped)

    stats = run_single_game(
        model=model,
        policy_type=policy_type,
        args=args,
        checkpoint_path=args.checkpoint,
        device=device,
        deterministic=args.deterministic,
        max_steps=args.max_steps,
        show_obs=args.show_obs,
        show_board=args.show_board,
        gameplay_only=args.gameplay_only,
    )

    print("\n=== GAME RESULT ===")
    print(f"winner={stats['winner']}")
    print(f"victory_points={stats['victory_points']}")
    print(f"total_steps={stats['total_steps']}")
    print(f"done={stats['done']}")
    print(stats["final_board_state"])

    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    suffix = "_gameplay_only" if args.gameplay_only else ""
    out_path = os.path.join("gameplay", f"gameplay_{checkpoint_name}{suffix}_steps{stats['total_steps']}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=== GAME RESULT ===\n")
        f.write(f"winner={stats['winner']}\n")
        f.write(f"victory_points={stats['victory_points']}\n")
        f.write(f"total_steps={stats['total_steps']}\n")
        f.write(f"done={stats['done']}\n\n")
        f.write(stats["final_board_state"])
        f.write("\n\n=== STEPS ===\n")

        for step in stats["report"]:
            f.write(
                f"step={step['step']} phase={step['phase']} reward={step.get('reward')} "
                f"done={step.get('done')}\n"
            )

            if "player" in step:
                f.write(f"  player={step['player']}\n")
            if "legal_actions_before" in step:
                f.write(f"  legal_actions_before={step['legal_actions_before']}\n")
            if "env_action" in step:
                f.write(f"  env_action={step['env_action']}\n")
            if "action_dict" in step:
                f.write(f"  action_dict={step['action_dict']}\n")
            if "roll" in step:
                f.write(f"  roll={step['roll']}\n")
            if "setup_phase" in step:
                f.write(f"  setup_phase={step['setup_phase']}\n")
            if "setup_stage" in step:
                f.write(f"  setup_stage={step['setup_stage']}\n")
            if "robber_event" in step and step["robber_event"] is not None:
                f.write(f"  robber_event={step['robber_event']}\n")
            if "value" in step:
                f.write(f"  value={step['value']}\n")
            if "vp_before" in step:
                f.write(f"  vp_before={step['vp_before']}\n")
            if "vp_after" in step:
                f.write(f"  vp_after={step['vp_after']}\n")
            if "resources_before" in step:
                f.write(f"  resources_before={step['resources_before']}\n")
            if "resources_after" in step:
                f.write(f"  resources_after={step['resources_after']}\n")
            if "resource_delta" in step:
                f.write(f"  resource_delta={step['resource_delta']}\n")
            if "tom" in step and step["tom"] is not None:
                f.write(f"  tom={step['tom']}\n")
            f.write("\n")

    print(f"Saved detailed game log to {out_path}")


if __name__ == "__main__":
    main()