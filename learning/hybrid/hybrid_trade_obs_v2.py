from __future__ import annotations

from typing import Dict, List

import torch

from core.constants import Resource
from environment.catan_env import CatanEnv


ACTION_TYPE_INDEX = {
    "propose_trade": 0,
    "accept_trade": 1,
    "reject_trade": 2,
    "counter_trade": 3,
    "skip_trade": 4,
}

TRADE_HISTORY_WINDOW = 4


def _resource_slot(resource_value) -> int:
    if resource_value is None:
        return -1
    try:
        return int(Resource(resource_value))
    except (ValueError, TypeError):
        return -1


def _player_vector(state: dict) -> List[float]:
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


def _pad_1d(values, length: int, fill_value=0):
    trimmed = list(values)[-length:]
    if len(trimmed) < length:
        trimmed = [fill_value] * (length - len(trimmed)) + trimmed
    return trimmed


def _pad_2d(values, length: int, width: int, fill_value=0.0):
    trimmed = list(values)[-length:]
    normalized = []
    for row in trimmed:
        row_list = list(row)
        if len(row_list) < width:
            row_list = row_list + [fill_value] * (width - len(row_list))
        normalized.append(row_list[:width])
    while len(normalized) < length:
        normalized.insert(0, [fill_value] * width)
    return normalized


def build_trade_obs(env: CatanEnv, device: str) -> Dict[str, torch.Tensor]:
    raw = env.get_observation()
    player = raw["player"]
    other_players = [v for k, v in raw["players"].items() if k != raw["game"]["current_player"]]
    game = raw["game"]

    self_vec = torch.tensor([_player_vector(player)], dtype=torch.float32, device=device)

    opponent_vec = [0.0] * 64
    if other_players:
        sums = [0.0] * 64
        for opp in other_players:
            opp_vec = _player_vector(opp)
            for i in range(64):
                sums[i] += opp_vec[i]
        opponent_vec = [value / len(other_players) for value in sums]

    board_vec = torch.zeros((1, 64), dtype=torch.float32, device=device)
    board_vec[0, 0] = float(game.get("turn_number", 0))
    board_vec[0, 1] = float(env.get_phase().value)
    board_vec[0, 2] = float(game.get("last_roll") or 0)
    board_vec[0, 3] = 1.0 if game.get("robber_pending", False) else 0.0
    board_vec[0, 4] = float(game.get("dev_card_deck_size", 0))
    board_vec[0, 5] = float(len(raw.get("legal_actions", [])))

    pending = raw.get("trade")
    if pending is not None:
        board_vec[0, 6] = 1.0
        board_vec[0, 7] = float(pending.counter_count)
        board_vec[0, 8] = float(int(pending.proposer))
        board_vec[0, 9] = float(int(pending.target))

    trade_history_raw = raw["trade_history"]
    trade_history = {
        "proposer_ids": torch.tensor([_pad_1d(trade_history_raw.get("proposer_ids", []), TRADE_HISTORY_WINDOW, 0)], dtype=torch.long, device=device),
        "target_ids": torch.tensor([_pad_1d(trade_history_raw.get("target_ids", []), TRADE_HISTORY_WINDOW, 0)], dtype=torch.long, device=device),
        "response_types": torch.tensor([_pad_1d(trade_history_raw.get("response_types", []), TRADE_HISTORY_WINDOW, 0)], dtype=torch.long, device=device),
        "offers": torch.tensor([_pad_2d(trade_history_raw.get("offers", []), TRADE_HISTORY_WINDOW, 5, 0.0)], dtype=torch.float32, device=device),
        "requests": torch.tensor([_pad_2d(trade_history_raw.get("requests", []), TRADE_HISTORY_WINDOW, 5, 0.0)], dtype=torch.float32, device=device),
        "accepted_flags": torch.tensor([_pad_1d(trade_history_raw.get("accepted_flags", []), TRADE_HISTORY_WINDOW, 0.0)], dtype=torch.float32, device=device),
        "turn_numbers": torch.tensor([_pad_1d(trade_history_raw.get("turn_numbers", []), TRADE_HISTORY_WINDOW, 0.0)], dtype=torch.float32, device=device),
    }

    return {
        "board": board_vec,
        "self": self_vec,
        "opponent": torch.tensor([opponent_vec], dtype=torch.float32, device=device),
        "trade_history": trade_history,
    }


def build_trade_action_masks(env: CatanEnv, device: str) -> Dict[str, torch.Tensor]:
    legal_actions = env.get_legal_actions()
    action_type_mask = torch.zeros((1, 5), dtype=torch.float32, device=device)
    target_mask = torch.zeros((1, 4), dtype=torch.float32, device=device)
    offer_mask = torch.zeros((1, 5), dtype=torch.float32, device=device)
    request_mask = torch.zeros((1, 5), dtype=torch.float32, device=device)

    for action in legal_actions:
        action_type = action.get("type", "")
        idx = ACTION_TYPE_INDEX.get(action_type)
        if idx is not None:
            action_type_mask[0, idx] = 1.0

        target = action.get("target")
        if target is not None:
            target_mask[0, int(target)] = 1.0

        offer = action.get("offer") or action.get("counter_offer") or {}
        request = action.get("request") or action.get("counter_request") or {}

        for resource, amount in offer.items():
            if int(amount) <= 0:
                continue
            slot = _resource_slot(resource)
            if slot >= 0:
                offer_mask[0, slot] = 1.0

        for resource, amount in request.items():
            if int(amount) <= 0:
                continue
            slot = _resource_slot(resource)
            if slot >= 0:
                request_mask[0, slot] = 1.0

    if action_type_mask.sum() == 0:
        action_type_mask[0, ACTION_TYPE_INDEX["skip_trade"]] = 1.0
    if target_mask.sum() == 0:
        target_mask[:] = 1.0
    if offer_mask.sum() == 0:
        offer_mask[:] = 1.0
    if request_mask.sum() == 0:
        request_mask[:] = 1.0

    return {
        "action_type": action_type_mask,
        "target": target_mask,
        "offer": offer_mask,
        "request": request_mask,
    }
