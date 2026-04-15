from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from core.constants import Resource


RESOURCE_ORDER = [
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
]


def _empty_target() -> torch.Tensor:
    return torch.zeros(len(RESOURCE_ORDER), dtype=torch.float32)


def resource_dict_to_tensor(resource_dict: Dict[Resource, int] | None) -> torch.Tensor:
    out = _empty_target()
    if resource_dict is None:
        return out

    for i, r in enumerate(RESOURCE_ORDER):
        out[i] = float(resource_dict.get(r, 0))
    return out


def normalize_target(x: torch.Tensor) -> torch.Tensor:
    s = x.sum()
    if s <= 0:
        return torch.ones_like(x) / float(len(x))
    return x / s


def build_need_target_from_trade_event(event: Dict) -> Tuple[torch.Tensor, float]:
    """
    Weak supervision target from observed trade behavior and gameplay actions.

    Accepted trade:
      - target likely wanted what was offered to them
    Rejected trade:
      - weak negative / uniform fallback
    Counter trade:
      - use the counter_request if available, otherwise weak uniform
    Gameplay need:
      - direct supervision from resource costs of build actions
    """
    response_type = event.get("response_type", "")
    offer = event.get("offer", None)
    counter_request = event.get("counter_request", None)

    if response_type == "accept":
        target = resource_dict_to_tensor(offer)
        return normalize_target(target), 1.0

    if response_type == "counter":
        target = resource_dict_to_tensor(counter_request)
        if target.sum() > 0:
            return normalize_target(target), 0.6
        return normalize_target(_empty_target()), 0.0

    if response_type == "reject":
        return normalize_target(_empty_target()), 0.0

    if response_type == "gameplay_need":
        # Strong supervision from gameplay actions
        target = resource_dict_to_tensor(offer)
        return normalize_target(target), 1.0

    return normalize_target(_empty_target()), 0.0


def build_batch_need_targets(trade_events: List[Dict]) -> tuple[torch.Tensor, torch.Tensor]:
    targets = []
    mask = []

    for event in trade_events:
        t, m = build_need_target_from_trade_event(event)
        targets.append(t)
        mask.append(m)

    if not targets:
        return torch.zeros((0, len(RESOURCE_ORDER))), torch.zeros((0,))

    return torch.stack(targets, dim=0), torch.tensor(mask, dtype=torch.float32)