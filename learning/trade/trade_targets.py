from __future__ import annotations

from typing import Dict, List

import torch


def build_tom_targets_from_history(batch_obs: Dict) -> Dict[str, torch.Tensor]:
    """
    Builds simple ToM targets from trade history.

    Current targets:
    - acceptance_target: whether recent trade was accepted
    - resource_pref_target: most requested resource index

    This is a heuristic bootstrap target.
    """

    trade_history = batch_obs["trade_history"]

    accepted = trade_history["accepted_flags"][:, -1]
    requests = trade_history["requests"][:, -1]

    resource_pref = torch.argmax(requests, dim=-1)

    return {
        "acceptance_target": accepted.float(),
        "resource_pref_target": resource_pref.long(),
    }


def build_empty_tom_targets(batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
    return {
        "acceptance_target": torch.zeros(batch_size, device=device),
        "resource_pref_target": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


def merge_tom_targets(target_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(target_list) == 0:
        return {}

    acceptance = torch.cat([t["acceptance_target"] for t in target_list], dim=0)
    resource_pref = torch.cat([t["resource_pref_target"] for t in target_list], dim=0)

    return {
        "acceptance_target": acceptance,
        "resource_pref_target": resource_pref,
    }