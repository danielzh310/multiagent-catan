from __future__ import annotations

from typing import Dict

from core.constants import Resource


RESOURCE_ORDER = [
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
]


def _sum_resources(x: Dict[Resource, int] | None) -> int:
    if x is None:
        return 0
    return sum(int(v) for v in x.values())


def _resource_diversity(x: Dict[Resource, int] | None) -> int:
    if x is None:
        return 0
    return sum(1 for v in x.values() if int(v) > 0)


def estimate_trade_surplus(
    proposer_offer: Dict[Resource, int] | None,
    proposer_request: Dict[Resource, int] | None,
) -> float:
    offer_total = _sum_resources(proposer_offer)
    request_total = _sum_resources(proposer_request)

    if offer_total <= 0 or request_total <= 0:
        return -0.05

    volume_balance = 1.0 - abs(float(offer_total - request_total)) / max(offer_total + request_total, 1.0)
    request_div = _resource_diversity(proposer_request)
    offer_div = _resource_diversity(proposer_offer)

    return 0.05 * volume_balance + 0.01 * (request_div + offer_div)


def accepted_trade_reward(
    proposer_offer: Dict[Resource, int] | None,
    proposer_request: Dict[Resource, int] | None,
    opponent_need_score: float,
) -> float:
    bilateral = estimate_trade_surplus(proposer_offer, proposer_request)
    return 0.08 + bilateral + 0.05 * float(opponent_need_score)


def rejected_trade_reward() -> float:
    return -0.01


def skipped_trade_reward() -> float:
    return -0.003