from __future__ import annotations

from typing import Dict

from core.constants import Resource


def estimate_build_progress(resources: Dict[Resource, int]) -> float:
    wood = float(resources.get(Resource.WOOD, 0))
    brick = float(resources.get(Resource.BRICK, 0))
    sheep = float(resources.get(Resource.SHEEP, 0))
    wheat = float(resources.get(Resource.WHEAT, 0))
    ore = float(resources.get(Resource.ORE, 0))

    settlement_progress = min(wood, 1.0) + min(brick, 1.0) + min(sheep, 1.0) + min(wheat, 1.0)
    city_progress = min(wheat / 2.0, 1.0) + min(ore / 3.0, 1.0)
    dev_progress = min(sheep, 1.0) + min(wheat, 1.0) + min(ore, 1.0)

    return (settlement_progress / 4.0 + city_progress / 2.0 + dev_progress / 3.0) / 3.0

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
    accepter_resources: Dict[Resource, int] | None,
    opponent_need_scores: Dict[Resource, float] | None = None,
) -> float:
    """
    Calculate bilateral trade surplus considering both parties' perspectives.

    IMPORTANT: All rewards are scaled down significantly (10x-100x smaller than VP rewards of ~0.16)
    to prevent trade rewards from dominating gameplay learning. The agent should learn
    to trade because it helps win games through VP progression, not because trading gives high rewards.
    """
    offer_total = _sum_resources(proposer_offer)
    request_total = _sum_resources(proposer_request)

    if offer_total <= 0 or request_total <= 0:
        return -0.005  # Much smaller penalty

    # Base volume balance (encourages fair trades) - scaled down
    volume_balance = 1.0 - abs(float(offer_total - request_total)) / max(offer_total + request_total, 1.0)

    # Diversity bonus (encourages trading different types of resources) - scaled down
    request_div = _resource_diversity(proposer_request)
    offer_div = _resource_diversity(proposer_offer)
    diversity_bonus = 0.001 * (request_div + offer_div)  # Much smaller

    # Opponent's surplus (from their perspective) based on ToM predictions
    opponent_surplus = 0.0
    if opponent_need_scores is not None:
        # Calculate how much the opponent values the resources they're receiving
        opponent_request_value = sum(
            opponent_need_scores.get(resource, 0.2) * count
            for resource, count in (proposer_request or {}).items()
        )
        # Calculate how much the opponent values the resources they're giving away (neutral valuation)
        proposer_offer_value = sum(
            0.2 * count for count in (proposer_offer or {}).values()  # Assume neutral proposer valuation
        )
        opponent_surplus = 0.005 * (opponent_request_value - proposer_offer_value) / max(request_total, 1.0)

    # Agent's own surplus based on change in build progress
    agent_surplus_progress = 0.0
    if accepter_resources and proposer_offer and proposer_request:
        before_progress = estimate_build_progress(accepter_resources)

        after_resources = accepter_resources.copy()
        for res, count in proposer_offer.items():
            after_resources[res] = after_resources.get(res, 0) + count
        for res, count in proposer_request.items():
            after_resources[res] = after_resources.get(res, 0) - count

        after_progress = estimate_build_progress(after_resources)
        agent_surplus_progress = after_progress - before_progress

    # Combine all surplus components. Agent's progress is weighted to be significant.
    return 0.005 * volume_balance + diversity_bonus + opponent_surplus + 0.05 * agent_surplus_progress


def accepted_trade_reward(
    proposer_offer: Dict[Resource, int] | None,
    proposer_request: Dict[Resource, int] | None,
    accepter_resources: Dict[Resource, int] | None,
    opponent_need_scores: Dict[Resource, float] | None = None,
) -> float:
    """
    Reward for accepting a trade, based on bilateral surplus calculation.

    Scaled down to ~0.008-0.027 range to be much smaller than VP rewards (~0.16),
    ensuring trade learning doesn't dominate gameplay learning.
    """
    bilateral_surplus = estimate_trade_surplus(proposer_offer, proposer_request, accepter_resources, opponent_need_scores)
    #return 0.008 + bilateral_surplus  # Much smaller base reward
    return 0.0 + bilateral_surplus  # No fixed base reward, only surplus


def rejected_trade_reward() -> float:
    """Small penalty for rejecting trades to encourage participation without dominating learning."""
    #return -0.001
    return 0.0  # No penalty for rejection to avoid discouraging strategic rejections


def skipped_trade_reward() -> float:
    """Very small penalty for skipping trades."""
    #return -0.0003
    return 0.0  # No penalty for skipping to avoid discouraging strategic skipping