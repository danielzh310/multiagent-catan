from __future__ import annotations

import math
from typing import Dict

from core.constants import Resource, PlayerId
from learning.trade.trade_reward import estimate_trade_surplus

# This alpha hyperparameter scales the normalized surplus.
# It corresponds to `α` in the formula `r_trade = α * σ(S)`.
TRADE_REWARD_ALPHA = 0.025


def _sigmoid(x: float) -> float:
    """Sigmoid function to normalize surplus into [0, 1]."""
    # We scale the input to the sigmoid to ensure it's not always saturated near 0.5
    # for the small values that estimate_trade_surplus produces. This is an implementation detail.
    return 1 / (1 + math.exp(-x * 100))


def unified_accepted_trade_reward(
    proposer_offer: Dict[Resource, int] | None,
    proposer_request: Dict[Resource, int] | None,
    accepter_resources: Dict[Resource, int] | None,
    opponent_need_scores: Dict[Resource, float] | None = None,
) -> float:
    """
    Reward for accepting a trade, based on the formula r_trade = α * σ(S).
    This is specific to the Unified PPO model to align with its design.
    """
    # S = bilateral_surplus
    bilateral_surplus = estimate_trade_surplus(proposer_offer, proposer_request, accepter_resources, opponent_need_scores)

    # α * σ(S)
    return TRADE_REWARD_ALPHA * _sigmoid(bilateral_surplus)


def unified_rejected_trade_reward() -> float:
    """Unified model specific reward for rejecting a trade."""
    return 0.0 # No base reward, penalties/bonuses handled by RewardShaper


def unified_skipped_trade_reward() -> float:
    """Unified model specific reward for skipping a trade."""
    return 0.0 # No base reward, penalties/bonuses handled by RewardShaper