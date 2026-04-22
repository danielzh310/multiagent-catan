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


def unified_proposed_trade_reward() -> float:
    """
    Reward for proposing a trade. Set to 0.0 to prevent trade spamming.
    The agent should only be rewarded when a trade is ACCEPTED and finalized,
    not for simply proposing trades that may be rejected multiple times.
    """
    return 0.0


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
    """
    Reward for rejecting a trade.
    Set to 0.0 - no penalty or reward for rejecting. The action mask (trade fatigue)
    will prevent excessive propose->reject loops by limiting proposals per turn.
    """
    return 0.0


def unified_skipped_trade_reward() -> float:
    """
    Reward for skipping a trade phase.
    Set to 0.0 - skipping trade is a neutral action from a reward perspective.
    """
    return 0.0