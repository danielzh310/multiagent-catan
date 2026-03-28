from __future__ import annotations

from learning.trade.trade_policy import TradePolicy


def build_trade_model(device: str = "cpu") -> TradePolicy:
    """
    Factory for the trade model (ToM PPO).

    Centralizes construction so rollout workers, trainer,
    and evaluation all use identical initialization.
    """
    model = TradePolicy(device=device)
    return model