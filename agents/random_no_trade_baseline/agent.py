from __future__ import annotations

import random
from typing import Any, Iterable, Optional


class RandomNoTradeBaselineAgent:
    """
    Random legal-action agent that declines player-to-player trades.

    The environment can still allow normal gameplay actions, including bank trades.
    Player trade proposals/responses are skipped or rejected whenever they appear.
    """

    TRADE_DECLINE_ACTIONS = {"skip_trade", "reject_trade"}
    PLAYER_TRADE_ACTIONS = {"propose_trade", "accept_trade", "counter_trade"}

    def __init__(self, seed: Optional[int] = None, allow_bank_trades: bool = True):
        self.random = random.Random(seed)
        self.allow_bank_trades = allow_bank_trades

    def select_action(self, legal_actions: Iterable[dict[str, Any]]) -> Optional[dict[str, Any]]:
        actions = list(legal_actions)
        if not actions:
            return None

        decline_actions = [
            action for action in actions
            if action.get("type") in self.TRADE_DECLINE_ACTIONS
        ]
        if decline_actions:
            return self.random.choice(decline_actions)

        candidates = [
            action for action in actions
            if action.get("type") not in self.PLAYER_TRADE_ACTIONS
        ]
        if not self.allow_bank_trades:
            candidates = [
                action for action in candidates
                if action.get("type") != "bank_trade"
            ]

        if not candidates:
            candidates = actions

        return self.random.choice(candidates)

    def act(self, observation: dict[str, Any]) -> Optional[dict[str, Any]]:
        return self.select_action(observation.get("legal_actions", []))
