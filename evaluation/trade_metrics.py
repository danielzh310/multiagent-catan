from __future__ import annotations

from typing import Dict, List


def compute_trade_metrics(trade_events: List[dict]) -> Dict[str, float]:
    """
    Computes trade-related metrics from a list of trade events.
    """

    if len(trade_events) == 0:
        return {
            "num_trades": 0,
            "acceptance_rate": 0.0,
            "counter_rate": 0.0,
            "avg_trade_volume": 0.0,
        }

    num_trades = len(trade_events)

    accepted = sum(1 for e in trade_events if e.get("accepted", False))
    counters = sum(1 for e in trade_events if e.get("response_type") == "counter")

    total_volume = 0
    for e in trade_events:
        offer = e.get("offer", {})
        request = e.get("request", {})

        total_volume += sum(offer.values())
        total_volume += sum(request.values())

    return {
        "num_trades": num_trades,
        "acceptance_rate": accepted / num_trades,
        "counter_rate": counters / num_trades,
        "avg_trade_volume": total_volume / num_trades,
    }


def compute_player_trade_metrics(trade_events: List[dict], player_id) -> Dict[str, float]:
    """
    Metrics for a specific player.
    """

    relevant = [
        e for e in trade_events
        if e.get("proposer") == player_id or e.get("target") == player_id
    ]

    if len(relevant) == 0:
        return {
            "num_trades": 0,
            "acceptance_rate": 0.0,
            "counter_rate": 0.0,
            "avg_trade_volume": 0.0,
        }

    return compute_trade_metrics(relevant)


def compute_game_metrics(game_logs: List[dict]) -> Dict[str, float]:
    """
    Computes high-level metrics across multiple games.
    """

    total_trades = 0
    total_accepts = 0
    total_counters = 0
    total_volume = 0

    for game in game_logs:
        trades = game.get("trades", [])
        total_trades += len(trades)

        for e in trades:
            if e.get("accepted", False):
                total_accepts += 1
            if e.get("response_type") == "counter":
                total_counters += 1

            total_volume += sum(e.get("offer", {}).values())
            total_volume += sum(e.get("request", {}).values())

    if total_trades == 0:
        return {
            "num_trades": 0,
            "acceptance_rate": 0.0,
            "counter_rate": 0.0,
            "avg_trade_volume": 0.0,
        }

    return {
        "num_trades": total_trades,
        "acceptance_rate": total_accepts / total_trades,
        "counter_rate": total_counters / total_trades,
        "avg_trade_volume": total_volume / total_trades,
    }