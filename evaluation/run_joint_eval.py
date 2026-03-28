from __future__ import annotations

from typing import Dict, List

from environment.catan_env import CatanEnv
from evaluation.trade_metrics import compute_trade_metrics


def run_single_game(env: CatanEnv) -> Dict:
    obs = env.reset()

    done = False

    trade_events = []

    while not done:
        controller = env.get_active_model_name()

        if controller == "gameplay":
            action = {"type": "end_turn"}
        elif controller == "trade":
            action = {"type": "reject_trade", "response_type": "reject"}
        else:
            action = None

        obs, reward, done, _ = env.step(action)

    history = env.engine.trade_history.get_recent_events()

    for event in history:
        trade_events.append(event.to_dict())

    return {
        "trades": trade_events,
        "winner": env.engine.winner,
    }


def run_evaluation(
    num_games: int = 16,
    seed: int = 0,
) -> Dict[str, float]:
    env = CatanEnv(seed=seed)

    game_logs: List[Dict] = []

    for i in range(num_games):
        game_result = run_single_game(env)
        game_logs.append(game_result)

    all_trades = []
    wins = {}

    for game in game_logs:
        all_trades.extend(game["trades"])

        winner = game["winner"]
        if winner not in wins:
            wins[winner] = 0
        wins[winner] += 1

    trade_metrics = compute_trade_metrics(all_trades)

    win_rates = {
        str(player): wins.get(player, 0) / num_games
        for player in wins
    }

    return {
        **trade_metrics,
        "win_rates": win_rates,
    }


if __name__ == "__main__":
    results = run_evaluation(num_games=16)
    print(results)