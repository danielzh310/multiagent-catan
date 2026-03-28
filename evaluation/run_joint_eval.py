from __future__ import annotations

from typing import Dict, List

from environment.catan_env import CatanEnv
from evaluation.trade_metrics import compute_trade_metrics


def run_single_game(env: CatanEnv) -> Dict:
    env.reset()
    done = False

    trade_events = []
    gameplay_rewards = []
    trade_rewards = []

    while not done:
        controller = env.get_active_model_name()

        if controller == "gameplay":
            action = {"type": "end_turn"}
        elif controller == "trade":
            phase = env.get_phase()

            if phase.name == "TRADE_PROPOSE":
                action = {"type": "skip_trade"}
            elif phase.name == "TRADE_RESPOND":
                action = {"type": "reject_trade", "response_type": "reject"}
            else:
                action = None
        else:
            action = None

        _, reward, done, _ = env.step(action)

        if controller == "gameplay":
            gameplay_rewards.append(float(reward))
        elif controller == "trade":
            trade_rewards.append(float(reward))

    history = env.engine.trade_history.get_recent_events()

    for event in history:
        trade_events.append(event.to_dict())

    return {
        "trades": trade_events,
        "winner": env.engine.winner,
        "gameplay_reward_mean": sum(gameplay_rewards) / len(gameplay_rewards) if gameplay_rewards else 0.0,
        "trade_reward_mean": sum(trade_rewards) / len(trade_rewards) if trade_rewards else 0.0,
    }


def run_evaluation(
    num_games: int = 16,
    seed: int = 0,
) -> Dict[str, float]:
    env = CatanEnv(seed=seed)

    game_logs: List[Dict] = []

    for _ in range(num_games):
        game_result = run_single_game(env)
        game_logs.append(game_result)

    all_trades = []
    wins = {}
    gameplay_reward_means = []
    trade_reward_means = []

    for game in game_logs:
        all_trades.extend(game["trades"])

        winner = game["winner"]
        if winner not in wins:
            wins[winner] = 0
        wins[winner] += 1

        gameplay_reward_means.append(game["gameplay_reward_mean"])
        trade_reward_means.append(game["trade_reward_mean"])

    trade_metrics = compute_trade_metrics(all_trades)

    win_rates = {
        str(player): wins.get(player, 0) / num_games
        for player in wins
    }

    return {
        **trade_metrics,
        "avg_gameplay_reward_mean": sum(gameplay_reward_means) / len(gameplay_reward_means) if gameplay_reward_means else 0.0,
        "avg_trade_reward_mean": sum(trade_reward_means) / len(trade_reward_means) if trade_reward_means else 0.0,
        "win_rates": win_rates,
    }


if __name__ == "__main__":
    results = run_evaluation(num_games=16)
    print("\n===== Evaluation Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")