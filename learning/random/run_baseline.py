from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.constants import PlayerId, Resource
from environment.catan_env import CatanEnv
from learning.random.agent import RandomNoTradeBaselineAgent


BUILD_ACTIONS = {"build_settlement", "build_road", "build_city"}


@dataclass
class GameResult:
    winner: Optional[PlayerId]
    controlled_vp: int
    winner_vp: int
    turns: int
    steps: int
    avg_reward: float
    build_efficiency: float
    controlled_longest_road: bool
    controlled_largest_army: bool
    controlled_longest_road_length: int
    controlled_played_knights: int
    controlled_dev_cards_bought: int
    controlled_held_dev_cards: int
    controlled_hidden_vp_cards: int
    controlled_settlements: int
    controlled_cities: int
    controlled_roads: int
    brick_wood_balance: float
    truncated: bool


def _resource_count(env: CatanEnv, player_id: PlayerId, resource: Resource) -> int:
    return int(env.engine.players[player_id].resources.get(resource, 0))


def _brick_wood_balance(env: CatanEnv, player_id: PlayerId) -> float:
    wood = _resource_count(env, player_id, Resource.WOOD)
    brick = _resource_count(env, player_id, Resource.BRICK)
    total = wood + brick
    if total == 0:
        return 1.0
    return min(wood, brick) / max(wood, brick, 1)


def run_game(
    seed: int,
    max_steps: int,
    controlled_player: PlayerId = PlayerId.WHITE,
    allow_bank_trades: bool = True,
) -> GameResult:
    env = CatanEnv(seed=seed, enable_trading=False, max_steps=max_steps)
    agents = {
        player_id: RandomNoTradeBaselineAgent(
            seed=seed * 100 + int(player_id),
            allow_bank_trades=allow_bank_trades,
        )
        for player_id in env.engine.players
    }

    obs = env.reset()
    done = False
    rewards: list[float] = []
    build_opportunities = 0
    successful_builds = 0
    controlled_dev_cards_bought = 0
    steps = 0
    truncated = False

    while not done:
        current_player = env.get_current_player_id()
        legal_actions = obs.get("legal_actions", env.get_legal_actions())
        if any(action.get("type") in BUILD_ACTIONS for action in legal_actions):
            build_opportunities += 1

        action = agents[current_player].select_action(legal_actions)
        action_type = action.get("type") if action else None

        obs, reward, done, info = env.step(action)
        rewards.append(float(reward))
        steps += 1
        truncated = bool(info.get("truncated", False))

        if action_type in BUILD_ACTIONS and reward > 0:
            successful_builds += 1
        if current_player == controlled_player and action_type == "buy_dev_card" and reward > 0:
            controlled_dev_cards_bought += 1

    controlled_state = env.engine.players[controlled_player]
    controlled_vp = int(controlled_state.update_victory_points())
    winner_vp = (
        int(env.engine.players[env.engine.winner].update_victory_points())
        if env.engine.winner is not None
        else max(int(player.update_victory_points()) for player in env.engine.players.values())
    )
    build_efficiency = (
        successful_builds / build_opportunities
        if build_opportunities
        else 0.0
    )

    return GameResult(
        winner=env.engine.winner,
        controlled_vp=controlled_vp,
        winner_vp=winner_vp,
        turns=int(env.engine.turn_number),
        steps=steps,
        avg_reward=mean(rewards) if rewards else 0.0,
        build_efficiency=build_efficiency,
        controlled_longest_road=env.engine.longest_road_owner == controlled_player,
        controlled_largest_army=env.engine.largest_army_owner == controlled_player,
        controlled_longest_road_length=int(controlled_state.longest_road_length),
        controlled_played_knights=int(controlled_state.played_knights),
        controlled_dev_cards_bought=controlled_dev_cards_bought,
        controlled_held_dev_cards=len(controlled_state.dev_cards) + len(controlled_state.new_dev_cards),
        controlled_hidden_vp_cards=int(controlled_state.hidden_vp_cards),
        controlled_settlements=int(controlled_state.n_settlements),
        controlled_cities=int(controlled_state.n_cities),
        controlled_roads=int(controlled_state.n_roads),
        brick_wood_balance=_brick_wood_balance(env, controlled_player),
        truncated=truncated,
    )


def summarize(results: list[GameResult], controlled_player: PlayerId) -> dict[str, object]:
    games = len(results)
    wins = sum(1 for result in results if result.winner == controlled_player)
    return {
        "model": "Random / No-Trade Baseline",
        "games_played": games,
        "win_rate": wins / games if games else 0.0,
        "avg_victory_points": mean(result.controlled_vp for result in results) if games else 0.0,
        "avg_game_length_turns": mean(result.turns for result in results) if games else 0.0,
        "avg_reward": mean(result.avg_reward for result in results) if games else 0.0,
        "build_efficiency": mean(result.build_efficiency for result in results) if games else 0.0,
        "longest_road_rate": mean(float(result.controlled_longest_road) for result in results) if games else 0.0,
        "largest_army_rate": mean(float(result.controlled_largest_army) for result in results) if games else 0.0,
        "avg_played_knights": mean(result.controlled_played_knights for result in results) if games else 0.0,
        "avg_dev_cards_bought": mean(result.controlled_dev_cards_bought for result in results) if games else 0.0,
        "avg_hidden_vp_cards": mean(result.controlled_hidden_vp_cards for result in results) if games else 0.0,
        "avg_longest_road_length": mean(result.controlled_longest_road_length for result in results) if games else 0.0,
        "truncation_rate": mean(float(result.truncated) for result in results) if games else 0.0,
        "brick_wood_balance": mean(result.brick_wood_balance for result in results) if games else 0.0,
        "notes": "Real random legal-action self-play baseline with player trading disabled.",
    }


def write_summary_csv(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "games_played",
        "win_rate",
        "avg_victory_points",
        "avg_game_length_turns",
        "avg_reward",
        "build_efficiency",
        "longest_road_rate",
        "largest_army_rate",
        "avg_played_knights",
        "avg_dev_cards_bought",
        "avg_hidden_vp_cards",
        "avg_longest_road_length",
        "truncation_rate",
        "brick_wood_balance",
        "notes",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(summary)


def build_training_log_header(total_games: int) -> str:
    return "\n".join(
        [
            f"Starting Random / No-Trade Baseline self-play evaluation for {total_games} games",
            "Config: player_trades=disabled, policy=random legal action",
            "Primary metric: controlled-player win rate",
            "",
        ]
    )


def build_training_log_update(
    update: int,
    result: GameResult,
) -> str:
    controlled_win = result.winner == PlayerId.WHITE
    winner_name = result.winner.name if result.winner is not None else "NONE"

    return "\n".join(
        [
            f"Game {update}",
            (
                "result  | "
                f"winner={winner_name} controlled_win={int(controlled_win)} "
                f"truncated={int(result.truncated)} turns={result.turns} steps={result.steps}"
            ),
            (
                "score   | "
                f"controlled_vp={result.controlled_vp} winner_vp={result.winner_vp} "
                f"avg_reward={result.avg_reward:.6f}"
            ),
            (
                "army    | "
                f"played_knights={result.controlled_played_knights} "
                f"largest_army={int(result.controlled_largest_army)}"
            ),
            (
                "dev     | "
                f"bought={result.controlled_dev_cards_bought} "
                f"held={result.controlled_held_dev_cards} "
                f"hidden_vp={result.controlled_hidden_vp_cards}"
            ),
            (
                "builds  | "
                f"settlements={result.controlled_settlements} cities={result.controlled_cities} "
                f"roads={result.controlled_roads} longest_road_len={result.controlled_longest_road_length} "
                f"longest_road_award={int(result.controlled_longest_road)} "
                f"build_efficiency={result.build_efficiency:.6f}"
            ),
            "",
        ]
    )


def build_training_log_summary(results: list[GameResult], controlled_player: PlayerId) -> str:
    summary = summarize(results, controlled_player)
    return "\n".join(
        [
            "Summary",
            (
                "win     | "
                f"games={summary['games_played']} "
                f"win_rate={float(summary['win_rate']):.4f} "
                f"truncation_rate={float(summary['truncation_rate']):.4f}"
            ),
            (
                "score   | "
                f"avg_vp={float(summary['avg_victory_points']):.3f} "
                f"avg_turns={float(summary['avg_game_length_turns']):.3f} "
                f"avg_reward={float(summary['avg_reward']):.6f}"
            ),
            (
                "army    | "
                f"largest_army_rate={float(summary['largest_army_rate']):.4f} "
                f"avg_played_knights={float(summary['avg_played_knights']):.3f}"
            ),
            (
                "dev     | "
                f"avg_dev_cards_bought={float(summary['avg_dev_cards_bought']):.3f} "
                f"avg_hidden_vp_cards={float(summary['avg_hidden_vp_cards']):.3f}"
            ),
            (
                "builds  | "
                f"longest_road_rate={float(summary['longest_road_rate']):.4f} "
                f"avg_longest_road_length={float(summary['avg_longest_road_length']):.3f} "
                f"build_efficiency={float(summary['build_efficiency']):.4f}"
            ),
            "",
        ]
    )


def build_training_log_text(results: list[GameResult]) -> str:
    """
    Build a unified-style training log.

    This baseline is not trained, so optimization losses are logged as zero. The
    reward, rollout, trade-skip, and per-game quality signals are real game data.
    """

    lines: list[str] = [build_training_log_header(len(results))]

    for update, result in enumerate(results):
        lines.append(build_training_log_update(update, result))

    lines.append(build_training_log_summary(results, PlayerId.WHITE))

    return "\n".join(lines)


def write_training_log(results: list[GameResult], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_text = build_training_log_text(results)
    log_path.write_text(log_text, encoding="utf-8")
    return log_text


def write_text_with_fallback(path: Path, text: str) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(text, encoding="utf-8")
        return fallback


def _result_rows(results: list[GameResult]) -> list[dict[str, object]]:
    rows = []
    for idx, result in enumerate(results):
        rows.append(
            {
                "game": idx,
                "winner": result.winner.name if result.winner is not None else "NONE",
                "controlled_win": int(result.winner == PlayerId.WHITE),
                "controlled_vp": result.controlled_vp,
                "winner_vp": result.winner_vp,
                "turns": result.turns,
                "steps": result.steps,
                "avg_reward": result.avg_reward,
                "largest_army": int(result.controlled_largest_army),
                "played_knights": result.controlled_played_knights,
                "dev_cards_bought": result.controlled_dev_cards_bought,
                "held_dev_cards": result.controlled_held_dev_cards,
                "hidden_vp_cards": result.controlled_hidden_vp_cards,
                "longest_road_award": int(result.controlled_longest_road),
                "longest_road_length": result.controlled_longest_road_length,
                "settlements": result.controlled_settlements,
                "cities": result.controlled_cities,
                "roads": result.controlled_roads,
                "build_efficiency": result.build_efficiency,
                "brick_wood_balance": result.brick_wood_balance,
                "truncated": int(result.truncated),
            }
        )
    return rows


def write_game_results_csv(results: list[GameResult], output_path: Path) -> None:
    rows = _result_rows(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_text(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Random / No-Trade Baseline Summary",
        f"games: {summary['games_played']}",
        f"win_rate: {float(summary['win_rate']):.4f}",
        f"avg_victory_points: {float(summary['avg_victory_points']):.3f}",
        f"avg_game_length_turns: {float(summary['avg_game_length_turns']):.3f}",
        f"largest_army_rate: {float(summary['largest_army_rate']):.4f}",
        f"avg_played_knights: {float(summary['avg_played_knights']):.3f}",
        f"avg_dev_cards_bought: {float(summary['avg_dev_cards_bought']):.3f}",
        f"avg_hidden_vp_cards: {float(summary['avg_hidden_vp_cards']):.3f}",
        f"longest_road_rate: {float(summary['longest_road_rate']):.4f}",
        f"avg_longest_road_length: {float(summary['avg_longest_road_length']):.3f}",
        f"build_efficiency: {float(summary['build_efficiency']):.4f}",
        f"truncation_rate: {float(summary['truncation_rate']):.4f}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def plot_baseline_results(results: list[GameResult], summary: dict[str, object], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    def add_bar_labels(bars, values, value_format: str = "{:.2f}") -> None:
        max_value = max([float(value) for value in values] + [1.0])
        offset = max_value * 0.025
        axis = plt.gca()
        axis.set_ylim(top=max(axis.get_ylim()[1], max_value + offset * 4.0))
        for bar, value in zip(bars, values):
            numeric_value = float(value)
            label = str(int(numeric_value)) if numeric_value.is_integer() else value_format.format(numeric_value)
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + offset,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_game_results_csv(results, output_dir / "random_game_results.csv")
    write_summary_csv(summary, output_dir / "random_summary.csv")
    write_summary_text(summary, output_dir / "random_summary.txt")

    if not results:
        return

    games = int(summary["games_played"])
    wins = sum(1 for result in results if result.winner == PlayerId.WHITE)
    losses = games - wins
    truncated = sum(1 for result in results if result.truncated)
    completed = games - truncated

    plt.figure(figsize=(9, 6))
    bars = plt.bar(
        ["WHITE wins", "Other wins", "Truncated"],
        [wins, losses, truncated],
        color=["#2f855a", "#718096", "#d69e2e"],
    )
    plt.title(f"Random No-Trade Baseline Win Rate: {float(summary['win_rate']) * 100:.1f}%")
    plt.ylabel("Games")
    add_bar_labels(bars, [wins, losses, truncated], "{:.0f}")
    plt.tight_layout()
    plt.savefig(output_dir / "random_win_rate_overview.png", dpi=300)
    plt.close()

    winner_counts = {player.name: 0 for player in PlayerId}
    winner_counts["NONE"] = 0
    for result in results:
        winner_counts[result.winner.name if result.winner is not None else "NONE"] += 1
    labels = list(winner_counts.keys())
    values = [winner_counts[label] for label in labels]
    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, values, color="#4c78a8")
    plt.title("Winner Distribution")
    plt.ylabel("Games")
    add_bar_labels(bars, values, "{:.0f}")
    plt.tight_layout()
    plt.savefig(output_dir / "random_winner_distribution.png", dpi=300)
    plt.close()

    vps = [result.controlled_vp for result in results]
    turns = [result.turns for result in results]
    plt.figure(figsize=(10, 7))
    plt.subplot(2, 1, 1)
    plt.hist(vps, bins=range(0, max(max(vps) + 2, 12)), color="#2b6cb0", alpha=0.85)
    plt.axvline(float(summary["avg_victory_points"]), color="#1a202c", linestyle="--", label="Average VP")
    plt.title("Controlled Player Victory Points")
    plt.ylabel("Games")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(turns, bins=20, color="#805ad5", alpha=0.85)
    plt.axvline(float(summary["avg_game_length_turns"]), color="#1a202c", linestyle="--", label="Average turns")
    plt.title("Game Length")
    plt.xlabel("Turns")
    plt.ylabel("Games")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "random_vp_and_turns.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    labels = ["Largest Army Rate", "Avg Knights", "Avg Dev Cards", "Avg Hidden VP"]
    values = [
        float(summary["largest_army_rate"]),
        float(summary["avg_played_knights"]),
        float(summary["avg_dev_cards_bought"]),
        float(summary["avg_hidden_vp_cards"]),
    ]
    bars = plt.bar(labels, values, color=["#c05621", "#dd6b20", "#3182ce", "#38a169"])
    plt.title("Army and Development Card Outcomes")
    plt.ylabel("Rate or average count")
    plt.xticks(rotation=10)
    add_bar_labels(bars, values)
    plt.tight_layout()
    plt.savefig(output_dir / "random_army_and_dev_cards.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    labels = ["Settlements", "Cities", "Roads", "Longest Road Len", "Build Eff."]
    values = [
        mean(result.controlled_settlements for result in results),
        mean(result.controlled_cities for result in results),
        mean(result.controlled_roads for result in results),
        float(summary["avg_longest_road_length"]),
        float(summary["build_efficiency"]),
    ]
    bars = plt.bar(labels, values, color=["#38a169", "#2f855a", "#d69e2e", "#b7791f", "#718096"])
    plt.title("Construction Outcomes")
    plt.ylabel("Average count or rate")
    plt.xticks(rotation=10)
    add_bar_labels(bars, values)
    plt.tight_layout()
    plt.savefig(output_dir / "random_construction_outcomes.png", dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Random / No-Trade baseline.")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("evaluation/random_no_trade_baseline_real.csv"))
    parser.add_argument("--log-file", type=Path, default=Path("random_training_log.txt"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/random_figures"))
    parser.add_argument("--plot", action="store_true", help="Generate the unified-style diagnostic plots after the run.")
    parser.add_argument("--stdout-log", action="store_true", help="Print the training log to stdout for shell redirection.")
    parser.add_argument("--no-bank-trades", action="store_true")
    args = parser.parse_args()

    controlled_player = PlayerId.WHITE
    results: list[GameResult] = []

    if args.stdout_log:
        print(build_training_log_header(args.games), flush=True)

    for game_idx in range(args.games):
        result = run_game(
            seed=args.seed + game_idx,
            max_steps=args.max_steps,
            controlled_player=controlled_player,
            allow_bank_trades=not args.no_bank_trades,
        )
        results.append(result)

        if args.stdout_log:
            print(
                build_training_log_update(game_idx, result),
                flush=True,
            )

    summary = summarize(results, controlled_player)
    write_summary_csv(summary, args.output)
    log_text = build_training_log_text(results)

    if args.stdout_log:
        print(build_training_log_summary(results, controlled_player), flush=True)

    if not args.stdout_log:
        args.log_file = write_text_with_fallback(args.log_file, log_text)

    if args.plot:
        plot_baseline_results(results, summary, args.figures_dir)

    status_stream = sys.stderr if args.stdout_log else sys.stdout
    truncated = sum(1 for result in results if result.truncated)
    print("Random / No-Trade baseline complete", file=status_stream)
    print(f"Games: {summary['games_played']} ({truncated} truncated at max_steps)", file=status_stream)
    print(f"Win rate for {controlled_player.name}: {summary['win_rate']:.3f}", file=status_stream)
    print(f"Avg VP: {summary['avg_victory_points']:.2f}", file=status_stream)
    print(f"Avg turns: {summary['avg_game_length_turns']:.2f}", file=status_stream)
    print(f"Avg reward: {summary['avg_reward']:.4f}", file=status_stream)
    print(f"Wrote: {args.output}", file=status_stream)
    if args.stdout_log and not args.plot:
        print("Wrote log: stdout", file=status_stream)
    else:
        print(f"Wrote log: {args.log_file}", file=status_stream)
    if args.plot:
        print(f"Wrote figures: {args.figures_dir}", file=status_stream)


if __name__ == "__main__":
    main()
