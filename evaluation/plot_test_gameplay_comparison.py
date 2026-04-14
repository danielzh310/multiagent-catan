from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "Unified PPO": "#1f6f5f",
    "DQN Gameplay": "#d17a22",
    "Random / No-Trade Baseline": "#6c757d",
}


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected = {
        "model",
        "win_rate",
        "avg_victory_points",
        "avg_game_length_turns",
        "avg_reward",
        "build_efficiency",
        "longest_road_rate",
        "largest_army_rate",
    }
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def make_figure(df: pd.DataFrame, output_path: str) -> None:
    models = df["model"].tolist()
    colors = [COLORS.get(model, "#4c78a8") for model in models]

    fig = plt.figure(figsize=(15.5, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], wspace=0.55, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    win_rates = df["win_rate"] * 100.0
    bars = ax1.bar(models, win_rates, color=colors)
    ax1.set_title("Win Rate")
    ax1.set_ylabel("Percent")
    ax1.set_ylim(0, max(win_rates) * 1.25)
    for bar, value in zip(bars, win_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}%", ha="center", va="bottom", fontsize=10)
    ax1.tick_params(axis="x", rotation=12)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(models, df["avg_victory_points"], color=colors)
    ax2.set_title("Avg Victory Points")
    ax2.set_ylabel("Points")
    ax2.set_ylim(0, max(df["avg_victory_points"]) * 1.25)
    for x_pos, value in enumerate(df["avg_victory_points"]):
        ax2.text(x_pos, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    ax2.tick_params(axis="x", rotation=12)

    ax3 = fig.add_subplot(gs[1, 0])
    radar_metrics = [
        ("Avg Reward", "avg_reward"),
        ("Build Eff.", "build_efficiency"),
        ("Longest Rd", "longest_road_rate"),
        ("Largest Army", "largest_army_rate"),
    ]
    metric_labels = [label for label, _ in radar_metrics]
    x = range(len(metric_labels))
    width = 0.22
    offsets = [-width, 0.0, width]
    for offset, (_, series) in zip(offsets, df.iterrows()):
        vals = [series[col] for _, col in radar_metrics]
        ax3.bar([i + offset for i in x], vals, width=width, color=COLORS.get(series["model"], "#4c78a8"), label=series["model"])
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(metric_labels)
    ax3.set_ylim(0, 1.0)
    ax3.set_title("Gameplay Quality Signals")
    ax3.legend(frameon=False, loc="upper left")

    ax4 = fig.add_subplot(gs[1, 1])
    norm_turns = 1.0 - (df["avg_game_length_turns"] - df["avg_game_length_turns"].min()) / (
        max(df["avg_game_length_turns"].max() - df["avg_game_length_turns"].min(), 1e-9)
    )
    scorecard = pd.DataFrame(
        {
            "Win Rate": df["win_rate"].to_list(),
            "Avg VP": (df["avg_victory_points"] / 10.0).to_list(),
            "Reward": (df["avg_reward"] / max(df["avg_reward"].max(), 1e-9)).to_list(),
            "Build Eff.": df["build_efficiency"].to_list(),
            "Shorter Games": norm_turns.to_list(),
        },
        index=models,
    )
    im = ax4.imshow(scorecard.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)
    ax4.set_xticks(range(scorecard.shape[1]))
    ax4.set_xticklabels(scorecard.columns, rotation=20, ha="right")
    ax4.set_yticks(range(scorecard.shape[0]))
    ax4.set_yticklabels(scorecard.index)
    ax4.tick_params(axis="y", pad=16)
    pos = ax4.get_position()
    ax4.set_position([pos.x0 + 0.035, pos.y0, pos.width - 0.035, pos.height])
    ax4.set_title("Normalized Comparison Scorecard")
    for i in range(scorecard.shape[0]):
        for j in range(scorecard.shape[1]):
            ax4.text(j, i, f"{scorecard.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    fig.suptitle("Gameplay Comparison: Unified vs DQN vs Baseline", fontsize=16, y=0.98)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot test gameplay comparison figure.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = load_data(args.input_csv)
    make_figure(df, args.output)
    print(f"Saved comparison figure to {args.output}")


if __name__ == "__main__":
    main()
