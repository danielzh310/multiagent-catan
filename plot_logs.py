import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path: Path) -> dict:
    """Parses a training log file to extract key metrics from various algorithm outputs."""
    metrics = defaultdict(list)
    updates = []

    # Regex patterns to find specific lines and extract values from different log formats
    patterns = {
        "update": re.compile(r"(?:^Update|^Step)\s+(\d+)"),
        "win_rate": re.compile(r"Agent Wins: \d+ \(([\d\.]+)%\)|Win Rate: ([\d\.]+)"),
        "gameplay_reward": re.compile(r"gameplay=([\d\.-]+)"),
        "trade_reward": re.compile(r"trade=([\d\.-]+)"),
        "gameplay_policy_loss": re.compile(r"(?:ppo gameplay\s*\|\s*policy|gp_policy)=([\d\.-]+)"),
        "gameplay_value_loss": re.compile(r"(?:ppo gameplay\s*\|.*value|gp_value)=([\d\.-]+)"),
        "gameplay_entropy": re.compile(r"(?:ppo gameplay\s*\|.*entropy|gp_entropy)=([\d\.-]+)"),
        "trade_policy_loss": re.compile(r"(?:ppo trade\s*\|\s*policy|tr_policy)=([\d\.-]+)"),
        "trade_value_loss": re.compile(r"(?:ppo trade\s*\|.*value|tr_value)=([\d\.-]+)"),
        "trade_entropy": re.compile(r"(?:ppo trade\s*\|.*entropy|tr_entropy)=([\d\.-]+)"),
        "tom_loss": re.compile(r"tom=([\d\.-]+)"),
        "total_loss": re.compile(r"(?:total_loss|ppo total\s*\|\s*loss)=([\d\.-]+)"),
        "td_loss": re.compile(r"td_loss=([\d\.-]+)"),
        "q_mean": re.compile(r"q_mean=([\d\.-]+)"),
        "eval_score": re.compile(r"Eval - Score: ([\d\.-]+)"),
    }

    current_update = -1
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            update_match = patterns["update"].search(line)
            if update_match:
                new_update_val = int(update_match.group(1))
                if new_update_val > current_update:
                    current_update = new_update_val
                    updates.append(current_update)
                    # Add placeholder for all potential metrics
                    for key in patterns:
                        if key != "update":
                            metrics[key].append(np.nan)

            if current_update == -1:
                continue

            # Parse all other metrics from the line
            for key, pattern in patterns.items():
                if key == "update":
                    continue

                match = pattern.search(line)
                if match:
                    # Find the first non-None captured group to handle patterns with OR |
                    value_str = next((g for g in match.groups() if g is not None), None)
                    if value_str is not None:
                        # Overwrite the NaN placeholder for the current update step
                        if len(metrics[key]) == len(updates):
                            metrics[key][-1] = float(value_str)

    metrics["updates"] = updates
    return metrics


def plot_metrics(metrics: dict, output_dir: Path):
    """Generates and saves plots for the extracted metrics."""
    output_dir.mkdir(exist_ok=True)
    updates = metrics.get("updates", [])
    if not updates:
        print("No updates found in log file. Cannot generate plots.")
        return

    for key, values in metrics.items():
        if key == "updates" or len(values) != len(updates):
            continue

        # Don't create a plot for a metric that was never found
        if all(np.isnan(v) for v in values):
            continue

        plt.figure(figsize=(12, 6))
        plt.plot(updates, values, label=key)

        # Simple moving average
        # Only plot if there are enough data points to make an average meaningful
        if np.count_nonzero(~np.isnan(values)) > 50:
            moving_avg = np.convolve(np.nan_to_num(values), np.ones(50)/50, mode='valid')
            avg_updates = updates[len(updates) - len(moving_avg):]
            plt.plot(avg_updates, moving_avg, label=f'{key} (50-step avg)', linestyle='--')

        plt.xlabel("Training Updates / Steps")
        plt.ylabel(key.replace("_", " ").title())
        plt.title(f"Training Progress: {key.replace('_', ' ').title()}")
        plt.grid(True)
        plt.legend()
        save_path = output_dir / f"{key}_plot.png"
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from a log file.")
    parser.add_argument("log_file", type=str, help="Path to the training log file.")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        exit(1)

    output_path = log_path.parent / f"{log_path.stem}_plots"

    parsed_metrics = parse_log_file(log_path)
    plot_metrics(parsed_metrics, output_path)