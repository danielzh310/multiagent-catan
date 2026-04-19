from __future__ import annotations

import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


UPDATE_RE = re.compile(r".*Update\s+(\d+).*")
ROLLOUTS_RE = re.compile(r"rollouts\s*\|\s*gameplay=(\d+)\s+trade=(\d+)")
REWARD_RE = re.compile(
    r"reward\s*\|\s*gameplay=([-+]?\d*\.?\d+)\s+\(avg=([-+]?\d*\.?\d+)\)\s+trade=([-+]?\d*\.?\d+)\s+\(avg=([-+]?\d*\.?\d+)\)"
)
TRADE_ACTIONS_RE = re.compile(
    r"trade actions\s*\|\s*propose=(\d+)\s+accept=(\d+)\s+reject=(\d+)\s+counter=(\d+)\s+skip=(\d+)"
)
ENTROPY_RE = re.compile(
    r"entropy\s*\|\s*gameplay=([-+]?\d*\.?\d+)\s+\(coef=([-+]?\d*\.?\d+)\)\s+trade=([-+]?\d*\.?\d+)\s+\(avg=([-+]?\d*\.?\d+),\s*coef=([-+]?\d*\.?\d+)\)"
)
LOSSES_RE = re.compile(
    r"losses\s*\|\s*gp_policy=([-+]?\d*\.?\d+)\s+gp_value=([-+]?\d*\.?\d+)\s+tr_policy=([-+]?\d*\.?\d+)\s+tr_value=([-+]?\d*\.?\d+)\s+tr_tom=([-+]?\d*\.?\d+)"
)


def moving_average(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def read_text_with_fallbacks(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find log file: {path}")

    with open(path, "rb") as f:
        raw = f.read()

    if len(raw) == 0:
        raise ValueError(f"Log file is empty: {path}")

    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]
        text = None
        last_error = None
        for encoding in encodings:
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError as e:
                last_error = e

        if text is None:
            raise UnicodeDecodeError(
                "unknown",
                b"",
                0,
                1,
                f"Could not decode file with tried encodings. Last error: {last_error}",
            )

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def parse_training_log(log_path: str) -> pd.DataFrame:
    text = read_text_with_fallbacks(log_path)
    lines = text.splitlines()

    rows: List[Dict] = []
    current: Dict = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        update_match = UPDATE_RE.match(line)
        if update_match:
            if current:
                rows.append(current)
            current = {"update": int(update_match.group(1))}
            continue

        if not current:
            continue

        m = ROLLOUTS_RE.search(line)
        if m:
            current["gameplay_rollouts"] = int(m.group(1))
            current["trade_rollouts"] = int(m.group(2))
            continue

        m = REWARD_RE.search(line)
        if m:
            current["gameplay_reward"] = float(m.group(1))
            current["gameplay_reward_avg_logged"] = float(m.group(2))
            current["trade_reward"] = float(m.group(3))
            current["trade_reward_avg_logged"] = float(m.group(4))
            continue

        m = TRADE_ACTIONS_RE.search(line)
        if m:
            current["trade_propose"] = int(m.group(1))
            current["trade_accept"] = int(m.group(2))
            current["trade_reject"] = int(m.group(3))
            current["trade_counter"] = int(m.group(4))
            current["trade_skip"] = int(m.group(5))
            continue

        m = ENTROPY_RE.search(line)
        if m:
            current["gameplay_entropy"] = float(m.group(1))
            current["gameplay_entropy_coef"] = float(m.group(2))
            current["trade_entropy"] = float(m.group(3))
            current["trade_entropy_avg_logged"] = float(m.group(4))
            current["trade_entropy_coef"] = float(m.group(5))
            continue

        m = LOSSES_RE.search(line)
        if m:
            current["gp_policy_loss"] = float(m.group(1))
            current["gp_value_loss"] = float(m.group(2))
            current["tr_policy_loss"] = float(m.group(3))
            current["tr_value_loss"] = float(m.group(4))
            current["tr_tom_loss"] = float(m.group(5))
            continue

    if current:
        rows.append(current)

    if not rows:
        raise ValueError(
            "No updates were parsed from the log file. "
            "Check that the file contains lines like 'Update 53' or '===== Update 53 ====='."
        )

    df = pd.DataFrame(rows).sort_values("update").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "update"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if {"trade_propose", "trade_accept", "trade_reject", "trade_counter", "trade_skip"}.issubset(out.columns):
        total_trade_actions = (
            out["trade_propose"]
            + out["trade_accept"]
            + out["trade_reject"]
            + out["trade_counter"]
            + out["trade_skip"]
        ).replace(0, pd.NA)

        out["trade_propose_rate"] = out["trade_propose"] / total_trade_actions
        out["trade_accept_rate"] = out["trade_accept"] / total_trade_actions
        out["trade_reject_rate"] = out["trade_reject"] / total_trade_actions
        out["trade_counter_rate"] = out["trade_counter"] / total_trade_actions
        out["trade_skip_rate"] = out["trade_skip"] / total_trade_actions

        denom = out["trade_propose"].replace(0, pd.NA)
        out["trade_accept_given_propose"] = out["trade_accept"] / denom

    if "trade_reward" in out.columns:
        out["trade_reward_ma10"] = moving_average(out["trade_reward"], 10)

    if "gameplay_reward" in out.columns:
        out["gameplay_reward_ma10"] = moving_average(out["gameplay_reward"], 10)

    if "trade_entropy" in out.columns:
        out["trade_entropy_ma10"] = moving_average(out["trade_entropy"], 10)

    if "tr_value_loss" in out.columns:
        out["tr_value_loss_ma10"] = moving_average(out["tr_value_loss"], 10)

    if "gp_value_loss" in out.columns:
        out["gp_value_loss_ma10"] = moving_average(out["gp_value_loss"], 10)

    return out


def save_summary(df: pd.DataFrame, output_dir: str) -> None:
    summary_lines = []

    summary_lines.append(f"Parsed updates: {len(df)}")
    summary_lines.append(f"Update range: {int(df['update'].min())} to {int(df['update'].max())}")

    if "trade_reward" in df.columns:
        summary_lines.append(f"Mean trade reward: {df['trade_reward'].mean():.6f}")
        summary_lines.append(f"Min trade reward: {df['trade_reward'].min():.6f}")
        summary_lines.append(f"Max trade reward: {df['trade_reward'].max():.6f}")

    if "trade_entropy" in df.columns:
        summary_lines.append(f"First trade entropy: {df['trade_entropy'].iloc[0]:.6f}")
        summary_lines.append(f"Last trade entropy: {df['trade_entropy'].iloc[-1]:.6f}")
        summary_lines.append(f"Min trade entropy: {df['trade_entropy'].min():.6f}")
        summary_lines.append(f"Max trade entropy: {df['trade_entropy'].max():.6f}")

    if "tr_value_loss" in df.columns:
        summary_lines.append(f"Mean trade value loss: {df['tr_value_loss'].mean():.6f}")
        summary_lines.append(f"Min trade value loss: {df['tr_value_loss'].min():.6f}")
        summary_lines.append(f"Max trade value loss: {df['tr_value_loss'].max():.6f}")

    if "trade_skip_rate" in df.columns:
        summary_lines.append(f"Mean trade skip rate: {df['trade_skip_rate'].mean():.6f}")
        summary_lines.append(f"Mean trade propose rate: {df['trade_propose_rate'].mean():.6f}")
        summary_lines.append(f"Mean trade accept rate: {df['trade_accept_rate'].mean():.6f}")

    with open(os.path.join(output_dir, "failure_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))


def make_trade_reward_plot(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["trade_reward"], alpha=0.35, label="Per-update reward")
    plt.plot(df["update"], df["trade_reward_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, label="Zero reward")
    plt.title("Trade Reward Curve")
    plt.xlabel("Training update")
    plt.ylabel("Mean trade reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trade_reward_curve.png"), dpi=300)
    plt.close()


def make_trade_entropy_plot(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["trade_entropy"], alpha=0.4, label="Per-update entropy")
    plt.plot(df["update"], df["trade_entropy_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.title("Trade Policy Entropy")
    plt.xlabel("Training update")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trade_entropy_curve.png"), dpi=300)
    plt.close()


def make_trade_action_rate_plot(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(df["update"], df["trade_propose_rate"], label="Propose rate")
    plt.plot(df["update"], df["trade_accept_rate"], label="Accept rate")
    plt.plot(df["update"], df["trade_reject_rate"], label="Reject rate")
    plt.plot(df["update"], df["trade_counter_rate"], label="Counter rate")
    plt.plot(df["update"], df["trade_skip_rate"], label="Skip rate")
    plt.title("Trade Action Distribution Over Time")
    plt.xlabel("Training update")
    plt.ylabel("Fraction of trade decisions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trade_action_distribution.png"), dpi=300)
    plt.close()


def make_trade_value_loss_plot(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["tr_value_loss"], alpha=0.4, label="Per-update value loss")
    plt.plot(df["update"], df["tr_value_loss_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.title("Trade Value Loss")
    plt.xlabel("Training update")
    plt.ylabel("Value loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trade_value_loss_curve.png"), dpi=300)
    plt.close()


def make_reward_comparison_plot(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["gameplay_reward_ma10"], linewidth=2.0, label="Gameplay reward MA(10)")
    plt.plot(df["update"], df["trade_reward_ma10"], linewidth=2.0, label="Trade reward MA(10)")
    plt.title("Gameplay vs Trade Reward")
    plt.xlabel("Training update")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gameplay_vs_trade_reward.png"), dpi=300)
    plt.close()


def make_two_panel_overview(df: pd.DataFrame, output_dir: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(df["update"], df["trade_reward"], alpha=0.3, label="Per-update reward")
    axes[0].plot(df["update"], df["trade_reward_ma10"], linewidth=2.0, label="10-step moving avg")
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("Trade Reward")
    axes[0].set_xlabel("Training update")
    axes[0].set_ylabel("Mean reward")
    axes[0].legend()

    axes[1].plot(df["update"], df["trade_entropy"], alpha=0.3, label="Per-update entropy")
    axes[1].plot(df["update"], df["trade_entropy_ma10"], linewidth=2.0, label="10-step moving avg")
    axes[1].set_title("Trade Entropy")
    axes[1].set_xlabel("Training update")
    axes[1].set_ylabel("Entropy")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trade_reward_entropy_overview.png"), dpi=300)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse training logs and generate failure-diagnostic figures.")
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to the saved training log text file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures/failure_figures",
        help="Directory to save CSV summaries and figures.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = parse_training_log(args.log_file)
    df = add_derived_metrics(df)

    df.to_csv(os.path.join(args.output_dir, "parsed_training_metrics.csv"), index=False)
    save_summary(df, args.output_dir)

    required_cols = {
        "trade_reward",
        "trade_entropy",
        "trade_propose_rate",
        "trade_accept_rate",
        "trade_reject_rate",
        "trade_counter_rate",
        "trade_skip_rate",
        "tr_value_loss",
        "gameplay_reward",
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required parsed columns: {missing}")

    make_trade_reward_plot(df, args.output_dir)
    make_trade_entropy_plot(df, args.output_dir)
    make_trade_action_rate_plot(df, args.output_dir)
    make_trade_value_loss_plot(df, args.output_dir)
    make_reward_comparison_plot(df, args.output_dir)
    make_two_panel_overview(df, args.output_dir)

    print(f"Saved figures and CSV to: {args.output_dir}")


if __name__ == "__main__":
    main()
