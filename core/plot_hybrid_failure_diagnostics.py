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
DQN_RE = re.compile(
    r"dqn\s*\|\s*epsilon=([-+]?\d*\.?\d+)\s+td_loss=([-+]?\d*\.?\d+)\s+q_mean=([-+]?\d*\.?\d+)"
)
TRADE_RE = re.compile(
    r"trade\s*\|\s*policy=([-+]?\d*\.?\d+)\s+value=([-+]?\d*\.?\d+)\s+entropy=([-+]?\d*\.?\d+)\s+\(coef=([-+]?\d*\.?\d+)\)\s+tom=([-+]?\d*\.?\d+)"
)
TRADE_ACTIONS_RE = re.compile(
    r"trade actions\s*\|\s*propose=(\d+)\s+accept=(\d+)\s+reject=(\d+)\s+counter=(\d+)\s+skip=(\d+)"
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
        last_error = None
        text = None

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

        m = UPDATE_RE.match(line)
        if m:
            if current:
                rows.append(current)
            current = {"update": int(m.group(1))}
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

        m = DQN_RE.search(line)
        if m:
            current["epsilon"] = float(m.group(1))
            current["td_loss"] = float(m.group(2))
            current["q_mean"] = float(m.group(3))
            continue

        m = TRADE_RE.search(line)
        if m:
            current["trade_policy_loss"] = float(m.group(1))
            current["trade_value_loss"] = float(m.group(2))
            current["trade_entropy"] = float(m.group(3))
            current["trade_entropy_coef"] = float(m.group(4))
            current["trade_tom_loss"] = float(m.group(5))
            continue

        m = TRADE_ACTIONS_RE.search(line)
        if m:
            current["trade_propose"] = int(m.group(1))
            current["trade_accept"] = int(m.group(2))
            current["trade_reject"] = int(m.group(3))
            current["trade_counter"] = int(m.group(4))
            current["trade_skip"] = int(m.group(5))
            continue

    if current:
        rows.append(current)

    if not rows:
        sample = "\n".join(lines[:20])
        raise ValueError(
            "No updates were parsed from the log file. "
            "The file may still be encoded unexpectedly or may not contain training output.\n\n"
            f"First lines seen:\n{sample}"
        )

    df = pd.DataFrame(rows).sort_values("update").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "update"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "trade_reward" in out.columns:
        out["trade_reward_ma10"] = moving_average(out["trade_reward"], 10)

    if "gameplay_reward" in out.columns:
        out["gameplay_reward_ma10"] = moving_average(out["gameplay_reward"], 10)

    if "trade_entropy" in out.columns:
        out["trade_entropy_ma10"] = moving_average(out["trade_entropy"], 10)

    if "trade_value_loss" in out.columns:
        out["trade_value_loss_ma10"] = moving_average(out["trade_value_loss"], 10)

    if "td_loss" in out.columns:
        out["td_loss_ma10"] = moving_average(out["td_loss"], 10)

    if "q_mean" in out.columns:
        out["q_mean_ma10"] = moving_average(out["q_mean"], 10)

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

    return out


def save_summary(df: pd.DataFrame, output_dir: str) -> None:
    lines = []
    lines.append(f"Parsed updates: {len(df)}")
    lines.append(f"Update range: {int(df['update'].min())} to {int(df['update'].max())}")

    for col in ["gameplay_reward", "trade_reward", "trade_entropy", "trade_value_loss", "td_loss", "q_mean"]:
        if col in df.columns and df[col].notna().any():
            lines.append(f"{col} mean: {df[col].mean():.6f}")
            lines.append(f"{col} min: {df[col].min():.6f}")
            lines.append(f"{col} max: {df[col].max():.6f}")

    with open(os.path.join(output_dir, "hybrid_failure_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_trade_reward(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["trade_reward"], alpha=0.35, label="Per-update reward")
    plt.plot(df["update"], df["trade_reward_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, label="Zero reward")
    plt.title("Hybrid Model Trade Reward")
    plt.xlabel("Training update")
    plt.ylabel("Mean trade reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_trade_reward_curve.png"), dpi=300)
    plt.close()


def plot_trade_entropy(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["trade_entropy"], alpha=0.35, label="Per-update entropy")
    plt.plot(df["update"], df["trade_entropy_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.title("Hybrid Model Trade Entropy")
    plt.xlabel("Training update")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_trade_entropy_curve.png"), dpi=300)
    plt.close()


def plot_trade_actions(df: pd.DataFrame, output_dir: str) -> None:
    if "trade_propose_rate" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["update"], df["trade_propose_rate"], label="Propose rate")
    plt.plot(df["update"], df["trade_accept_rate"], label="Accept rate")
    plt.plot(df["update"], df["trade_reject_rate"], label="Reject rate")
    plt.plot(df["update"], df["trade_counter_rate"], label="Counter rate")
    plt.plot(df["update"], df["trade_skip_rate"], label="Skip rate")
    plt.title("Hybrid Model Trade Action Distribution")
    plt.xlabel("Training update")
    plt.ylabel("Fraction of trade decisions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_trade_action_distribution.png"), dpi=300)
    plt.close()


def plot_trade_value_loss(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["trade_value_loss"], alpha=0.35, label="Per-update value loss")
    plt.plot(df["update"], df["trade_value_loss_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.title("Hybrid Model Trade Value Loss")
    plt.xlabel("Training update")
    plt.ylabel("Value loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_trade_value_loss_curve.png"), dpi=300)
    plt.close()


def plot_reward_comparison(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["update"], df["gameplay_reward_ma10"], linewidth=2.0, label="Gameplay reward MA(10)")
    plt.plot(df["update"], df["trade_reward_ma10"], linewidth=2.0, label="Trade reward MA(10)")
    plt.title("Hybrid Model Gameplay vs Trade Reward")
    plt.xlabel("Training update")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_gameplay_vs_trade_reward.png"), dpi=300)
    plt.close()


def plot_dqn_metrics(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["td_loss"], alpha=0.35, label="Per-update TD loss")
    plt.plot(df["update"], df["td_loss_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.title("Hybrid Model Gameplay TD Loss")
    plt.xlabel("Training update")
    plt.ylabel("TD loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["q_mean"], alpha=0.35, label="Per-update Q mean")
    plt.plot(df["update"], df["q_mean_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.title("Hybrid Model Gameplay Q Mean")
    plt.xlabel("Training update")
    plt.ylabel("Q mean")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_dqn_metrics.png"), dpi=300)
    plt.close()


def plot_entropy_reward_overview(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["trade_reward"], alpha=0.35, label="Per-update reward")
    plt.plot(df["update"], df["trade_reward_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.title("Trade Reward")
    plt.xlabel("Training update")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["trade_entropy"], alpha=0.35, label="Per-update entropy")
    plt.plot(df["update"], df["trade_entropy_ma10"], linewidth=2.0, label="10-step moving avg")
    plt.title("Trade Entropy")
    plt.xlabel("Training update")
    plt.ylabel("Entropy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "hybrid_trade_reward_entropy_overview.png"), dpi=300)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse hybrid DQN+PPO logs and generate figures.")
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="hybrid_failure_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = parse_training_log(args.log_file)
    df = add_derived_metrics(df)

    csv_path = os.path.join(args.output_dir, "parsed_hybrid_training_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    save_summary(df, args.output_dir)

    plot_trade_reward(df, args.output_dir)
    plot_trade_entropy(df, args.output_dir)
    plot_trade_actions(df, args.output_dir)
    plot_trade_value_loss(df, args.output_dir)
    plot_reward_comparison(df, args.output_dir)
    plot_dqn_metrics(df, args.output_dir)
    plot_entropy_reward_overview(df, args.output_dir)

    print(f"Saved figures and CSV to: {args.output_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()