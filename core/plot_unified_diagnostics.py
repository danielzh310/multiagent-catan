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
PPO_GAMEPLAY_RE = re.compile(
    r"ppo gameplay\s*\|\s*policy=([-+]?\d*\.?\d+)\s+value=([-+]?\d*\.?\d+)\s+entropy=([-+]?\d*\.?\d+)\s+\(avg=([-+]?\d*\.?\d+)\)"
)
PPO_TRADE_RE = re.compile(
    r"ppo trade\s*\|\s*policy=([-+]?\d*\.?\d+)\s+value=([-+]?\d*\.?\d+)\s+entropy=([-+]?\d*\.?\d+)\s+\(avg=([-+]?\d*\.?\d+)\)\s+tom=([-+]?\d*\.?\d+)"
)
PPO_TOTAL_RE = re.compile(
    r"ppo total\s*\|\s*loss=([-+]?\d*\.?\d+)\s+entropy_coef=([-+]?\d*\.?\d+)\s+gamma=([-+]?\d*\.?\d+)"
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

        m = TRADE_ACTIONS_RE.search(line)
        if m:
            current["trade_propose"] = int(m.group(1))
            current["trade_accept"] = int(m.group(2))
            current["trade_reject"] = int(m.group(3))
            current["trade_counter"] = int(m.group(4))
            current["trade_skip"] = int(m.group(5))
            continue

        m = PPO_GAMEPLAY_RE.search(line)
        if m:
            current["gameplay_policy_loss"] = float(m.group(1))
            current["gameplay_value_loss"] = float(m.group(2))
            current["gameplay_entropy"] = float(m.group(3))
            current["gameplay_entropy_avg_logged"] = float(m.group(4))
            continue

        m = PPO_TRADE_RE.search(line)
        if m:
            current["trade_policy_loss"] = float(m.group(1))
            current["trade_value_loss"] = float(m.group(2))
            current["trade_entropy"] = float(m.group(3))
            current["trade_entropy_avg_logged"] = float(m.group(4))
            current["tom_loss"] = float(m.group(5))
            continue

        m = PPO_TOTAL_RE.search(line)
        if m:
            current["total_loss"] = float(m.group(1))
            current["entropy_coef"] = float(m.group(2))
            current["gamma"] = float(m.group(3))
            continue

    if current:
        rows.append(current)

    if not rows:
        raise ValueError("No updates were parsed from the log file.")

    df = pd.DataFrame(rows).sort_values("update").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "update"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "gameplay_reward",
        "trade_reward",
        "gameplay_entropy",
        "trade_entropy",
        "gameplay_value_loss",
        "trade_value_loss",
        "gameplay_policy_loss",
        "trade_policy_loss",
        "tom_loss",
        "total_loss",
    ]:
        if col in out.columns:
            out[f"{col}_ma10"] = moving_average(out[col], 10)

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

    for col in [
        "gameplay_reward",
        "trade_reward",
        "gameplay_entropy",
        "trade_entropy",
        "gameplay_value_loss",
        "trade_value_loss",
        "tom_loss",
        "total_loss",
    ]:
        if col in df.columns and df[col].notna().any():
            lines.append(f"{col} mean: {df[col].mean():.6f}")
            lines.append(f"{col} min: {df[col].min():.6f}")
            lines.append(f"{col} max: {df[col].max():.6f}")

    with open(os.path.join(output_dir, "unified_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _save(fig_name: str, output_dir: str) -> None:
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, fig_name), dpi=300)
    plt.close()


def plot_reward_overview(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["gameplay_reward"], alpha=0.35, label="Per-update gameplay reward")
    plt.plot(df["update"], df["gameplay_reward_ma10"], linewidth=2.0, label="Gameplay reward MA(10)")
    plt.title("Gameplay Reward")
    plt.xlabel("Training update")
    plt.ylabel("Reward")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["trade_reward"], alpha=0.35, label="Per-update trade reward")
    plt.plot(df["update"], df["trade_reward_ma10"], linewidth=2.0, label="Trade reward MA(10)")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.title("Trade Reward")
    plt.xlabel("Training update")
    plt.ylabel("Reward")
    plt.legend()

    _save("unified_reward_overview.png", output_dir)


def plot_entropy_overview(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["gameplay_entropy"], alpha=0.35, label="Per-update gameplay entropy")
    plt.plot(df["update"], df["gameplay_entropy_ma10"], linewidth=2.0, label="Gameplay entropy MA(10)")
    plt.title("Gameplay Entropy")
    plt.xlabel("Training update")
    plt.ylabel("Entropy")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["trade_entropy"], alpha=0.35, label="Per-update trade entropy")
    plt.plot(df["update"], df["trade_entropy_ma10"], linewidth=2.0, label="Trade entropy MA(10)")
    plt.title("Trade Entropy")
    plt.xlabel("Training update")
    plt.ylabel("Entropy")
    plt.legend()

    _save("unified_entropy_overview.png", output_dir)


def plot_value_losses(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["gameplay_value_loss"], alpha=0.35, label="Per-update gameplay value loss")
    plt.plot(df["update"], df["gameplay_value_loss_ma10"], linewidth=2.0, label="Gameplay value MA(10)")
    plt.title("Gameplay Value Loss")
    plt.xlabel("Training update")
    plt.ylabel("Value loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["trade_value_loss"], alpha=0.35, label="Per-update trade value loss")
    plt.plot(df["update"], df["trade_value_loss_ma10"], linewidth=2.0, label="Trade value MA(10)")
    plt.title("Trade Value Loss")
    plt.xlabel("Training update")
    plt.ylabel("Value loss")
    plt.legend()

    _save("unified_value_losses.png", output_dir)


def plot_policy_losses(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["gameplay_policy_loss"], alpha=0.35, label="Per-update gameplay policy loss")
    plt.plot(df["update"], df["gameplay_policy_loss_ma10"], linewidth=2.0, label="Gameplay policy MA(10)")
    plt.title("Gameplay Policy Loss")
    plt.xlabel("Training update")
    plt.ylabel("Policy loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["trade_policy_loss"], alpha=0.35, label="Per-update trade policy loss")
    plt.plot(df["update"], df["trade_policy_loss_ma10"], linewidth=2.0, label="Trade policy MA(10)")
    plt.title("Trade Policy Loss")
    plt.xlabel("Training update")
    plt.ylabel("Policy loss")
    plt.legend()

    _save("unified_policy_losses.png", output_dir)


def plot_trade_actions(df: pd.DataFrame, output_dir: str) -> None:
    if "trade_propose_rate" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["update"], df["trade_propose_rate"], label="Propose rate")
    plt.plot(df["update"], df["trade_accept_rate"], label="Accept rate")
    plt.plot(df["update"], df["trade_reject_rate"], label="Reject rate")
    plt.plot(df["update"], df["trade_counter_rate"], label="Counter rate")
    plt.plot(df["update"], df["trade_skip_rate"], label="Skip rate")
    plt.title("Trade Action Distribution")
    plt.xlabel("Training update")
    plt.ylabel("Fraction of trade decisions")
    plt.legend()
    _save("unified_trade_action_distribution.png", output_dir)


def plot_tom_and_total(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["tom_loss"], alpha=0.35, label="Per-update ToM loss")
    plt.plot(df["update"], df["tom_loss_ma10"], linewidth=2.0, label="ToM loss MA(10)")
    plt.title("ToM Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["total_loss"], alpha=0.35, label="Per-update total loss")
    plt.plot(df["update"], df["total_loss_ma10"], linewidth=2.0, label="Total loss MA(10)")
    plt.title("Total PPO Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    _save("unified_tom_and_total_loss.png", output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse unified PPO logs and generate diagnostics.")
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="unified_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = parse_training_log(args.log_file)
    df = add_derived_metrics(df)

    csv_path = os.path.join(args.output_dir, "parsed_unified_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    save_summary(df, args.output_dir)

    plot_reward_overview(df, args.output_dir)
    plot_entropy_overview(df, args.output_dir)
    plot_value_losses(df, args.output_dir)
    plot_policy_losses(df, args.output_dir)
    plot_trade_actions(df, args.output_dir)
    plot_tom_and_total(df, args.output_dir)

    print(f"Saved figures and CSV to: {args.output_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()