from __future__ import annotations

import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


UPDATE_RE = re.compile(r".*Update\s+(\d+).*")
ROLLOUTS_RE = re.compile(r"rollouts\s*\|\s*gameplay=(\d+)\s+trade=(\d+)\s+epsilon=([-+]?\d*\.?\d+)\s+replay=(\d+)")
REWARD_RE = re.compile(
    r"reward\s*\|\s*gameplay=([-+]?\d*\.?\d+)\s+\(avg=([-+]?\d*\.?\d+)\)\s+trade=([-+]?\d*\.?\d+)\s+\(avg=([-+]?\d*\.?\d+)\)"
)
TRADE_ACTIONS_RE = re.compile(
    r"trade actions\s*\|\s*propose=(\d+)\s+accept=(\d+)\s+reject=(\d+)\s+counter=(\d+)\s+skip=(\d+)"
)
DQN_GAMEPLAY_RE = re.compile(
    r"dqn gameplay\s*\|\s*td=([-+]?\d*\.?\d+)\s+q=([-+]?\d*\.?\d+)\s+target_q=([-+]?\d*\.?\d+)"
)
PPO_TRADE_RE = re.compile(
    r"ppo trade\s*\|\s*policy=([-+]?\d*\.?\d+)\s+value=([-+]?\d*\.?\d+)\s+entropy=([-+]?\d*\.?\d+)\s+tom=([-+]?\d*\.?\d+)"
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
            raise UnicodeDecodeError("unknown", b"", 0, 1, f"Could not decode file. Last error: {last_error}")

    return text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def parse_training_log(log_path: str) -> pd.DataFrame:
    lines = read_text_with_fallbacks(log_path).splitlines()

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
            current["epsilon"] = float(m.group(3))
            current["replay_size"] = int(m.group(4))
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

        m = DQN_GAMEPLAY_RE.search(line)
        if m:
            current["dqn_td_loss"] = float(m.group(1))
            current["dqn_q_mean"] = float(m.group(2))
            current["dqn_target_q_mean"] = float(m.group(3))
            continue

        m = PPO_TRADE_RE.search(line)
        if m:
            current["trade_policy_loss"] = float(m.group(1))
            current["trade_value_loss"] = float(m.group(2))
            current["trade_entropy"] = float(m.group(3))
            current["tom_loss"] = float(m.group(4))
            continue

    if current:
        rows.append(current)

    if not rows:
        raise ValueError("No updates were parsed from the log file.")

    df = pd.DataFrame(rows).sort_values("update").reset_index(drop=True)
    for col in [c for c in df.columns if c != "update"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "gameplay_reward",
        "trade_reward",
        "dqn_td_loss",
        "dqn_q_mean",
        "dqn_target_q_mean",
        "trade_policy_loss",
        "trade_value_loss",
        "trade_entropy",
        "tom_loss",
        "epsilon",
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
    lines = [
        f"Parsed updates: {len(df)}",
        f"Update range: {int(df['update'].min())} to {int(df['update'].max())}",
    ]

    for col in [
        "gameplay_reward",
        "trade_reward",
        "dqn_td_loss",
        "trade_policy_loss",
        "trade_value_loss",
        "trade_entropy",
        "tom_loss",
        "epsilon",
    ]:
        if col in df.columns and df[col].notna().any():
            lines.append(f"{col} mean: {df[col].mean():.6f}")
            lines.append(f"{col} min: {df[col].min():.6f}")
            lines.append(f"{col} max: {df[col].max():.6f}")

    with open(os.path.join(output_dir, "hybrid_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_stability_table(df: pd.DataFrame, output_dir: str) -> None:
    trade_policy_thresholds = [1e3, 1e4, 1e5, 1e6]
    rows = []

    for threshold in trade_policy_thresholds:
        rows.append(
            {
                "metric": f"trade_policy_loss_gt_{threshold:.0e}",
                "value": int((df.get("trade_policy_loss", pd.Series(dtype=float)) > threshold).sum()),
            }
        )

    if "trade_reward" in df.columns:
        rows.append(
            {
                "metric": "negative_trade_reward_updates",
                "value": int((df["trade_reward"] < 0).sum()),
            }
        )

    if {"trade_counter_rate", "trade_accept_rate"}.issubset(df.columns):
        rows.append(
            {
                "metric": "high_counter_low_accept_updates",
                "value": int(((df["trade_counter_rate"] > 0.5) & (df["trade_accept_rate"] < 0.1)).sum()),
            }
        )

    if "trade_rollouts" in df.columns:
        rows.append(
            {
                "metric": "high_trade_rollout_updates",
                "value": int((df["trade_rollouts"] > df["trade_rollouts"].median()).sum()),
            }
        )

    for col in ["trade_policy_loss", "trade_value_loss", "trade_entropy", "tom_loss", "dqn_td_loss"]:
        if col in df.columns and df[col].notna().any():
            rows.append({"metric": f"{col}_max", "value": float(df[col].max())})
            rows.append({"metric": f"{col}_mean", "value": float(df[col].mean())})

    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "hybrid_stability_table.csv"),
        index=False,
        encoding="utf-8",
    )


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

    _save("hybrid_reward_overview.png", output_dir)


def plot_dqn_metrics(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(df["update"], df["dqn_td_loss"], alpha=0.35, label="Per-update TD loss")
    plt.plot(df["update"], df["dqn_td_loss_ma10"], linewidth=2.0, label="TD loss MA(10)")
    plt.title("DQN TD Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df["update"], df["dqn_q_mean"], alpha=0.35, label="Per-update Q mean")
    plt.plot(df["update"], df["dqn_q_mean_ma10"], linewidth=2.0, label="Q mean MA(10)")
    plt.title("DQN Q Mean")
    plt.xlabel("Training update")
    plt.ylabel("Q")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df["update"], df["dqn_target_q_mean"], alpha=0.35, label="Per-update target Q mean")
    plt.plot(df["update"], df["dqn_target_q_mean_ma10"], linewidth=2.0, label="Target Q mean MA(10)")
    plt.title("DQN Target Q Mean")
    plt.xlabel("Training update")
    plt.ylabel("Target Q")
    plt.legend()

    _save("hybrid_dqn_metrics.png", output_dir)


def plot_trade_metrics(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 10))

    plt.subplot(3, 1, 1)
    plt.plot(df["update"], df["trade_policy_loss"], alpha=0.35, label="Per-update trade policy loss")
    plt.plot(df["update"], df["trade_policy_loss_ma10"], linewidth=2.0, label="Trade policy MA(10)")
    plt.title("Trade Policy Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df["update"], df["trade_value_loss"], alpha=0.35, label="Per-update trade value loss")
    plt.plot(df["update"], df["trade_value_loss_ma10"], linewidth=2.0, label="Trade value MA(10)")
    plt.title("Trade Value Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df["update"], df["trade_entropy"], alpha=0.35, label="Per-update trade entropy")
    plt.plot(df["update"], df["trade_entropy_ma10"], linewidth=2.0, label="Trade entropy MA(10)")
    plt.title("Trade Entropy")
    plt.xlabel("Training update")
    plt.ylabel("Entropy")
    plt.legend()

    _save("hybrid_trade_metrics.png", output_dir)


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
    _save("hybrid_trade_action_distribution.png", output_dir)


def plot_schedule_and_tom(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(df["update"], df["epsilon"], alpha=0.35, label="Per-update epsilon")
    plt.plot(df["update"], df["epsilon_ma10"], linewidth=2.0, label="Epsilon MA(10)")
    plt.title("DQN Epsilon")
    plt.xlabel("Training update")
    plt.ylabel("Epsilon")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(df["update"], df["tom_loss"], alpha=0.35, label="Per-update ToM loss")
    plt.plot(df["update"], df["tom_loss_ma10"], linewidth=2.0, label="ToM loss MA(10)")
    plt.title("ToM Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    _save("hybrid_schedule_and_tom.png", output_dir)


def _phase_slices(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    n = len(df)
    if n < 3:
        return {"All": df}

    one_third = max(1, n // 3)
    two_third = max(one_third + 1, (2 * n) // 3)
    return {
        "Early": df.iloc[:one_third],
        "Mid": df.iloc[one_third:two_third],
        "Late": df.iloc[two_third:],
    }


def _boxplot_metric_by_phase(df: pd.DataFrame, metric: str, title: str, ylabel: str) -> bool:
    phases = _phase_slices(df)
    labels = []
    series = []
    for label, part in phases.items():
        if metric in part.columns:
            values = part[metric].dropna()
            if not values.empty:
                labels.append(label)
                series.append(values)

    if not series:
        return False

    plt.boxplot(series, tick_labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Training Phase")
    return True


def plot_box_whisker_overview(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(10, 8))

    plotted = 0

    plt.subplot(2, 2, 1)
    plotted += int(_boxplot_metric_by_phase(df, "gameplay_reward", "Gameplay Reward Distribution", "Reward"))

    plt.subplot(2, 2, 2)
    plotted += int(_boxplot_metric_by_phase(df, "trade_reward", "Trade Reward Distribution", "Reward"))

    plt.subplot(2, 2, 3)
    plotted += int(_boxplot_metric_by_phase(df, "dqn_td_loss", "DQN TD Loss Distribution", "Loss"))

    plt.subplot(2, 2, 4)
    plotted += int(_boxplot_metric_by_phase(df, "trade_entropy", "Trade Entropy Distribution", "Entropy"))

    if plotted == 0:
        plt.close()
        return

    _save("hybrid_box_whisker_overview.png", output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Parse hybrid_v2 logs and generate diagnostics.")
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="figures/hybrid_figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = parse_training_log(args.log_file)
    df = add_derived_metrics(df)
    df.to_csv(os.path.join(args.output_dir, "parsed_hybrid_metrics.csv"), index=False)

    save_summary(df, args.output_dir)
    save_stability_table(df, args.output_dir)
    plot_reward_overview(df, args.output_dir)
    plot_dqn_metrics(df, args.output_dir)
    plot_trade_metrics(df, args.output_dir)
    plot_trade_actions(df, args.output_dir)
    plot_schedule_and_tom(df, args.output_dir)
    plot_box_whisker_overview(df, args.output_dir)

    print(f"Saved hybrid diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
