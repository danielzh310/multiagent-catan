from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def smoothstep(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def logistic(x: float, center: float, width: float) -> float:
    width = max(width, 1e-6)
    return 1.0 / (1.0 + math.exp(-(x - center) / width))


def pseudo_noise(update: int, scale: float = 1.0) -> float:
    return scale * (
        0.55 * math.sin(update / 37.0)
        + 0.25 * math.cos(update / 89.0)
        + 0.12 * math.sin(update / 173.0 + 0.7)
        + 0.08 * math.cos(update / 421.0 + 1.1)
    )


def training_bump(update: int, center: float, width: float, height: float) -> float:
    z = (update - center) / max(width, 1.0)
    return height * math.exp(-(z * z))


def saw(update: int, period: float, amplitude: float) -> float:
    phase = (update % period) / period
    return amplitude * (2.0 * phase - 1.0)


def long_drift(update: int, total_updates: int, points: list[tuple[float, float]]) -> float:
    if not points:
        return 0.0
    pos = update / max(total_updates - 1, 1)
    pts = sorted(points)
    if pos <= pts[0][0]:
        return pts[0][1]
    if pos >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        if x0 <= pos <= x1:
            t = (pos - x0) / max(x1 - x0, 1e-9)
            return y0 + t * (y1 - y0)
    return pts[-1][1]


def build_mock_projection(df: pd.DataFrame, total_updates: int) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Input metrics CSV is empty.")

    real = df.copy().sort_values("update").reset_index(drop=True).ffill()
    last = real.iloc[-1]
    real_updates = len(real)

    rows = []
    for _, row in real.iterrows():
        rows.append(row.to_dict())

    for update in range(real_updates, total_updates):
        prev = rows[-1]
        x = update / max(total_updates - 1, 1)
        progress = smoothstep((update - real_updates + 1) / max(total_updates - real_updates, 1))
        n1 = pseudo_noise(update, 1.0)
        n2 = pseudo_noise(update + 211, 1.0)
        n3 = pseudo_noise(update + 977, 1.0)
        short_jitter = 0.7 * math.sin(update / 5.0) + 0.5 * math.cos(update / 9.0) + saw(update, 17.0, 0.9)
        medium_jitter = 0.6 * math.sin(update / 23.0 + 0.3) + 0.4 * math.cos(update / 41.0) + saw(update, 61.0, 0.6)
        long_jitter = 0.9 * math.sin(update / 290.0 + 0.2) + 0.7 * math.cos(update / 510.0 + 1.1) + saw(update, 211.0, 0.45)
        spike_train = (
            training_bump(update, total_updates * 0.14, total_updates * 0.008, 1.0)
            + training_bump(update, total_updates * 0.31, total_updates * 0.010, 1.0)
            + training_bump(update, total_updates * 0.47, total_updates * 0.012, 1.0)
            + training_bump(update, total_updates * 0.58, total_updates * 0.009, 1.0)
            + training_bump(update, total_updates * 0.73, total_updates * 0.011, 1.0)
            + training_bump(update, total_updates * 0.86, total_updates * 0.008, 1.0)
            + training_bump(update, total_updates * 0.94, total_updates * 0.006, 1.0)
        )

        explore_to_stable = logistic(x, 0.18, 0.05)
        mid_training = logistic(x, 0.45, 0.06)
        late_training = logistic(x, 0.72, 0.07)

        regression_band = (
            training_bump(update, total_updates * 0.22, total_updates * 0.035, 1.0)
            + training_bump(update, total_updates * 0.53, total_updates * 0.045, 1.0)
            + training_bump(update, total_updates * 0.81, total_updates * 0.03, 1.0)
        )
        gameplay_entropy_drift = long_drift(
            update,
            total_updates,
            [(0.0, 0.06), (0.18, -0.02), (0.37, 0.04), (0.56, -0.03), (0.78, 0.05), (1.0, -0.01)],
        )
        trade_entropy_drift = long_drift(
            update,
            total_updates,
            [(0.0, 0.10), (0.20, -0.06), (0.42, 0.08), (0.61, -0.10), (0.83, 0.07), (1.0, -0.02)],
        )
        gameplay_value_drift = long_drift(
            update,
            total_updates,
            [(0.0, 0.14), (0.15, -0.04), (0.33, 0.12), (0.49, -0.02), (0.68, 0.10), (0.86, -0.06), (1.0, 0.03)],
        )
        trade_value_drift = long_drift(
            update,
            total_updates,
            [(0.0, 0.18), (0.19, -0.03), (0.36, 0.16), (0.57, -0.01), (0.73, 0.14), (0.89, -0.08), (1.0, 0.05)],
        )
        tom_drift = long_drift(
            update,
            total_updates,
            [(0.0, 0.28), (0.14, -0.10), (0.26, 0.22), (0.41, -0.16), (0.55, 0.19), (0.69, -0.13), (0.82, 0.24), (0.93, -0.18), (1.0, 0.06)],
        )

        gameplay_rollouts = int(round(
            1980
            + 140 * explore_to_stable
            + 65 * mid_training
            + 35 * late_training
            + 55 * n1
            + 18 * short_jitter
            - 40 * regression_band
        ))
        trade_rollouts = int(round(
            1380
            + 60 * explore_to_stable
            + 120 * mid_training
            + 90 * late_training
            + 50 * n2
            + 22 * medium_jitter
            + 35 * regression_band
        ))

        gameplay_reward_trend = (
            float(last["gameplay_reward"])
            + 0.004 * explore_to_stable
            + 0.008 * mid_training
            + 0.007 * late_training
        )
        gameplay_reward = clamp(
            0.82 * float(prev["gameplay_reward"])
            + 0.18 * (
                gameplay_reward_trend
                + 0.0018 * n1
                + 0.0009 * short_jitter
                - 0.0025 * regression_band
                + training_bump(update, total_updates * 0.63, total_updates * 0.05, 0.0035)
            ),
            0.008,
            0.038,
        )

        trade_reward_trend = (
            float(last["trade_reward"])
            + 0.001 * explore_to_stable
            + 0.0045 * mid_training
            + 0.0055 * late_training
        )
        trade_reward = clamp(
            0.80 * float(prev["trade_reward"])
            + 0.20 * (
                trade_reward_trend
                + 0.0012 * n2
                + 0.0008 * medium_jitter
                - 0.0022 * regression_band
                + training_bump(update, total_updates * 0.67, total_updates * 0.07, 0.0025)
            ),
            -0.002,
            0.015,
        )

        gameplay_entropy_target = (
            float(last["gameplay_entropy"])
            - 0.05 * explore_to_stable
            - 0.06 * mid_training
            - 0.04 * late_training
            + 0.03 * regression_band
            + 0.02 * n3
            + gameplay_entropy_drift
        )
        gameplay_entropy = clamp(
            0.88 * float(prev["gameplay_entropy"])
            + 0.12 * gameplay_entropy_target
            + 0.018 * short_jitter,
            0.50,
            1.05,
        )
        trade_entropy_target = (
            float(last["trade_entropy"])
            - 0.18 * explore_to_stable
            - 0.22 * mid_training
            - 0.20 * late_training
            + 0.10 * regression_band
            + 0.05 * n1
            + 0.06 * medium_jitter
            + trade_entropy_drift
        )
        trade_entropy = clamp(
            0.78 * float(prev["trade_entropy"])
            + 0.22 * trade_entropy_target
            + 0.070 * short_jitter
            + 0.060 * medium_jitter
            + 0.120 * spike_train
            + 0.090 * long_jitter,
            0.92,
            2.45,
        )

        gameplay_value_target = (
            float(last["gameplay_value_loss"])
            - 0.06 * explore_to_stable
            - 0.09 * mid_training
            - 0.07 * late_training
            + 0.07 * regression_band
            + 0.03 * n2
            + 0.06 * short_jitter
            + training_bump(update, total_updates * 0.31, total_updates * 0.012, 0.07)
            + training_bump(update, total_updates * 0.58, total_updates * 0.015, 0.09)
            + training_bump(update, total_updates * 0.88, total_updates * 0.01, 0.06)
            + gameplay_value_drift
        )
        gameplay_value_loss = clamp(
            0.58 * float(prev["gameplay_value_loss"])
            + 0.42 * gameplay_value_target
            + 0.13 * short_jitter
            + 0.09 * medium_jitter
            + 0.16 * spike_train
            + 0.12 * long_jitter,
            0.22,
            1.10,
        )
        trade_value_target = (
            float(last["trade_value_loss"])
            - 0.04 * explore_to_stable
            - 0.07 * mid_training
            - 0.06 * late_training
            + 0.09 * regression_band
            + 0.025 * n3
            + 0.07 * medium_jitter
            + training_bump(update, total_updates * 0.27, total_updates * 0.018, 0.08)
            + training_bump(update, total_updates * 0.52, total_updates * 0.02, 0.11)
            + training_bump(update, total_updates * 0.79, total_updates * 0.016, 0.10)
            + trade_value_drift
        )
        trade_value_loss = clamp(
            0.52 * float(prev["trade_value_loss"])
            + 0.48 * trade_value_target
            + 0.18 * short_jitter
            + 0.14 * medium_jitter
            + 0.22 * spike_train
            + 0.16 * long_jitter,
            0.16,
            1.30,
        )

        gameplay_policy_target = (
            float(last["gameplay_policy_loss"])
            - 0.03 * explore_to_stable
            - 0.025 * mid_training
            + 0.035 * regression_band
            + 0.025 * n1
        )
        gameplay_policy_loss = clamp(
            0.75 * float(prev["gameplay_policy_loss"])
            + 0.25 * gameplay_policy_target
            + 0.05 * short_jitter
            + 0.05 * spike_train,
            -0.18,
            0.18,
        )
        trade_policy_target = (
            float(last["trade_policy_loss"])
            - 0.05 * explore_to_stable
            - 0.035 * mid_training
            + 0.05 * regression_band
            + 0.035 * n2
        )
        trade_policy_loss = clamp(
            0.68 * float(prev["trade_policy_loss"])
            + 0.32 * trade_policy_target
            + 0.08 * short_jitter
            + 0.06 * medium_jitter
            + 0.18 * spike_train,
            -0.22,
            0.32,
        )
        tom_baseline = (
            1.52
            - 0.07 * explore_to_stable
            - 0.10 * mid_training
            - 0.06 * late_training
            + 0.10 * regression_band
            + 0.10 * tom_drift
            + 0.06 * math.sin(update / 85.0)
            + 0.05 * math.cos(update / 170.0 + 0.4)
        )
        tom_shock = (
            training_bump(update, total_updates * 0.14, total_updates * 0.020, 0.16)
            - training_bump(update, total_updates * 0.22, total_updates * 0.015, 0.12)
            + training_bump(update, total_updates * 0.37, total_updates * 0.022, 0.20)
            - training_bump(update, total_updates * 0.50, total_updates * 0.018, 0.14)
            + training_bump(update, total_updates * 0.68, total_updates * 0.025, 0.22)
            - training_bump(update, total_updates * 0.82, total_updates * 0.018, 0.16)
            + training_bump(update, total_updates * 0.92, total_updates * 0.015, 0.12)
        )
        tom_loss = clamp(
            0.82 * float(prev["tom_loss"])
            + 0.18 * tom_baseline
            + 0.025 * medium_jitter
            + 0.035 * long_jitter
            + tom_shock,
            0.90,
            2.20,
        )

        entropy_coef = (
            0.0012
            if x < 0.70
            else 0.0012 - 0.00025 * smoothstep((x - 0.70) / 0.30)
        )
        gamma = clamp(
            float(last["gamma"])
            + 0.008 * explore_to_stable
            + 0.006 * mid_training
            + 0.004 * late_training,
            float(last["gamma"]),
            0.979,
        )

        propose_rate = clamp(0.48 + 0.05 * explore_to_stable + 0.05 * mid_training + 0.02 * n1, 0.38, 0.62)
        accept_rate = clamp(0.003 + 0.03 * explore_to_stable + 0.05 * mid_training + 0.05 * late_training + 0.008 * n2, 0.002, 0.16)
        reject_rate = clamp(0.48 - 0.10 * explore_to_stable - 0.10 * mid_training + 0.02 * regression_band + 0.015 * n3, 0.18, 0.52)
        counter_rate = clamp(0.035 + 0.01 * explore_to_stable - 0.015 * mid_training - 0.008 * late_training + 0.01 * regression_band + 0.008 * n1, 0.01, 0.10)
        skip_rate = clamp(1.0 - propose_rate - accept_rate - reject_rate - counter_rate, 0.015, 0.09)

        total_rate = propose_rate + accept_rate + reject_rate + counter_rate + skip_rate
        propose_rate /= total_rate
        accept_rate /= total_rate
        reject_rate /= total_rate
        counter_rate /= total_rate
        skip_rate /= total_rate

        total_actions = max(trade_rollouts, 1)
        trade_propose = int(round(total_actions * propose_rate))
        trade_accept = int(round(total_actions * accept_rate))
        trade_reject = int(round(total_actions * reject_rate))
        trade_counter = int(round(total_actions * counter_rate))
        trade_skip = max(total_actions - trade_propose - trade_accept - trade_reject - trade_counter, 0)

        total_loss = clamp(
            (
                0.06
                + 0.28 * gameplay_value_loss
                + 0.20 * trade_value_loss
                + 0.06 * tom_loss
                + 0.10 * regression_band
                - 0.18 * gameplay_reward
                - 0.25 * trade_reward
                - 0.03 * accept_rate
                + 0.02 * n2
                + 0.03 * short_jitter
                - 0.04 * progress
            ),
            -0.10,
            0.35,
        )

        rows.append(
            {
                "update": update,
                "gameplay_rollouts": gameplay_rollouts,
                "trade_rollouts": trade_rollouts,
                "gameplay_reward": gameplay_reward,
                "trade_reward": trade_reward,
                "trade_propose": trade_propose,
                "trade_accept": trade_accept,
                "trade_reject": trade_reject,
                "trade_counter": trade_counter,
                "trade_skip": trade_skip,
                "gameplay_policy_loss": gameplay_policy_loss,
                "gameplay_value_loss": gameplay_value_loss,
                "gameplay_entropy": gameplay_entropy,
                "trade_policy_loss": trade_policy_loss,
                "trade_value_loss": trade_value_loss,
                "trade_entropy": trade_entropy,
                "tom_loss": tom_loss,
                "total_loss": total_loss,
                "entropy_coef": entropy_coef,
                "gamma": gamma,
            }
        )

    out = pd.DataFrame(rows).sort_values("update").reset_index(drop=True)
    out["gameplay_reward_avg_logged"] = out["gameplay_reward"].rolling(window=10, min_periods=1).mean()
    out["trade_reward_avg_logged"] = out["trade_reward"].rolling(window=10, min_periods=1).mean()
    out["gameplay_entropy_avg_logged"] = out["gameplay_entropy"].rolling(window=10, min_periods=1).mean()
    out["trade_entropy_avg_logged"] = out["trade_entropy"].rolling(window=10, min_periods=1).mean()
    return out


def write_mock_log(df: pd.DataFrame, output_path: Path) -> None:
    lines = []
    lines.append(
        f"MOCK unified training projection generated from partial real run for {len(df)} updates"
    )

    for _, row in df.iterrows():
        lines.append("")
        lines.append(f"Update {int(row['update'])}")
        lines.append(
            f"rollouts | gameplay={int(row['gameplay_rollouts'])} trade={int(row['trade_rollouts'])}"
        )
        lines.append(
            f"reward   | gameplay={row['gameplay_reward']:.4f} (avg={row['gameplay_reward_avg_logged']:.4f}) "
            f"trade={row['trade_reward']:.4f} (avg={row['trade_reward_avg_logged']:.4f})"
        )
        lines.append(
            f"trade actions | propose={int(row['trade_propose'])} accept={int(row['trade_accept'])} "
            f"reject={int(row['trade_reject'])} counter={int(row['trade_counter'])} skip={int(row['trade_skip'])}"
        )
        total_trade = max(
            int(row["trade_propose"]) + int(row["trade_accept"]) + int(row["trade_reject"]) + int(row["trade_counter"]) + int(row["trade_skip"]),
            1,
        )
        lines.append(
            f"trade rates   | propose={int(row['trade_propose'])/total_trade:.3f} "
            f"accept={int(row['trade_accept'])/total_trade:.3f} "
            f"reject={int(row['trade_reject'])/total_trade:.3f} "
            f"counter={int(row['trade_counter'])/total_trade:.3f} "
            f"skip={int(row['trade_skip'])/total_trade:.3f}"
        )
        lines.append(
            f"ppo gameplay | policy={row['gameplay_policy_loss']:.4f} value={row['gameplay_value_loss']:.4f} "
            f"entropy={row['gameplay_entropy']:.4f} (avg={row['gameplay_entropy_avg_logged']:.4f})"
        )
        lines.append(
            f"ppo trade    | policy={row['trade_policy_loss']:.4f} value={row['trade_value_loss']:.4f} "
            f"entropy={row['trade_entropy']:.4f} (avg={row['trade_entropy_avg_logged']:.4f}) tom={row['tom_loss']:.6f}"
        )
        lines.append(
            f"ppo total    | loss={row['total_loss']:.4f} entropy_coef={row['entropy_coef']:.6f} "
            f"gamma={row['gamma']:.5f} tom_coef={clamp(0.02 + 0.04 * (int(row['update']) / max(len(df) - 1, 1)), 0.02, 0.06):.5f} "
            f"trade_score={row['trade_reward_avg_logged'] * 4.0:.4f}"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mock projected unified training log.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-log", type=str, required=True)
    parser.add_argument("--total-updates", type=int, default=500)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    projected = build_mock_projection(df, args.total_updates)
    write_mock_log(projected, Path(args.output_log))
    print(f"Wrote mock projection log to {args.output_log}")


if __name__ == "__main__":
    main()
