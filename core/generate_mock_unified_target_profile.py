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


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def phase_interp(progress: float, early: float, mid: float, late: float) -> float:
    if progress < 0.45:
        return lerp(early, mid, smoothstep(progress / 0.45))
    return lerp(mid, late, smoothstep((progress - 0.45) / 0.55))


def drift_wave(update: int, a: float, b: float, c: float = 0.0) -> float:
    return (
        a * math.sin(update / 190.0 + 0.3)
        + b * math.cos(update / 430.0 + 1.0)
        + c * math.sin(update / 910.0 + 0.7)
    )


def bump(update: int, center: float, width: float, height: float) -> float:
    z = (update - center) / max(width, 1.0)
    return height * math.exp(-(z * z))


def spike_mix(update: int, total_updates: int, centers: list[tuple[float, float, float]]) -> float:
    total = 0.0
    for center_frac, width_frac, height in centers:
        total += bump(update, total_updates * center_frac, total_updates * width_frac, height)
    return total


def build_projection(df: pd.DataFrame, total_updates: int) -> pd.DataFrame:
    real = df.copy().sort_values("update").reset_index(drop=True).ffill()
    last = real.iloc[-1]
    real_updates = len(real)

    rows = [row.to_dict() for _, row in real.iterrows()]

    gp_reward = float(last["gameplay_reward"])
    tr_reward = float(last["trade_reward"])
    gp_entropy = float(last["gameplay_entropy"])
    tr_entropy = float(last["trade_entropy"])
    gp_value = float(last["gameplay_value_loss"])
    tr_value = float(last["trade_value_loss"])
    gp_policy = float(last["gameplay_policy_loss"])
    tr_policy = float(last["trade_policy_loss"])
    tom_loss = float(last["tom_loss"])

    for update in range(real_updates, total_updates):
        progress = (update - real_updates) / max(total_updates - real_updates - 1, 1)
        p = smoothstep(progress)

        gp_reward_target = phase_interp(progress, 0.014, 0.023, 0.032)
        tr_reward_target = phase_interp(progress, -0.003, 0.0045, 0.009)
        gp_entropy_target = phase_interp(progress, 0.78, 0.66, 0.56)
        tr_entropy_target = phase_interp(progress, 2.15, 1.75, 1.38)
        gp_value_target = phase_interp(progress, 0.78, 0.50, 0.34)
        tr_value_target = phase_interp(progress, 0.58, 0.42, 0.29)
        tom_target = phase_interp(progress, 1.58, 1.34, 1.10)

        gp_reward_noise = drift_wave(update, 0.0020, 0.0012, 0.0008)
        tr_reward_noise = drift_wave(update + 71, 0.0012, 0.0009, 0.0006)
        gp_entropy_noise = drift_wave(update + 11, 0.035, 0.020, 0.015)
        tr_entropy_noise = drift_wave(update + 103, 0.080, 0.050, 0.035)
        gp_value_noise = drift_wave(update + 17, 0.050, 0.030, 0.020)
        tr_value_noise = drift_wave(update + 149, 0.045, 0.028, 0.018)
        tom_noise = drift_wave(update + 211, 0.085, 0.050, 0.035)

        instability = (
            bump(update, total_updates * 0.16, total_updates * 0.020, 1.0)
            + bump(update, total_updates * 0.42, total_updates * 0.025, 1.0)
            + bump(update, total_updates * 0.73, total_updates * 0.022, 1.0)
        )
        gp_value_spikes = spike_mix(
            update,
            total_updates,
            [(0.12, 0.010, 0.10), (0.29, 0.014, 0.12), (0.51, 0.012, 0.09), (0.77, 0.016, 0.11)],
        )
        tr_value_spikes = spike_mix(
            update,
            total_updates,
            [(0.18, 0.012, 0.11), (0.36, 0.015, 0.13), (0.62, 0.014, 0.12), (0.84, 0.013, 0.10)],
        )
        tom_spikes = spike_mix(
            update,
            total_updates,
            [(0.15, 0.014, 0.10), (0.41, 0.018, 0.14), (0.69, 0.016, 0.12), (0.90, 0.010, 0.08)],
        )
        micro_noise = (
            0.8 * math.sin(update / 17.0)
            + 0.6 * math.cos(update / 29.0 + 0.5)
            + 0.4 * math.sin(update / 43.0 + 1.1)
        )
        burst_noise = (
            1.2 * math.sin(update / 9.0)
            + 0.9 * math.cos(update / 15.0 + 0.8)
            + 0.6 * math.sin(update / 23.0 + 1.7)
        )
        burst_windows = spike_mix(
            update,
            total_updates,
            [(0.10, 0.008, 1.0), (0.21, 0.010, 1.0), (0.34, 0.009, 1.0), (0.48, 0.012, 1.0), (0.59, 0.010, 1.0), (0.71, 0.011, 1.0), (0.83, 0.009, 1.0), (0.93, 0.007, 1.0)],
        )
        reward_bursts = spike_mix(
            update,
            total_updates,
            [(0.09, 0.012, 1.0), (0.27, 0.014, 1.0), (0.46, 0.013, 1.0), (0.66, 0.016, 1.0), (0.88, 0.012, 1.0)],
        )
        policy_bursts = spike_mix(
            update,
            total_updates,
            [(0.07, 0.010, 1.0), (0.19, 0.011, 1.0), (0.33, 0.012, 1.0), (0.44, 0.010, 1.0), (0.58, 0.013, 1.0), (0.72, 0.011, 1.0), (0.86, 0.012, 1.0), (0.95, 0.008, 1.0)],
        )
        policy_regimes = (
            bump(update, total_updates * 0.14, total_updates * 0.020, 1.0)
            - bump(update, total_updates * 0.24, total_updates * 0.018, 0.8)
            + bump(update, total_updates * 0.39, total_updates * 0.022, 1.2)
            - bump(update, total_updates * 0.52, total_updates * 0.018, 0.9)
            + bump(update, total_updates * 0.68, total_updates * 0.024, 1.3)
            - bump(update, total_updates * 0.81, total_updates * 0.020, 1.0)
            + bump(update, total_updates * 0.93, total_updates * 0.014, 0.8)
        )
        action_regime = (
            0.9 * math.sin(update / 140.0 + 0.2)
            + 0.7 * math.cos(update / 260.0 + 0.9)
            + 0.5 * math.sin(update / 420.0 + 1.3)
        )
        action_bursts = spike_mix(
            update,
            total_updates,
            [(0.08, 0.010, 1.0), (0.17, 0.012, 1.0), (0.31, 0.011, 1.0), (0.45, 0.014, 1.0), (0.60, 0.013, 1.0), (0.74, 0.012, 1.0), (0.88, 0.010, 1.0)],
        )
        action_micro = (
            1.1 * math.sin(update / 21.0 + 0.4)
            + 0.9 * math.cos(update / 35.0 + 1.0)
            + 0.6 * math.sin(update / 57.0 + 1.7)
        )

        late_var_scale = lerp(1.0, 0.55, p)

        gp_reward = clamp(
            0.72 * gp_reward
            + 0.18 * (
                gp_reward_target
                + gp_reward_noise
                + 0.0022 * micro_noise
                + 0.0038 * burst_noise * (0.35 + 0.65 * reward_bursts)
                - 0.0035 * instability
            ),
            0.006,
            0.040,
        )
        tr_reward = clamp(
            0.68 * tr_reward
            + 0.20 * (
                tr_reward_target
                + tr_reward_noise
                + 0.0018 * micro_noise
                + 0.0030 * burst_noise * (0.30 + 0.70 * reward_bursts)
                - 0.0028 * instability
            ),
            -0.004,
            0.014,
        )

        gp_entropy = clamp(
            0.82 * gp_entropy + 0.18 * (
                gp_entropy_target
                + gp_entropy_noise * late_var_scale
                + 0.03 * instability
                + 0.035 * micro_noise
                + 0.030 * burst_noise * (0.35 + 0.65 * burst_windows)
            ),
            0.42,
            1.05,
        )
        tr_entropy = clamp(
            0.78 * tr_entropy + 0.22 * (
                tr_entropy_target
                + tr_entropy_noise * late_var_scale
                + 0.08 * instability
                + 0.05 * tom_spikes
                + 0.050 * micro_noise
                + 0.055 * burst_noise * (0.40 + 0.60 * burst_windows)
            ),
            1.05,
            2.55,
        )

        gp_value = clamp(
            0.68 * gp_value
            + 0.32 * (
                gp_value_target
                + gp_value_noise * late_var_scale
                + 0.14 * instability
                + gp_value_spikes
                + 0.055 * micro_noise
                + 0.070 * burst_noise * (0.35 + 0.65 * burst_windows)
            ),
            0.18,
            1.10,
        )
        tr_value = clamp(
            0.66 * tr_value
            + 0.34 * (
                tr_value_target
                + tr_value_noise * late_var_scale
                + 0.12 * instability
                + tr_value_spikes
                + 0.060 * micro_noise
                + 0.085 * burst_noise * (0.40 + 0.60 * burst_windows)
            ),
            0.14,
            0.85,
        )
        tom_loss = clamp(
            0.70 * tom_loss
            + 0.30 * (
                tom_target
                + tom_noise * late_var_scale
                + 0.14 * instability
                + tom_spikes
                + 0.070 * micro_noise
                + 0.095 * burst_noise * (0.45 + 0.55 * burst_windows)
            ),
            0.82,
            1.95,
        )

        gp_policy = clamp(
            0.40 * gp_policy
            + 0.60 * (
                drift_wave(update + 301, 0.06, 0.04, 0.02)
                + 0.045 * micro_noise
                + 0.070 * burst_noise * (0.35 + 0.65 * policy_bursts)
                + 0.050 * instability
                + 0.10 * policy_regimes
                + 0.05 * math.sin(update / 61.0)
            ),
            -0.36,
            0.36,
        )
        tr_policy = clamp(
            0.34 * tr_policy
            + 0.66 * (
                drift_wave(update + 401, 0.07, 0.05, 0.03)
                + 0.060 * micro_noise
                + 0.090 * burst_noise * (0.40 + 0.60 * policy_bursts)
                + 0.085 * instability
                + 0.14 * policy_regimes
                + 0.07 * math.cos(update / 53.0 + 0.6)
            ),
            -0.44,
            0.44,
        )

        gameplay_rollouts = int(round(1980 + 120 * p + 40 * math.sin(update / 300.0)))
        trade_rollouts = int(round(1380 + 140 * p + 55 * math.cos(update / 260.0)))

        skip_pressure = (
            0.14 * math.sin(update / 120.0 + 0.8)
            + 0.10 * math.cos(update / 215.0 + 0.3)
            + 0.08 * math.sin(update / 38.0 + 1.2) * (0.35 + 0.65 * action_bursts)
        )
        market_churn = (
            0.12 * math.cos(update / 52.0 + 0.4)
            + 0.09 * math.sin(update / 87.0 + 1.1)
            + 0.07 * math.cos(update / 23.0 + 0.9) * (0.30 + 0.70 * burst_windows)
        )

        raw_propose = (
            phase_interp(progress, 0.34, 0.40, 0.44)
            + drift_wave(update, 0.020, 0.014)
            + 0.060 * action_regime
            + 0.070 * market_churn
            + 0.060 * action_micro * (0.30 + 0.70 * action_bursts)
        )
        raw_accept = (
            phase_interp(progress, 0.015, 0.035, 0.060)
            + drift_wave(update + 19, 0.008, 0.005)
            + 0.018 * action_regime
            + 0.020 * market_churn
            + 0.028 * action_micro * (0.35 + 0.65 * action_bursts)
        )
        raw_reject = (
            phase_interp(progress, 0.30, 0.24, 0.18)
            + drift_wave(update + 37, 0.018, 0.012)
            - 0.055 * action_regime
            + 0.060 * market_churn
            - 0.040 * action_micro * (0.30 + 0.70 * action_bursts)
        )
        raw_counter = (
            phase_interp(progress, 0.07, 0.05, 0.04)
            + drift_wave(update + 61, 0.006, 0.004)
            + 0.022 * action_regime
            + 0.030 * burst_windows * math.sin(update / 27.0)
            + 0.020 * action_micro * (0.40 + 0.60 * action_bursts)
        )
        skip_rate = (
            phase_interp(progress, 0.28, 0.23, 0.18)
            + 0.18 * skip_pressure
            - 0.035 * market_churn
            + 0.030 * burst_windows * math.cos(update / 31.0)
        )
        skip_rate = clamp(skip_rate, 0.12, 0.42)

        action_rest = max(1.0 - skip_rate, 1e-6)
        raw_rates = [max(raw_propose, 0.06), max(raw_accept, 0.005), max(raw_reject, 0.05), max(raw_counter, 0.02)]
        raw_total = sum(raw_rates)
        propose_rate = action_rest * raw_rates[0] / raw_total
        accept_rate = action_rest * raw_rates[1] / raw_total
        reject_rate = action_rest * raw_rates[2] / raw_total
        counter_rate = action_rest * raw_rates[3] / raw_total

        trade_propose = int(round(trade_rollouts * propose_rate))
        trade_accept = int(round(trade_rollouts * accept_rate))
        trade_reject = int(round(trade_rollouts * reject_rate))
        trade_counter = int(round(trade_rollouts * counter_rate))
        trade_skip = max(trade_rollouts - trade_propose - trade_accept - trade_reject - trade_counter, 0)

        gamma = clamp(float(last["gamma"]) + 0.010 * p, float(last["gamma"]), 0.972)
        entropy_coef = 0.0012 if progress < 0.70 else lerp(0.0012, 0.00095, smoothstep((progress - 0.70) / 0.30))

        total_loss = clamp(
            0.10
            + 0.18 * gp_value
            + 0.15 * tr_value
            + 0.03 * tom_loss
            - 0.10 * gp_reward
            - 0.14 * tr_reward
            + 0.04 * instability,
            -0.08,
            0.35,
        )

        rows.append(
            {
                "update": update,
                "gameplay_rollouts": gameplay_rollouts,
                "trade_rollouts": trade_rollouts,
                "gameplay_reward": gp_reward,
                "trade_reward": tr_reward,
                "trade_propose": trade_propose,
                "trade_accept": trade_accept,
                "trade_reject": trade_reject,
                "trade_counter": trade_counter,
                "trade_skip": trade_skip,
                "gameplay_policy_loss": gp_policy,
                "gameplay_value_loss": gp_value,
                "gameplay_entropy": gp_entropy,
                "trade_policy_loss": tr_policy,
                "trade_value_loss": tr_value,
                "trade_entropy": tr_entropy,
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


def write_log(df: pd.DataFrame, output_path: Path) -> None:
    lines = [f"MOCK unified target-profile projection for {len(df)} updates"]
    for _, row in df.iterrows():
        lines.append("")
        lines.append(f"Update {int(row['update'])}")
        lines.append(f"rollouts | gameplay={int(row['gameplay_rollouts'])} trade={int(row['trade_rollouts'])}")
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
            f"gamma={row['gamma']:.5f} tom_coef={lerp(0.02, 0.06, int(row['update']) / max(len(df) - 1, 1)):.5f} "
            f"trade_score={row['trade_reward_avg_logged'] * 4.0:.4f}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a mock unified log from target metric ranges.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-log", type=str, required=True)
    parser.add_argument("--total-updates", type=int, default=10000)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    projected = build_projection(df, args.total_updates)
    write_log(projected, Path(args.output_log))
    print(f"Wrote target-profile mock log to {args.output_log}")


if __name__ == "__main__":
    main()
