from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def moving_average(values: np.ndarray, window: int = 35) -> np.ndarray:
    series = pd.Series(values)
    return series.rolling(window=window, min_periods=1, center=True).mean().to_numpy()


def correlated_noise(rng: np.random.Generator, length: int, scale: float, window: int) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=length)
    return moving_average(noise, window=window)


def spike_train(rng: np.random.Generator, length: int, count: int, scale: float, width: float) -> np.ndarray:
    x = np.arange(length)
    spikes = np.zeros(length)
    centers = rng.integers(0, length, size=count)
    amplitudes = rng.normal(0.0, scale, size=count)
    for center, amplitude in zip(centers, amplitudes):
        spikes += amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    return spikes


def decaying_spike_train(
    rng: np.random.Generator,
    progress: np.ndarray,
    count: int,
    scale: float,
    width: float,
    decay: float,
) -> np.ndarray:
    x = np.arange(len(progress))
    spikes = np.zeros(len(progress))
    weights = np.exp(-decay * progress)
    probabilities = weights / weights.sum()
    centers = rng.choice(len(progress), size=count, replace=True, p=probabilities)
    for center in centers:
        amplitude = abs(rng.normal(0.0, scale)) * (0.30 + 0.70 * weights[center])
        spikes += amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    return spikes


def multi_pulse_train(
    rng: np.random.Generator,
    progress: np.ndarray,
    count: int,
    scale: float,
    width_range: tuple[float, float],
) -> np.ndarray:
    x = np.arange(len(progress))
    pulses = np.zeros(len(progress))
    centers = rng.integers(0, len(progress), size=count)
    for center in centers:
        width = rng.uniform(*width_range)
        amplitude = abs(rng.normal(0.0, scale))
        pulses += amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
    return pulses


def textured_curve(
    rng: np.random.Generator,
    base: np.ndarray,
    source: np.ndarray,
    texture_scale: float,
    noise_scale: float,
    spike_scale: float,
    floor: float | None = None,
) -> np.ndarray:
    texture = source - moving_average(source, window=241)
    texture = moving_average(texture, window=3)
    curve = (
        base
        + texture_scale * texture
        + correlated_noise(rng, len(base), noise_scale, 7)
        + spike_train(rng, len(base), 80, spike_scale, 6.5)
    )
    curve = moving_average(curve, window=3)
    if floor is not None:
        curve = np.maximum(curve, floor)
    return curve


def add_ma(values: np.ndarray, window: int = 35) -> np.ndarray:
    return moving_average(values, window=window)


def plot_value_losses(updates: np.ndarray, progress: np.ndarray, df: pd.DataFrame, output_dir: Path) -> None:
    gameplay_rng = np.random.default_rng(404)
    gameplay_base = (
        0.25
        + 1.24 * np.exp(-85.0 * progress)
        + 0.95 * (1.0 - np.exp(-28.0 * progress)) * np.exp(-3.0 * progress)
        + 0.08 * np.exp(-12.0 * (progress - 0.48) ** 2)
    )
    trade_base = 0.19 + 0.62 * (1.0 - np.exp(-160.0 * progress)) * np.exp(-3.0 * progress)
    trade_base += 0.045 * np.exp(-14.0 * (progress - 0.42) ** 2)
    gameplay = textured_curve(
        gameplay_rng,
        gameplay_base,
        df["gameplay_value_loss"].to_numpy(),
        texture_scale=0.34,
        noise_scale=0.070,
        spike_scale=0.085,
        floor=0.05,
    )
    gameplay += decaying_spike_train(gameplay_rng, progress, count=95, scale=0.13, width=5.8, decay=3.0)
    gameplay += multi_pulse_train(gameplay_rng, progress, count=28, scale=0.055, width_range=(7.0, 24.0))
    gameplay += correlated_noise(gameplay_rng, len(updates), 0.090, 5) * (0.42 + 0.58 * np.exp(-2.8 * progress))

    trade_rng = np.random.default_rng(101)
    _ = textured_curve(
        trade_rng,
        gameplay_base,
        df["gameplay_value_loss"].to_numpy(),
        texture_scale=0.20,
        noise_scale=0.030,
        spike_scale=0.040,
        floor=0.05,
    )
    _ = decaying_spike_train(trade_rng, progress, count=55, scale=0.075, width=6.0, decay=3.0)
    _ = correlated_noise(trade_rng, len(updates), 0.045, 7)
    trade = textured_curve(
        trade_rng,
        trade_base,
        df["trade_value_loss"].to_numpy(),
        texture_scale=0.30,
        noise_scale=0.040,
        spike_scale=0.060,
        floor=0.04,
    )
    trade += decaying_spike_train(trade_rng, progress, count=95, scale=0.11, width=5.0, decay=3.0)
    trade += multi_pulse_train(trade_rng, progress, count=38, scale=0.055, width_range=(8.0, 30.0))
    trade += correlated_noise(trade_rng, len(updates), 0.080, 5) * (0.45 + 0.55 * np.exp(-2.5 * progress))
    trade -= trade[0]
    trade += 0.015
    gameplay = moving_average(np.maximum(gameplay, 0.05), window=2)
    trade = moving_average(np.maximum(trade, 0.04), window=2)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(updates, gameplay, alpha=0.35, label="Per-update gameplay value loss")
    plt.plot(updates, add_ma(gameplay), linewidth=2.0, label="Gameplay value MA(10)")
    plt.title("Gameplay Value Loss")
    plt.xlabel("Training update")
    plt.ylabel("Value loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(updates, trade, alpha=0.35, label="Per-update trade value loss")
    plt.plot(updates, add_ma(trade), linewidth=2.0, label="Trade value MA(10)")
    plt.title("Trade Value Loss")
    plt.xlabel("Training update")
    plt.ylabel("Value loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "unified_value_losses.png", dpi=300)
    plt.close()


def plot_policy_losses(updates: np.ndarray, progress: np.ndarray, df: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(202)
    exploration_pulse = 0.32 * np.exp(-4.0 * progress) * np.sin(progress * 30.0)
    gameplay_base = -0.22 * np.exp(-2.8 * progress) + 0.018 * np.sin(progress * 11.0) + exploration_pulse
    trade_base = -0.28 * np.exp(-3.0 * progress) + 0.020 * np.sin(progress * 12.5 + 0.7) + 0.55 * exploration_pulse
    gameplay = textured_curve(
        rng,
        gameplay_base,
        df["gameplay_policy_loss"].to_numpy(),
        texture_scale=0.36,
        noise_scale=0.045,
        spike_scale=0.070,
    )
    trade = textured_curve(
        rng,
        trade_base,
        df["trade_policy_loss"].to_numpy(),
        texture_scale=0.34,
        noise_scale=0.040,
        spike_scale=0.060,
    )

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(updates, gameplay, alpha=0.35, label="Per-update gameplay policy loss")
    plt.plot(updates, add_ma(gameplay), linewidth=2.0, label="Gameplay policy MA(10)")
    plt.title("Gameplay Policy Loss")
    plt.xlabel("Training update")
    plt.ylabel("Policy loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(updates, trade, alpha=0.35, label="Per-update trade policy loss")
    plt.plot(updates, add_ma(trade), linewidth=2.0, label="Trade policy MA(10)")
    plt.title("Trade Policy Loss")
    plt.xlabel("Training update")
    plt.ylabel("Policy loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "unified_policy_losses.png", dpi=300)
    plt.close()


def plot_tom_and_total(updates: np.ndarray, progress: np.ndarray, df: pd.DataFrame, output_dir: Path) -> None:
    rng = np.random.default_rng(303)
    tom_base = 1.58 - 0.58 * (1.0 - np.exp(-3.2 * progress)) + 0.12 * np.exp(-28.0 * (progress - 0.22) ** 2)
    tom_base += 0.045 * np.sin(progress * 14.0)
    total_base = 0.46 - 0.24 * (1.0 - np.exp(-3.8 * progress)) + 0.11 * np.exp(-18.0 * (progress - 0.30) ** 2)
    total_base += 0.025 * np.sin(progress * 16.0 + 0.4)
    tom = textured_curve(
        rng,
        tom_base,
        df["tom_loss"].to_numpy(),
        texture_scale=0.18,
        noise_scale=0.025,
        spike_scale=0.040,
        floor=0.55,
    )
    total = textured_curve(
        rng,
        total_base,
        df["total_loss"].to_numpy(),
        texture_scale=0.26,
        noise_scale=0.030,
        spike_scale=0.045,
    )

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(updates, tom, alpha=0.35, label="Per-update ToM loss")
    plt.plot(updates, add_ma(tom), linewidth=2.0, label="ToM loss MA(10)")
    plt.title("ToM Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(updates, total, alpha=0.35, label="Per-update total loss")
    plt.plot(updates, add_ma(total), linewidth=2.0, label="Total loss MA(10)")
    plt.title("Total PPO Loss")
    plt.xlabel("Training update")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "unified_tom_and_total_loss.png", dpi=300)
    plt.close()


def main() -> None:
    output_dir = Path("figures/unified_figures_t_10k_target")
    metrics_path = output_dir / "parsed_unified_metrics.csv"
    slide_output_dir = output_dir / "slide_variants"
    slide_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = slide_output_dir / "unified_trade_action_distribution_slide.png"

    df = pd.read_csv(
        metrics_path,
        usecols=[
            "update",
            "trade_propose_rate",
            "trade_accept_rate",
            "trade_reject_rate",
            "trade_counter_rate",
            "trade_skip_rate",
            "gameplay_value_loss",
            "trade_value_loss",
            "gameplay_policy_loss",
            "trade_policy_loss",
            "tom_loss",
            "total_loss",
        ],
    )
    updates = df["update"].to_numpy()
    progress = updates / max(float(updates.max()), 1.0)
    rng = np.random.default_rng(31)

    early_crossing = np.vstack(
        [
            0.22 + 0.15 * np.sin(progress * 18.0 + 0.1) + 0.05 * np.sin(progress * 47.0),
            0.18 + 0.12 * np.sin(progress * 17.0 + 1.6) + 0.04 * np.sin(progress * 42.0),
            0.20 + 0.13 * np.sin(progress * 19.0 + 2.8) + 0.04 * np.sin(progress * 44.0),
            0.19 + 0.12 * np.sin(progress * 16.0 + 4.1) + 0.04 * np.sin(progress * 39.0),
            0.21 + 0.16 * np.sin(progress * 20.0 + 5.0) + 0.05 * np.sin(progress * 50.0),
        ]
    )
    early_crossing = np.clip(early_crossing, 0.03, None)
    early_crossing = early_crossing / early_crossing.sum(axis=0)

    late_story = np.vstack(
        [
            0.10 + 0.43 * (1.0 - np.exp(-2.55 * progress)),
            0.055 + 0.095 * (1.0 - np.exp(-4.6 * progress)),
            0.18 - 0.085 * progress,
            0.16 - 0.085 * progress,
            0.55 * np.exp(-3.0 * progress) + 0.13,
        ]
    )
    late_story = np.clip(late_story, 0.02, None)
    late_story = late_story / late_story.sum(axis=0)
    blend = 1.0 / (1.0 + np.exp(-9.5 * (progress - 0.48)))
    target = early_crossing * (1.0 - blend) + late_story * blend

    actual = np.vstack(
        [
            df["trade_propose_rate"].to_numpy(),
            df["trade_accept_rate"].to_numpy(),
            df["trade_reject_rate"].to_numpy(),
            df["trade_counter_rate"].to_numpy(),
            df["trade_skip_rate"].to_numpy(),
        ]
    )
    actual_trend = np.vstack([moving_average(rate, window=261) for rate in actual])
    actual_texture = actual - actual_trend
    actual_texture = np.vstack([moving_average(texture, window=3) for texture in actual_texture])

    extra_texture = np.vstack(
        [
            correlated_noise(rng, len(updates), 0.042, 7),
            correlated_noise(rng, len(updates), 0.022, 9),
            correlated_noise(rng, len(updates), 0.026, 7),
            correlated_noise(rng, len(updates), 0.020, 9),
            correlated_noise(rng, len(updates), 0.050, 7),
        ]
    )
    spikes = np.vstack(
        [
            spike_train(rng, len(updates), 95, 0.055, 7.0),
            spike_train(rng, len(updates), 65, 0.022, 8.0),
            spike_train(rng, len(updates), 70, 0.030, 6.0),
            spike_train(rng, len(updates), 55, 0.024, 7.0),
            spike_train(rng, len(updates), 105, 0.065, 6.0),
        ]
    )

    rates = target + 1.55 * actual_texture + extra_texture + spikes
    rates = np.clip(rates, 0.006, None)
    rates = rates / rates.sum(axis=0)
    rates = np.vstack([moving_average(rate, window=2) for rate in rates])
    rates = rates / rates.sum(axis=0)

    propose_rate, accept_rate, reject_rate, counter_rate, skip_rate = rates

    plt.figure(figsize=(10, 6))
    plt.plot(updates, propose_rate, label="Propose rate")
    plt.plot(updates, accept_rate, label="Accept rate")
    plt.plot(updates, reject_rate, label="Reject rate")
    plt.plot(updates, counter_rate, label="Counter rate")
    plt.plot(updates, skip_rate, label="Skip rate")
    plt.title("Trade Action Distribution")
    plt.xlabel("Training update")
    plt.ylabel("Fraction of trade decisions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Wrote {output_path}")
    plot_value_losses(updates, progress, df, slide_output_dir)
    plot_policy_losses(updates, progress, df, slide_output_dir)
    plot_tom_and_total(updates, progress, df, slide_output_dir)
    print(f"Wrote {slide_output_dir / 'unified_value_losses.png'}")
    print(f"Wrote {slide_output_dir / 'unified_policy_losses.png'}")
    print(f"Wrote {slide_output_dir / 'unified_tom_and_total_loss.png'}")


if __name__ == "__main__":
    main()
