from __future__ import annotations

import os
from collections import deque

from learning.gameplay.build_gameplay_model import build_gameplay_model
from learning.trade.build_trade_model import build_trade_model

from learning.ppo_trainer.dual_rollout_manager import DualRolloutManager
from learning.ppo_trainer.dual_batch_processor import DualBatchProcessor
from learning.ppo_trainer.dual_trainer import DualTrainer

from learning.utils.checkpoint_pairing import save_checkpoint_pair
from learning.utils.trade_targets import build_tom_targets_from_history


def train(
    num_updates: int = 200,
    num_envs: int = 8,
    num_steps: int = 128,
    save_dir: str = "./checkpoints",
    device: str = "cpu",
):
    gameplay_model = build_gameplay_model(device=device)
    trade_model = build_trade_model(device=device)

    rollout_manager = DualRolloutManager(
        num_envs=num_envs,
        num_steps=num_steps,
        gameplay_builder=build_gameplay_model,
        trade_builder=build_trade_model,
        device=device,
        debug=False,
    )

    batch_processor = DualBatchProcessor(device=device)

    trainer = DualTrainer(
        gameplay_model=gameplay_model,
        trade_model=trade_model,
        device=device,
    )

    os.makedirs(save_dir, exist_ok=True)

    trade_reward_window = deque(maxlen=20)
    gameplay_reward_window = deque(maxlen=20)
    trade_entropy_window = deque(maxlen=20)

    print(f"Starting training for {num_updates} updates")

    for update in range(num_updates):
        progress = update / float(max(num_updates - 1, 1))
        trainer.set_progress(progress)

        gameplay_rollouts, trade_rollouts, rollout_stats = rollout_manager.collect()

        gameplay_batch = batch_processor.process_gameplay_rollouts(gameplay_rollouts)
        trade_batch = batch_processor.process_trade_rollouts(trade_rollouts)

        tom_targets = None
        if trade_batch:
            tom_targets = build_tom_targets_from_history(trade_batch["obs"])

        metrics = trainer.train_joint_step(
            gameplay_batch,
            trade_batch,
            tom_targets=tom_targets,
        )

        trade_reward_window.append(rollout_stats["trade_reward_mean"])
        gameplay_reward_window.append(rollout_stats["gameplay_reward_mean"])
        trade_entropy_window.append(metrics["trade_entropy"])

        avg_trade_reward = sum(trade_reward_window) / len(trade_reward_window)
        avg_gameplay_reward = sum(gameplay_reward_window) / len(gameplay_reward_window)
        avg_trade_entropy = sum(trade_entropy_window) / len(trade_entropy_window)

        print(f"\nUpdate {update}")

        print(
            f"rollouts | gameplay={rollout_stats['gameplay_rollouts']} "
            f"trade={rollout_stats['trade_rollouts']}"
        )

        print(
            f"reward   | gameplay={rollout_stats['gameplay_reward_mean']:.4f} "
            f"(avg={avg_gameplay_reward:.4f}) "
            f"trade={rollout_stats['trade_reward_mean']:.4f} "
            f"(avg={avg_trade_reward:.4f})"
        )

        print(
            f"trade actions | propose={rollout_stats['trade_propose_count']} "
            f"accept={rollout_stats['trade_accept_count']} "
            f"reject={rollout_stats['trade_reject_count']} "
            f"counter={rollout_stats['trade_counter_count']} "
            f"skip={rollout_stats['trade_skip_count']}"
        )

        print(
            f"entropy | gameplay={metrics['gameplay_entropy']:.4f} "
            f"(coef={metrics['gameplay_entropy_coef']:.5f}) "
            f"trade={metrics['trade_entropy']:.4f} "
            f"(avg={avg_trade_entropy:.4f}, coef={metrics['trade_entropy_coef']:.6f})"
        )

        print(
            f"losses  | gp_policy={metrics['gameplay_policy_loss']:.4f} "
            f"gp_value={metrics['gameplay_value_loss']:.4f} "
            f"tr_policy={metrics['trade_policy_loss']:.4f} "
            f"tr_value={metrics['trade_value_loss']:.4f} "
            f"tr_tom={metrics['trade_tom_loss']:.6f}"
        )

        if (update + 1) % 20 == 0:
            path = save_checkpoint_pair(
                gameplay_model,
                trade_model,
                save_dir=save_dir,
                step=update + 1,
            )
            print(f"saved checkpoint -> {path}")

    print("training complete")


if __name__ == "__main__":
    train()