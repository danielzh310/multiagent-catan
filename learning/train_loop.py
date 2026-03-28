from __future__ import annotations

import os

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
    )

    batch_processor = DualBatchProcessor(device=device)

    trainer = DualTrainer(
        gameplay_model=gameplay_model,
        trade_model=trade_model,
        device=device,
    )

    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting training for {num_updates} updates")

    for update in range(num_updates):
        gameplay_rollouts, trade_rollouts = rollout_manager.collect()
        print("gameplay rollouts:", len(gameplay_rollouts))
        print("trade rollouts:", len(trade_rollouts))

        gameplay_batch = batch_processor.process_gameplay_rollouts(gameplay_rollouts)
        trade_batch = batch_processor.process_trade_rollouts(trade_rollouts)
        print("gameplay batch empty:", gameplay_batch == {})
        print("trade batch empty:", trade_batch == {})
        

        tom_targets = None
        if trade_batch:
            tom_targets = build_tom_targets_from_history(trade_batch["obs"])

        metrics = trainer.train_joint_step(
            gameplay_batch,
            trade_batch,
            tom_targets=tom_targets,
        )

        print(f"Update {update}: {metrics}")

        if (update + 1) % 8 == 0:
            path = save_checkpoint_pair(
                gameplay_model,
                trade_model,
                save_dir=save_dir,
                step=update + 1,
            )
            print(f"Saved checkpoint to {path}")

    print("Training complete")


if __name__ == "__main__":
    train()