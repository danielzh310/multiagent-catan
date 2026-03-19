"""
Main training loop for multi-agent Catan.

This file ties everything together:
- builds model
- initializes rollout workers
- runs PPO updates
- manages self-play pool
- runs evaluation
"""

import os
import copy
import time
import torch

from learning.networks.build_model import build_model
from learning.ppo_trainer.config import PPOConfig
from learning.ppo_trainer.trainer import PPOTrainer
from learning.ppo_trainer.batch_processor import BatchProcessor
from learning.ppo_trainer.rollout_manager import RolloutManager
from learning.ppo_trainer.distributed_rollout import DistributedRolloutManager
from learning.ppo_trainer.evaluator import Evaluator
from learning.ppo_trainer.distributed_eval import DistributedEvalManager
from learning.ppo_trainer.opponent_scheduler import refresh_opponents
from learning.ppo_trainer.eval_protocol import run_eval_protocol


def make_rollout_manager(config, seed_offset=0):
    def _fn():
        return RolloutManager(
            num_envs=config.num_envs_per_process,
            num_steps=config.num_steps,
            model_builder=build_model,
            seed=config.seed + seed_offset,
        )

    return _fn


def make_evaluator(config, seed_offset=0):
    def _fn():
        return Evaluator(
            model_builder=build_model,
            seed=config.seed + seed_offset,
        )

    return _fn


def train():
    config = PPOConfig()

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")

    model = build_model()
    model.to(device)

    trainer = PPOTrainer(model, config)
    batch_processor = BatchProcessor(config, lstm_dim=model.lstm_dim, device=device)

    rollout_fns = [
        make_rollout_manager(config, i * 1000)
        for i in range(config.num_processes)
    ]

    rollout_manager = DistributedRolloutManager(rollout_fns)

    eval_fns = [
        make_evaluator(config, i * 1000)
        for i in range(config.num_eval_processes)
    ]

    eval_manager = DistributedEvalManager(eval_fns)

    policy_pool = []
    random_policy = build_model().state_dict()

    total_updates = config.total_env_steps // (config.num_steps * config.num_processes * config.num_envs_per_process)

    print(f"Starting training for {total_updates} updates")

    start_time = time.time()

    for update in range(1, total_updates + 1):
        rollout_data = rollout_manager.gather_rollouts()
        batch_processor.process_rollouts(rollout_data)

        value_loss, action_loss, entropy_loss = trainer.update(batch_processor)

        if update % config.add_policy_every == 0:
            policy_pool.append(copy.deepcopy(model.state_dict()))

            if len(policy_pool) > config.num_policies_to_store:
                policy_pool.pop(0)

        if update % config.update_opponents_every == 0 and len(policy_pool) > 0:
            refresh_opponents(policy_pool, rollout_manager)

        if config.use_linear_lr_decay:
            lr = config.learning_rate * (1 - update / total_updates)
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = lr

        if update % config.eval_every == 0:
            log, summary = run_eval_protocol(
                distributed_eval_manager=eval_manager,
                current_policy=model,
                stored_policy_pool=policy_pool,
                random_policy_state=random_policy,
                config=config,
                update_idx=update,
            )

            print(summary)

        if update % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"[Update {update}] "
                f"value_loss={value_loss:.4f}, "
                f"policy_loss={action_loss:.4f}, "
                f"entropy={entropy_loss:.4f}, "
                f"time={elapsed:.2f}s"
            )

        if config.checkpoint_path and update % 100 == 0:
            os.makedirs(config.checkpoint_path, exist_ok=True)
            path = os.path.join(config.checkpoint_path, f"model_{update}.pt")
            torch.save(model.state_dict(), path)

    rollout_manager.close()
    eval_manager.close()

    print("Training complete.")


if __name__ == "__main__":
    train()