# python learning/dqn/train.py --num-updates 50000 --hidden-dim 192

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from typing import Dict, List, Any

import torch

from .dqn_policy import DQNBaselinePolicy
from .dqn_trainer import DQNTrainer
from .epsilon_scheduler import EpsilonScheduler
from .replay_buffer import ReplayBuffer


def save_checkpoint(trainer: DQNTrainer, save_dir: str, step: int, prefix: str = "dqn_checkpoint") -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_{step}.pt")
    trainer.save(path)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DQN baseline training for Catan")
    parser.add_argument("--num-updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--board-dim", type=int, default=64)
    parser.add_argument("--self-dim", type=int, default=64)
    parser.add_argument("--opponent-dim", type=int, default=64)
    parser.add_argument("--resources", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    return parser


def collect_rollout(
    env,
    policy: DQNBaselinePolicy,
    epsilon: float,
    device: str,
) -> List[Dict[str, Any]]:
    """
    Collect a single rollout from the environment using the DQN policy.
    This is a simplified version - in practice you'd want proper rollout collection.
    """
    rollout_data = []

    obs = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 1000:  # Safety limit
        # Convert obs to tensors
        obs_tensor = {k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(device)
                     for k, v in obs.items()}

        # Determine phase (simplified - you'd need proper phase detection)
        phase = "gameplay" if "gameplay_candidates" in obs else "trade"

        # Get action from policy
        with torch.no_grad():
            action = policy.act(obs_tensor, phase, epsilon)

        # Take action in environment
        next_obs, reward, done, info = env.step(action)

        # Store transition
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "phase": phase,
        }
        rollout_data.append(transition)

        obs = next_obs
        step_count += 1

    return rollout_data


def train(args: argparse.Namespace) -> None:
    device = args.device

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize DQN trainer with Catan-compatible dimensions
    trainer = DQNTrainer(
        board_dim=args.board_dim,
        self_dim=args.self_dim,
        opponent_dim=args.opponent_dim,
        hidden_dim=args.hidden_dim,
        resources=args.resources,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
    )

    replay_buffer = ReplayBuffer(args.buffer_size)
    epsilon_scheduler = EpsilonScheduler()

    loss_window = deque(maxlen=100)

    print(
        f"Starting DQN baseline training for {args.num_updates} updates "
        f"(batch_size={args.batch_size}, buffer_size={args.buffer_size}, hidden_dim={args.hidden_dim}, seed={args.seed})"
    )

    # Import environment here to avoid circular imports
    from ...environment.catan_env import CatanEnv

    env = CatanEnv()

    for update in range(args.num_updates):
        # Collect rollout
        epsilon = epsilon_scheduler.get_epsilon(update)
        rollout = collect_rollout(env, trainer.policy, epsilon, device)

        # Add to replay buffer
        for transition in rollout:
            replay_buffer.add(
                obs=transition["obs"],
                action=transition["action"],
                reward=transition["reward"],
                next_obs=transition["next_obs"],
                done=transition["done"],
                phase=transition["phase"],
            )

        # Update if buffer has enough data
        if len(replay_buffer) >= args.batch_size:
            batch = replay_buffer.sample(args.batch_size)

            # Convert batch to proper format for trainer
            formatted_batch = {
                "obs": {k: torch.stack([torch.tensor(item["obs"][k], dtype=torch.float32) for item in batch])
                       for k in batch[0]["obs"].keys()},
                "actions": {phase: torch.tensor([item["action"][phase] for item in batch if item["phase"] == phase])
                           for phase in ["gameplay", "trade"] if any(item["phase"] == phase for item in batch)},
                "rewards": torch.tensor([item["reward"] for item in batch], dtype=torch.float32),
                "next_obs": {k: torch.stack([torch.tensor(item["next_obs"][k], dtype=torch.float32) for item in batch])
                            for k in batch[0]["next_obs"].keys()},
                "dones": torch.tensor([item["done"] for item in batch], dtype=torch.float32),
                "phases": [item["phase"] for item in batch],
            }

            metrics = trainer.update(formatted_batch)
            loss_window.append(metrics["td_loss"])

            avg_loss = sum(loss_window) / len(loss_window)

            if update % 100 == 0:
                print(f"Update {update}: td_loss={avg_loss:.4f}, q_mean={metrics['q_mean']:.4f}")

        if (update + 1) % 1000 == 0:
            path = save_checkpoint(trainer, args.checkpoint_dir, update + 1)
            print(f"saved checkpoint -> {path}")

    print("DQN baseline training complete")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)
    args = parser.parse_args()
    train(args)