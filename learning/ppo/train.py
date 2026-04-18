# python learning/ppo/train.py --num-updates 50000 --hidden-dim 192

from __future__ import annotations

import argparse
import os
import sys
import random
from collections import deque
from typing import Dict, List
import numpy as np

import torch

# allow running from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from learning.league.league_manager import LeagueManager
from environment.catan_env import CatanEnv
from learning.ppo.ppo_policy import PPOPolicy
from learning.ppo.ppo_trainer import PPOTrainer
from learning.ppo.ppo_rollout_manager import PPORolloutManager

# Constants for observation/action feature dimensions
MAX_GAMEPLAY_ACTIONS = 256
GAMEPLAY_FEATURE_DIM = 40
MAX_TRADE_ACTIONS = 128
TRADE_FEATURE_DIM = 32


def save_checkpoint(trainer: PPOTrainer, save_dir: str, step: int, prefix: str = "ppo_checkpoint") -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_{step}.pt")
    trainer.save(path, step)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO baseline training for Catan")
    parser.add_argument("--num-updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--board-dim", type=int, default=64)
    parser.add_argument("--self-dim", type=int, default=64)
    parser.add_argument("--opponent-dim", type=int, default=64)
    parser.add_argument("--resources", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-loss-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--max-game-steps", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    return parser.parse_args()


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    policy = PPOPolicy(
        board_dim=args.board_dim,
        self_dim=args.self_dim,
        opponent_dim=args.opponent_dim,
        hidden_dim=args.hidden_dim,
        resources=args.resources,
        gameplay_feature_dim=GAMEPLAY_FEATURE_DIM,
        trade_feature_dim=TRADE_FEATURE_DIM,
    ).to(device)

    trainer = PPOTrainer(
        policy=policy,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        value_loss_coef=args.value_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        minibatch_size=args.batch_size,
        device=device,
    )
    
    rollout_manager = PPORolloutManager(
        num_envs=args.num_envs,
        num_workers=args.num_workers,
        device=device,
        enable_trading=True,
        max_steps=args.max_game_steps,
        opponent_paths=[],  # Start with default opponents
    )
    league = LeagueManager(checkpoint_dir=args.checkpoint_dir, frozen_ratio=0.2, load_existing=False)

    print(f"Starting PPO training for {args.num_updates} updates "
          f"(envs={args.num_envs}, rollout_steps={args.rollout_steps}, hidden_dim={args.hidden_dim}, seed={args.seed})")

    try:
        for update in range(args.num_updates):
            # --- LEAGUE-BASED SELF-PLAY ---
            opponent_paths = league.sample_opponents(k=3, policy_type="ppo") if len(league) > 0 else []
            rollout_manager.update_opponents(opponent_paths)

            raw_storage, next_value = rollout_manager.collect(policy=trainer.policy, steps=args.rollout_steps)

            # Apply shaped rewards if needed
            storage = raw_storage  # For now, no reward shaping

            metrics = trainer.update(storage, next_value)

            if update % args.log_interval == 0:
                print(f"Update {update} | "
                      f"total_loss={metrics['total_loss']:.4f} | "
                      f"gp_policy={metrics.get('gameplay_policy_loss', 0):.4f} | "
                      f"gp_value={metrics.get('gameplay_value_loss', 0):.4f} | "
                      f"gp_entropy={metrics.get('gameplay_entropy', 0):.4f} | "
                      f"tr_policy={metrics.get('trade_policy_loss', 0):.4f} | "
                      f"tr_value={metrics.get('trade_value_loss', 0):.4f} | "
                      f"tr_entropy={metrics.get('trade_entropy', 0):.4f}")

            if update % args.save_interval == 0 and update > 0:
                checkpoint_path = save_checkpoint(trainer, args.checkpoint_dir, update, "ppo")
                league.maybe_add_checkpoint(checkpoint_path, update)
                print(f"Saved checkpoint: {checkpoint_path}")

    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        if rollout_manager:
            rollout_manager.close()

    # Save final checkpoint
    final_path = save_checkpoint(trainer, args.checkpoint_dir, args.num_updates, "ppo_final")
    print(f"Training completed. Final checkpoint: {final_path}")


if __name__ == "__main__":
    args = build_arg_parser()
    train(args)