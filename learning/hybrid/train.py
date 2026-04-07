# python learning/hybrid/train.py --num-updates 200 --num-envs 8 --hidden-dim 192

from __future__ import annotations

import argparse
import os
import sys
import random
from collections import deque
from typing import Dict, List

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from learning.hybrid.hybrid_policy import HybridPolicy
from learning.hybrid.hybrid_rollout_manager import HybridRolloutManager
from learning.hybrid.hybrid_checkpoint import HybridCheckpointManager
from learning.dqn.dqn_policy import DQNBaselinePolicy
from learning.dqn.dqn_trainer import DQNTrainer
from learning.dqn.replay_buffer import ReplayBuffer
from learning.ppo_trainer.trainer import PPOTrainer
from learning.ppo_trainer.batch_processor import BatchProcessor
from learning.league.league_manager import LeagueManager
from learning.rewards.reward_shaper import RewardShaper
from learning.trade.trade_policy import TradePolicy


class HybridTrainer:
    def __init__(
        self,
        dqn_trainer: DQNTrainer,
        trade_trainer: PPOTrainer,
        device: str = "cpu",
    ):
        self.dqn_trainer = dqn_trainer
        self.trade_trainer = trade_trainer
        self.device = device

    def update(self, gameplay_batch: Dict, trade_batch_processor: Any) -> Dict[str, float]:
        # Update DQN for gameplay
        dqn_metrics = self.dqn_trainer.update(gameplay_batch)
        
        # Update PPO for trade
        trade_metrics = self.trade_trainer.update(trade_batch_processor)
        
        return {
            "dqn_loss": dqn_metrics.get("loss", 0.0),
            "trade_policy_loss": trade_metrics.get("action_loss", 0.0),
            "trade_value_loss": trade_metrics.get("value_loss", 0.0),
            "trade_entropy": trade_metrics.get("entropy_loss", 0.0),
        }


def save_checkpoint(
    dqn_trainer: DQNTrainer,
    trade_trainer: PPOTrainer,
    save_dir: str,
    step: int
) -> str:
    checkpoint_manager = HybridCheckpointManager(save_dir)
    return checkpoint_manager.save(step, dqn_trainer, trade_trainer)


def compute_rollout_stats(storage: List[Dict]) -> Dict[str, float]:
    gameplay_rewards = [x["reward"] for x in storage if x.get("phase") == "gameplay"]
    trade_rewards = [x["reward"] for x in storage if x.get("phase") == "trade"]

    counts = {
        "propose": 0,
        "accept": 0,
        "reject": 0,
        "counter": 0,
        "skip": 0,
    }

    for x in storage:
        if x.get("phase") != "trade":
            continue

        env_action = x.get("env_action", {})
        action_type = env_action.get("type", "")

        if action_type == "propose_trade":
            counts["propose"] += 1
        elif action_type == "accept_trade":
            counts["accept"] += 1
        elif action_type == "reject_trade":
            counts["reject"] += 1
        elif action_type == "counter_trade":
            counts["counter"] += 1
        elif action_type == "skip_trade":
            counts["skip"] += 1

    total_trade_actions = max(sum(counts.values()), 1)

    return {
        "gameplay_rollouts": len(gameplay_rewards),
        "trade_rollouts": len(trade_rewards),
        "gameplay_reward_mean": float(sum(gameplay_rewards) / max(len(gameplay_rewards), 1)),
        "trade_reward_mean": float(sum(trade_rewards) / max(len(trade_rewards), 1)),
        "trade_propose_count": counts["propose"],
        "trade_accept_count": counts["accept"],
        "trade_reject_count": counts["reject"],
        "trade_counter_count": counts["counter"],
        "trade_skip_count": counts["skip"],
        "trade_propose_rate": counts["propose"] / total_trade_actions,
        "trade_accept_rate": counts["accept"] / total_trade_actions,
        "trade_reject_rate": counts["reject"] / total_trade_actions,
        "trade_counter_rate": counts["counter"] / total_trade_actions,
        "trade_skip_rate": counts["skip"] / total_trade_actions,
    }


def collect_rollouts(
    rollout_manager: HybridRolloutManager,
    hybrid_policy: HybridPolicy,
    epsilon_scheduler: Any,
    num_steps: int,
    device: str,
) -> List[Dict]:
    """Collect rollouts using the hybrid policy."""
    storage = []

    for step in range(num_steps):
        # Get current phase for each environment
        phases = [rollout_manager._route_phase(env) for env in rollout_manager.envs]

        for env_idx, (env, phase) in enumerate(zip(rollout_manager.envs, phases)):
            if phase == "gameplay":
                # Build gameplay observation
                obs = rollout_manager._build_gameplay_obs(env)

                # Get action from DQN
                epsilon = epsilon_scheduler.get_epsilon(step) if epsilon_scheduler else 0.0
                action_dict, _ = hybrid_policy.get_gameplay_action(obs, epsilon)

                # Decode action for environment
                action_idx = int(action_dict["gameplay_action"].item())
                env_action = rollout_manager._decode_gameplay_action(action_idx, env)

                # Take action in environment
                next_obs_raw, reward, done, info = env.step(env_action)

                # Store transition
                transition = {
                    "obs": obs,
                    "action": action_dict,
                    "reward": reward,
                    "next_obs": rollout_manager._build_gameplay_obs(env),
                    "done": done,
                    "phase": phase,
                    "env_action": env_action,
                }
                storage.append(transition)

            elif phase == "trade":
                # Build trade observation
                obs = rollout_manager._build_trade_obs(env)

                # Get action from PPO (placeholder for now)
                action_dict, _ = hybrid_policy.get_trade_action(obs)

                # Decode action for environment (placeholder - use skip for now)
                env_action = {"type": "skip_trade"}

                # Take action in environment
                next_obs_raw, reward, done, info = env.step(env_action)

                # Store transition
                transition = {
                    "obs": obs,
                    "action": action_dict,
                    "reward": reward,
                    "next_obs": rollout_manager._build_trade_obs(env),
                    "done": done,
                    "phase": phase,
                    "env_action": env_action,
                }
                storage.append(transition)

            # Reset environment if done
            if done:
                rollout_manager.obs[env_idx] = env.reset()

    return storage


def apply_shaped_rewards(
    storage: List[Dict],
    reward_shaper: RewardShaper,
    progress: float,
    trade_curriculum_fraction: float = 0.55,
    trade_reward_scale_start: float = 0.35,
    trade_reward_scale_end: float = 1.00,
) -> List[Dict]:
    shaped_storage = []
    consecutive_skips = 0

    if progress <= trade_curriculum_fraction:
        scaled = progress / max(trade_curriculum_fraction, 1e-8)
        curriculum_scale = trade_reward_scale_start + (trade_reward_scale_end - trade_reward_scale_start) * scaled
    else:
        curriculum_scale = trade_reward_scale_end

    for item in storage:
        new_item = dict(item)

        if item.get("phase") == "trade":
            env_action = item.get("env_action", {})
            action_type = env_action.get("type", "skip_trade")

            if action_type == "skip_trade":
                consecutive_skips += 1
            else:
                consecutive_skips = 0

            shaped_reward = reward_shaper.trade_step_reward(
                action_type=action_type,
                reward_signal=float(item["reward"]),
                consecutive_skips=consecutive_skips,
                tom_loss=0.0,
                curriculum_scale=curriculum_scale,
            )
            new_item["reward"] = shaped_reward
        else:
            new_item["reward"] = float(item["reward"])

        shaped_storage.append(new_item)

    return shaped_storage


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid DQN+PPO training loop")
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def train(args: argparse.Namespace) -> None:
    device = args.device

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize policies
    dqn_policy = DQNBaselinePolicy(
        board_dim=64,
        self_dim=64,
        opponent_dim=64,
        hidden_dim=args.hidden_dim,
        resources=5,
        device=device,
    )
    trade_policy = TradePolicy(device=device)
    
    # Initialize trainers
    replay_buffer = ReplayBuffer(args.buffer_size)
    dqn_trainer = DQNTrainer(
        board_dim=64,
        self_dim=64,
        opponent_dim=64,
        hidden_dim=args.hidden_dim,
        resources=5,
        device=device,
        lr=1e-3,
        gamma=0.99,
        target_update_freq=1000,
    )
    
    # PPO config (placeholder - you'd need to define proper config)
    class PPOConfig:
        def __init__(self):
            self.clip_param = 0.1
            self.ppo_epochs = 3
            self.num_minibatches = 4
            self.value_loss_coef = 0.25
            self.entropy_coef_start = 0.01
            self.max_grad_norm = 0.25
            self.gamma = 0.99
            self.gae_lambda = 0.95
            self.recompute_returns = False
            self.learning_rate = 7.5e-5
            self.adam_eps = 1e-8
    
    ppo_config = PPOConfig()
    trade_trainer = PPOTrainer(trade_policy, ppo_config)
    
    # Initialize hybrid components
    rollout_manager = HybridRolloutManager(num_envs=args.num_envs, env_config={}, device=device)
    league = LeagueManager(checkpoint_dir=args.checkpoint_dir, frozen_ratio=0.2)
    reward_shaper = RewardShaper()

    gameplay_reward_window = deque(maxlen=20)
    trade_reward_window = deque(maxlen=20)

    best_trade_reward = float("-inf")
    best_trade_update = -1

    print(
        f"Starting hybrid training for {args.num_updates} updates "
        f"(envs={args.num_envs}, rollout_steps={args.rollout_steps}, hidden_dim={args.hidden_dim}, seed={args.seed})"
    )

    for update in range(args.num_updates):
        progress = update / float(max(args.num_updates - 1, 1))

        # Collect rollouts
        raw_storage = collect_rollouts(
            rollout_manager=rollout_manager,
            hybrid_policy=HybridPolicy(dqn_policy, trade_policy, device=device),
            epsilon_scheduler=None,  # Placeholder - implement epsilon scheduler
            num_steps=args.rollout_steps,
            device=device,
        )
        storage = apply_shaped_rewards(raw_storage, reward_shaper, progress=progress)

        # Separate gameplay and trade experiences
        gameplay_transitions = [item for item in storage if item["phase"] == "gameplay"]
        trade_transitions = [item for item in storage if item["phase"] == "trade"]

        # Add gameplay transitions to DQN replay buffer
        for transition in gameplay_transitions:
            replay_buffer.add(
                obs=transition["obs"],
                action=transition["action"],
                reward=transition["reward"],
                next_obs=transition["next_obs"],
                done=transition["done"],
                phase=transition["phase"],
            )

        # Update DQN if we have enough gameplay data
        dqn_metrics = {"td_loss": 0.0, "q_mean": 0.0, "target_q_mean": 0.0}
        if len(replay_buffer) >= args.batch_size:
            batch = replay_buffer.sample(args.batch_size)
            # Convert batch to proper format for DQN trainer
            # Handle nested trade_history structure
            formatted_batch = {
                "obs": {
                    "board": torch.stack([torch.tensor(item["obs"]["board"], dtype=torch.float32) for item in batch]),
                    "self": torch.stack([torch.tensor(item["obs"]["self"], dtype=torch.float32) for item in batch]),
                    "opponent": torch.stack([torch.tensor(item["obs"]["opponent"], dtype=torch.float32) for item in batch]),
                    "gameplay_candidates": torch.stack([torch.tensor(item["obs"]["gameplay_candidates"], dtype=torch.float32) for item in batch]),
                    "gameplay_mask": torch.stack([torch.tensor(item["obs"]["gameplay_mask"], dtype=torch.float32) for item in batch]),
                    "trade_history": {
                        "proposer_ids": torch.stack([torch.tensor(item["obs"]["trade_history"]["proposer_ids"], dtype=torch.long) for item in batch]),
                        "target_ids": torch.stack([torch.tensor(item["obs"]["trade_history"]["target_ids"], dtype=torch.long) for item in batch]),
                        "response_types": torch.stack([torch.tensor(item["obs"]["trade_history"]["response_types"], dtype=torch.long) for item in batch]),
                        "offers": torch.stack([torch.tensor(item["obs"]["trade_history"]["offers"], dtype=torch.float32) for item in batch]),
                        "requests": torch.stack([torch.tensor(item["obs"]["trade_history"]["requests"], dtype=torch.float32) for item in batch]),
                        "accepted_flags": torch.stack([torch.tensor(item["obs"]["trade_history"]["accepted_flags"], dtype=torch.float32) for item in batch]),
                        "turn_numbers": torch.stack([torch.tensor(item["obs"]["trade_history"]["turn_numbers"], dtype=torch.float32) for item in batch]),
                    }
                },
                "actions": {
                    "gameplay_action": torch.tensor([item["action"]["gameplay_action"] if "gameplay_action" in item["action"] else 0 for item in batch], dtype=torch.long),
                    "trade_action": torch.tensor([item["action"]["trade_action"] if "trade_action" in item["action"] else 0 for item in batch], dtype=torch.long)
                },
                "rewards": torch.tensor([item["reward"] for item in batch], dtype=torch.float32),
                "next_obs": {
                    "board": torch.stack([torch.tensor(item["next_obs"]["board"], dtype=torch.float32) for item in batch]),
                    "self": torch.stack([torch.tensor(item["next_obs"]["self"], dtype=torch.float32) for item in batch]),
                    "opponent": torch.stack([torch.tensor(item["next_obs"]["opponent"], dtype=torch.float32) for item in batch]),
                    "gameplay_candidates": torch.stack([torch.tensor(item["next_obs"]["gameplay_candidates"], dtype=torch.float32) for item in batch]),
                    "gameplay_mask": torch.stack([torch.tensor(item["next_obs"]["gameplay_mask"], dtype=torch.float32) for item in batch]),
                    "trade_history": {
                        "proposer_ids": torch.stack([torch.tensor(item["next_obs"]["trade_history"]["proposer_ids"], dtype=torch.long) for item in batch]),
                        "target_ids": torch.stack([torch.tensor(item["next_obs"]["trade_history"]["target_ids"], dtype=torch.long) for item in batch]),
                        "response_types": torch.stack([torch.tensor(item["next_obs"]["trade_history"]["response_types"], dtype=torch.long) for item in batch]),
                        "offers": torch.stack([torch.tensor(item["next_obs"]["trade_history"]["offers"], dtype=torch.float32) for item in batch]),
                        "requests": torch.stack([torch.tensor(item["next_obs"]["trade_history"]["requests"], dtype=torch.float32) for item in batch]),
                        "accepted_flags": torch.stack([torch.tensor(item["next_obs"]["trade_history"]["accepted_flags"], dtype=torch.float32) for item in batch]),
                        "turn_numbers": torch.stack([torch.tensor(item["next_obs"]["trade_history"]["turn_numbers"], dtype=torch.float32) for item in batch]),
                    }
                },
                "dones": torch.tensor([item["done"] for item in batch], dtype=torch.float32),
                "phases": [item["phase"] for item in batch],
            }
            dqn_metrics = dqn_trainer.update(formatted_batch)

        # Update PPO for trade (placeholder - need proper PPO batch processing)
        trade_metrics = {
            "action_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
        }

        metrics = {
            "dqn_loss": dqn_metrics["td_loss"],
            "trade_policy_loss": trade_metrics["action_loss"],
            "trade_value_loss": trade_metrics["value_loss"],
            "trade_entropy": trade_metrics["entropy_loss"],
        }

        stats = compute_rollout_stats(storage)

        gameplay_reward_window.append(stats["gameplay_reward_mean"])
        trade_reward_window.append(stats["trade_reward_mean"])

        avg_gameplay_reward = sum(gameplay_reward_window) / len(gameplay_reward_window)
        avg_trade_reward = sum(trade_reward_window) / len(trade_reward_window)

        trade_score = (
            stats["trade_reward_mean"]
            + 0.020 * stats["trade_propose_rate"]
            + 0.030 * stats["trade_accept_rate"]
            + 0.020 * stats["trade_counter_rate"]
            - 0.025 * stats["trade_skip_rate"]
        )

        print(f"\nUpdate {update}")
        print(
            f"rollouts | gameplay={stats['gameplay_rollouts']} "
            f"trade={stats['trade_rollouts']}"
        )
        print(
            f"reward   | gameplay={stats['gameplay_reward_mean']:.4f} (avg={avg_gameplay_reward:.4f}) "
            f"trade={stats['trade_reward_mean']:.4f} (avg={avg_trade_reward:.4f})"
        )
        print(
            f"trade actions | propose={stats['trade_propose_count']} "
            f"accept={stats['trade_accept_count']} "
            f"reject={stats['trade_reject_count']} "
            f"counter={stats['trade_counter_count']} "
            f"skip={stats['trade_skip_count']}"
        )
        print(
            f"hybrid losses | dqn={metrics['dqn_loss']:.4f} "
            f"trade_policy={metrics['trade_policy_loss']:.4f} "
            f"trade_value={metrics['trade_value_loss']:.4f} "
            f"trade_entropy={metrics['trade_entropy']:.4f} "
            f"trade_score={trade_score:.4f}"
        )

        if stats["trade_reward_mean"] > best_trade_reward:
            best_trade_reward = stats["trade_reward_mean"]
            best_trade_update = update + 1
            best_path = save_checkpoint(dqn_trainer, trade_trainer, args.checkpoint_dir, update + 1)
            print(f"saved best-trade checkpoint -> {best_path}")

        if (update + 1) % 20 == 0:
            path = save_checkpoint(dqn_trainer, trade_trainer, args.checkpoint_dir, update + 1)
            league.maybe_add_checkpoint(path, update + 1)
            print(f"saved checkpoint -> {path}")

    print(f"best trade reward checkpoint update: {best_trade_update} value={best_trade_reward:.6f}")
    print("hybrid training complete")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)