#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import numpy as np
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.constants import PlayerId
from learning.league.league_manager import LeagueManager
from environment.catan_env import CatanEnv
from learning.tom_dqn.tom_dqn_policy import ToMEnhancedDQNPolicy
from learning.tom_dqn.tom_dqn_trainer import ToMEnhancedDQNTrainer
from learning.tom_dqn.tom_dqn_rollout_manager import ToMEnhancedDQNRolloutManager, _worker_build_obs


class EpsilonScheduler:
    def __init__(self, start: float = 1.0, end: float = 0.01, decay_steps: int = 10000):
        self.start = start
        self.end = end
        self.decay_steps = decay_steps
        self.step_count = 0

    def get_epsilon(self) -> float:
        if self.step_count >= self.decay_steps:
            return self.end
        return self.start + (self.end - self.start) * (self.step_count / self.decay_steps)

    def step(self):
        self.step_count += 1


def train_tom_dqn(
    num_envs: int = 8,
    num_workers: int = 4,
    total_steps: int = 100000,
    batch_size: int = 64,
    buffer_size: int = 10000,
    lr: float = 1e-3,
    gamma: float = 0.99,
    tau: float = 0.005,
    tom_loss_coef: float = 0.1,
    epsilon_decay_steps: int = 50000,
    update_freq: int = 4,
    eval_freq: int = 1000,
    save_freq: int = 10000,
    device: str = "auto",
    enable_trading: bool = True,
    max_steps_per_game: int = 1000,
    checkpoint_dir: str = "checkpoints/tom_dqn",
    resume_from: str = None,
):
    """Train ToM-enhanced DQN agent."""

    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize components
    policy = ToMEnhancedDQNPolicy(device=device)
    target_policy = ToMEnhancedDQNPolicy(device=device)
    target_policy.load_state_dict(policy.state_dict())
    target_policy.eval()

    trainer = ToMEnhancedDQNTrainer(
        policy=policy,
        target_policy=target_policy,
        lr=lr,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        device=device,
        tom_loss_coef=tom_loss_coef,
    )

    rollout_manager = ToMEnhancedDQNRolloutManager(
        num_envs=num_envs,
        num_workers=num_workers,
        device=device,
        enable_trading=enable_trading,
        max_steps=max_steps_per_game,
        opponent_paths=[],  # Start with empty opponents
    )
    epsilon_scheduler = EpsilonScheduler(decay_steps=epsilon_decay_steps)
    load_existing = bool(resume_from)  # Load existing checkpoints only if resuming
    league = LeagueManager(checkpoint_dir=checkpoint_dir, frozen_ratio=0.2, load_existing=load_existing)

    # Resume from checkpoint if specified
    start_step = 0
    if resume_from:
        checkpoint_path = os.path.join(checkpoint_dir, resume_from)
        if os.path.exists(checkpoint_path):
            trainer.load(checkpoint_path)
            start_step = int(resume_from.split('_')[-1].split('.')[0])
            print(f"Resumed from step {start_step}")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting from scratch")

    # Training loop
    step_count = start_step

    try:
        while step_count < total_steps:
            # --- LEAGUE-BASED SELF-PLAY ---
            # Periodically sample new opponents from the league
            if step_count % eval_freq == 0:
                opponent_paths = league.sample_opponents(k=3, policy_type="tom_dqn") if len(league) > 0 else []
                rollout_manager.update_opponents(opponent_paths)
            # Collect transitions in a batched and phase-aware manner
            all_states = rollout_manager._vec_env.get_full_state()
            epsilon = epsilon_scheduler.get_epsilon()

            gameplay_envs, trade_envs, auto_envs = [], [], []
            gameplay_obs_list, trade_obs_list = [], []

            # Store original observations for the replay buffer
            obs_tensors_for_buffer = [None] * num_envs

            for env_idx, state in all_states.items():
                phase, legal_actions, obs_raw, player_stats, pending_info = state
                phase_name = rollout_manager._phase_name(phase)

                if phase_name == "auto":
                    auto_envs.append(env_idx)
                    continue

                obs_np = _worker_build_obs(obs_raw, legal_actions, player_stats, pending_info, phase)
                obs_tensors_for_buffer[env_idx] = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs_np.items()}

                if phase_name == "gameplay":
                    gameplay_envs.append(env_idx)
                    gameplay_obs_list.append(obs_np)
                elif phase_name == "trade":
                    trade_envs.append(env_idx)
                    trade_obs_list.append(obs_np)

            actions_to_step = {}
            all_action_dicts_for_buffer = {}

            # Process gameplay actions in a batch
            if gameplay_envs:
                obs_batch = {k: torch.from_numpy(np.stack([o[k] for o in gameplay_obs_list])).to(device) for k in gameplay_obs_list[0]}
                action_dicts = policy.act(obs_batch, "gameplay", epsilon)
                action_indices = action_dicts["gameplay_action"].cpu().numpy()
                for i, env_idx in enumerate(gameplay_envs):
                    all_action_dicts_for_buffer[env_idx] = {k: v[i:i+1] for k, v in action_dicts.items()}
                    phase, legal_actions, _, _, _ = all_states[env_idx]
                    actions_to_step[env_idx] = rollout_manager._decode_gameplay(action_indices[i], legal_actions, phase)

            # Process trade actions in a batch
            if trade_envs:
                obs_batch = {k: torch.from_numpy(np.stack([o[k] for o in trade_obs_list])).to(device) for k in trade_obs_list[0]}
                action_dicts = policy.act(obs_batch, "trade", epsilon)
                action_indices = action_dicts["trade_action"].cpu().numpy()
                for i, env_idx in enumerate(trade_envs):
                    all_action_dicts_for_buffer[env_idx] = {k: v[i:i+1] for k, v in action_dicts.items()}
                    phase, legal_actions, _, _, _ = all_states[env_idx]
                    actions_to_step[env_idx] = rollout_manager._decode_trade(action_indices[i], legal_actions, phase)

            # Prepare actions for auto-phase environments
            for env_idx in auto_envs:
                actions_to_step[env_idx] = None

            # Step all environments and store transitions
            results = rollout_manager._vec_env.step(actions_to_step)
            next_all_states = rollout_manager._vec_env.get_full_state()
            for env_idx, (_, reward, done, info) in results.items():
                if env_idx in auto_envs:
                    continue
                obs_tensor = obs_tensors_for_buffer[env_idx]
                action_dict = all_action_dicts_for_buffer[env_idx]
                phase_name = "gameplay" if env_idx in gameplay_envs else "trade"
                if done:
                    next_obs_tensor = {k: torch.zeros_like(v) for k, v in obs_tensor.items()}
                else:
                    phase, _, next_obs_raw, _, _ = next_all_states[env_idx]
                    next_obs_np = _worker_build_obs(next_obs_raw, [], {}, None, phase)
                    next_obs_tensor = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in next_obs_np.items()}
                trainer.store_transition(obs_tensor, action_dict, reward, next_obs_tensor, done, phase_name)

            # Update
            if step_count > 0 and step_count % update_freq == 0:
                metrics = trainer.update()

            epsilon_scheduler.step()
            step_count += 1

            # Evaluation
            if step_count > 0 and step_count % eval_freq == 0:
                eval_metrics = evaluate_policy(policy, device, enable_trading, max_steps_per_game)
                print(f"Step {step_count}: Eval - Score: {eval_metrics['avg_score']:.2f}, "
                      f"Win Rate: {eval_metrics['win_rate']:.2f}, Steps: {eval_metrics['avg_steps']:.1f}")

            # Save checkpoint
            if step_count > 0 and step_count % save_freq == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"tom_dqn_step_{step_count}.pt")
                trainer.save(checkpoint_path, step_count)
                print(f"Saved checkpoint: {checkpoint_path}")
                league.maybe_add_checkpoint(checkpoint_path, step_count)

            # Log progress
            if step_count > 0 and step_count % 1000 == 0:
                print(f"Step {step_count}/{total_steps} completed")

    except KeyboardInterrupt:
        print("Training interrupted by user")

    finally:
        # Save final checkpoint
        final_checkpoint = os.path.join(checkpoint_dir, f"tom_dqn_final_{step_count}.pt")
        trainer.save(final_checkpoint, step_count)
        print(f"Saved final checkpoint: {final_checkpoint}")

        if rollout_manager:
            rollout_manager.close()


def evaluate_policy(
    policy: nn.Module,
    device: str,
    enable_trading: bool,
    max_steps: int,
    num_games: int = 10,
) -> Dict[str, float]:
    """Evaluate the policy on a set of games."""
    policy.eval()

    total_scores = []
    wins = 0
    total_steps = []

    for _ in range(num_games):
        env = CatanEnv(enable_trading=enable_trading, max_steps=max_steps)
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            # Build observation
            obs_raw = obs
            legal_actions = env.get_legal_actions()
            phase = env.get_phase()
            current_player_id = env.get_current_player_id()
            player_engine_state = env.engine.players[current_player_id]
            player_stats = {
                "vp": float(player_engine_state.update_victory_points()),
                "n_settlements": float(player_engine_state.n_settlements),
                "n_cities": float(player_engine_state.n_cities),
                "n_roads": float(player_engine_state.n_roads),
                "resource_total": float(sum(int(v) for v in player_engine_state.resources.values())),
            }
            pending = env.engine.trade_manager.get_pending_trade()
            pending_info = None
            if pending is not None:
                pending_info = {
                    "counter_count": float(pending.counter_count),
                    "proposer": float(int(pending.proposer)),
                    "target": float(int(pending.target)),
                }
            obs_np = _worker_build_obs(obs_raw, legal_actions, player_stats, pending_info, phase)
            obs_tensor = {k: torch.from_numpy(v).unsqueeze(0).to(device) for k, v in obs_np.items()}

            phase_name = (
                "gameplay"
                if phase.name in ("SETUP", "MAIN_ACTION", "END_TURN")
                else "trade"
                if phase.name in ("TRADE_PROPOSE", "TRADE_RESPOND")
                else "auto"
            )

            if phase_name == "auto":
                action = None
            else:
                with torch.no_grad():
                    epsilon = 0.0  # Greedy evaluation
                    action_dict = policy.act(obs_tensor, phase_name, epsilon)
                    action_key = "gameplay_action" if phase_name == "gameplay" else "trade_action"
                    action_idx = int(action_dict[action_key].item())

                legal_actions = env.get_legal_actions()
                if legal_actions:
                    action_idx = min(action_idx, len(legal_actions) - 1)
                    action = legal_actions[action_idx]
                else:
                    action = {"type": "end_turn"} if phase_name == "gameplay" else {"type": "skip_trade"}

            obs, reward, done, info = env.step(action)
            steps += 1

        # Game is done, record stats
        if info and info.get("winner") == PlayerId.WHITE:
            wins += 1

        agent_score = env.engine.players[PlayerId.WHITE].update_victory_points()
        total_scores.append(agent_score)
        total_steps.append(steps)

    return {
        "avg_score": sum(total_scores) / len(total_scores) if total_scores else 0,
        "win_rate": wins / num_games,
        "avg_steps": sum(total_steps) / len(total_steps) if total_steps else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ToM-enhanced DQN for Catan")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--total_steps", type=int, default=100000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update parameter")
    parser.add_argument("--tom_loss_coef", type=float, default=0.1, help="ToM loss coefficient")
    parser.add_argument("--epsilon_decay_steps", type=int, default=50000, help="Epsilon decay steps")
    parser.add_argument("--update_freq", type=int, default=4, help="Update frequency")
    parser.add_argument("--eval_freq", type=int, default=1000, help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=10000, help="Checkpoint save frequency")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument("--enable_trading", action="store_true", default=True, help="Enable trading")
    parser.add_argument("--max_steps_per_game", type=int, default=2000, help="Max steps per game")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/tom_dqn", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    train_tom_dqn(**vars(args))