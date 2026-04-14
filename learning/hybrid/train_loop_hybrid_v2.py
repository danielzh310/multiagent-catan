from __future__ import annotations

import argparse
import random
from collections import deque
from typing import Dict, List

import torch

from learning.dqn.epsilon_scheduler import EpsilonScheduler
from learning.dqn.replay_buffer import ReplayBuffer
from learning.gameplay.build_gameplay_q_model import build_gameplay_q_model
from learning.hybrid.hybrid_checkpoint import HybridCheckpointManager
from learning.hybrid.hybrid_rollout_manager_v2 import HybridRolloutManagerV2
from learning.rewards.reward_shaper import RewardShaper
from learning.trade.build_trade_model import build_trade_model
from learning.trade.trade_labeler import build_batch_need_targets


class TradePPOTrainerV2:
    def __init__(
        self,
        policy,
        device: str = "cpu",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef_start: float = 5e-4,
        entropy_coef_end: float = 5e-5,
        tom_loss_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 128,
    ):
        self.policy = policy.to(device)
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.tom_loss_coef = tom_loss_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.progress = 0.0

    def set_progress(self, progress: float) -> None:
        self.progress = max(0.0, min(1.0, float(progress)))

    def entropy_coef(self) -> float:
        return self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * self.progress

    def _compute_returns_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        next_value = 0.0
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _build_tom_targets(self, rollouts: List[dict]):
        events = []
        for item in rollouts:
            env_action = item.get("env_action", {})
            events.append(
                {
                    "response_type": env_action.get("response_type", ""),
                    "offer": env_action.get("offer"),
                    "counter_request": env_action.get("counter_request"),
                }
            )
        need_targets, need_mask = build_batch_need_targets(events)
        acceptance_target = (need_mask > 0.0).float()
        resource_pref_target = torch.argmax(need_targets, dim=-1)
        return {
            "acceptance_target": acceptance_target,
            "resource_pref_target": resource_pref_target,
            "need_mask": need_mask,
        }

    def update(self, rollouts: List[dict]) -> Dict[str, float]:
        if not rollouts:
            return {
                "trade_policy_loss": 0.0,
                "trade_value_loss": 0.0,
                "trade_entropy": 0.0,
                "trade_tom_loss": 0.0,
                "trade_total_loss": 0.0,
                "trade_entropy_coef": self.entropy_coef(),
            }

        obs = {
            "board": torch.cat([item["obs"]["board"] for item in rollouts], dim=0).to(self.device),
            "self": torch.cat([item["obs"]["self"] for item in rollouts], dim=0).to(self.device),
            "opponent": torch.cat([item["obs"]["opponent"] for item in rollouts], dim=0).to(self.device),
            "trade_history": {
                key: torch.cat([item["obs"]["trade_history"][key] for item in rollouts], dim=0).to(self.device)
                for key in rollouts[0]["obs"]["trade_history"]
            },
        }
        action_masks = {
            key: torch.cat([item["action_masks"][key] for item in rollouts], dim=0).to(self.device)
            for key in rollouts[0]["action_masks"]
        }
        actions = {
            key: torch.cat([item["action"][key] for item in rollouts], dim=0).to(self.device)
            for key in rollouts[0]["action"]
        }
        rewards = torch.tensor([item["reward"] for item in rollouts], dtype=torch.float32, device=self.device)
        values = torch.cat([item["value"] for item in rollouts], dim=0).squeeze(-1).to(self.device)
        old_log_probs = torch.stack(
            [
                item["log_prob"]["action_type"].view(-1).mean()
                + item["log_prob"]["target"].view(-1).mean()
                + item["log_prob"]["offer"].view(-1).mean()
                + item["log_prob"]["request"].view(-1).mean()
                for item in rollouts
            ]
        ).to(self.device)
        dones = torch.tensor([item["done"] for item in rollouts], dtype=torch.float32, device=self.device)

        returns, advantages = self._compute_returns_advantages(rewards, values, dones)
        tom_targets = self._build_tom_targets(rollouts)
        tom_targets = {key: value.to(self.device) for key, value in tom_targets.items()}

        policy_losses = []
        value_losses = []
        entropies = []
        tom_losses = []

        n = rewards.shape[0]
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.mini_batch_size):
                mb = perm[start : start + self.mini_batch_size]
                mb_obs = {
                    "board": obs["board"][mb],
                    "self": obs["self"][mb],
                    "opponent": obs["opponent"][mb],
                    "trade_history": {key: value[mb] for key, value in obs["trade_history"].items()},
                }
                mb_masks = {key: value[mb] for key, value in action_masks.items()}
                mb_actions = {key: value[mb] for key, value in actions.items()}

                values_pred, log_prob_dict, entropy, _ = self.policy.evaluate_actions(
                    mb_obs,
                    mb_actions,
                    action_masks=mb_masks,
                    tom_targets=None,
                )
                new_log_probs = (
                    log_prob_dict["action_type"]
                    + log_prob_dict["target"]
                    + log_prob_dict["offer"].mean(dim=-1)
                    + log_prob_dict["request"].mean(dim=-1)
                )

                ratio = torch.exp(new_log_probs - old_log_probs[mb])
                surr1 = ratio * advantages[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages[mb]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (values_pred.squeeze(-1) - returns[mb]).pow(2).mean()
                tom_loss = self.policy.tom_head.compute_loss(
                    self.policy.forward(mb_obs)[2],
                    {
                        "acceptance_target": tom_targets["acceptance_target"][mb],
                        "resource_pref_target": tom_targets["resource_pref_target"][mb],
                    },
                )

                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef() * entropy
                    + self.tom_loss_coef * tom_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item() if torch.is_tensor(entropy) else entropy))
                tom_losses.append(float(tom_loss.item()))

        return {
            "trade_policy_loss": sum(policy_losses) / max(len(policy_losses), 1),
            "trade_value_loss": sum(value_losses) / max(len(value_losses), 1),
            "trade_entropy": sum(entropies) / max(len(entropies), 1),
            "trade_tom_loss": sum(tom_losses) / max(len(tom_losses), 1),
            "trade_total_loss": (
                (sum(policy_losses) / max(len(policy_losses), 1))
                + self.value_loss_coef * (sum(value_losses) / max(len(value_losses), 1))
                - self.entropy_coef() * (sum(entropies) / max(len(entropies), 1))
                + self.tom_loss_coef * (sum(tom_losses) / max(len(tom_losses), 1))
            ),
            "trade_entropy_coef": self.entropy_coef(),
        }


def apply_trade_shaping(rollouts: List[dict], reward_shaper: RewardShaper, progress: float) -> List[dict]:
    shaped = []
    consecutive_skips = 0
    curriculum_scale = 0.35 + 0.65 * progress
    for item in rollouts:
        new_item = dict(item)
        action_type = item.get("env_action", {}).get("type", "skip_trade")
        if action_type == "skip_trade":
            consecutive_skips += 1
        else:
            consecutive_skips = 0
        new_item["reward"] = reward_shaper.trade_step_reward(
            action_type=action_type,
            reward_signal=float(item["reward"]),
            consecutive_skips=consecutive_skips,
            tom_loss=0.0,
            curriculum_scale=curriculum_scale,
        )
        shaped.append(new_item)
    return shaped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid v2 DQN gameplay + PPO trade training")
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    return parser


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dqn_trainer = build_gameplay_q_model(
        {
            "state_dim": 64,
            "num_actions": 256,
            "hidden_dim": 256,
            "lr": 1e-3,
            "gamma": 0.99,
            "target_update_freq": 1000,
            "device": args.device,
        }
    )
    trade_policy = build_trade_model(device=args.device)
    trade_trainer = TradePPOTrainerV2(policy=trade_policy, device=args.device)
    rollout_manager = HybridRolloutManagerV2(num_envs=args.num_envs, device=args.device, seed=args.seed)
    replay_buffer = ReplayBuffer(capacity=100_000, device=args.device, seed=args.seed)
    epsilon_scheduler = EpsilonScheduler(start_epsilon=1.0, end_epsilon=0.05, decay_steps=100_000)
    checkpoint_manager = HybridCheckpointManager(checkpoint_dir=args.checkpoint_dir)
    reward_shaper = RewardShaper()

    gameplay_window = deque(maxlen=20)
    trade_window = deque(maxlen=20)
    total_env_steps = 0

    print(
        f"Starting hybrid v2 training for {args.num_updates} updates "
        f"(envs={args.num_envs}, rollout_steps={args.rollout_steps}, seed={args.seed})"
    )

    for update in range(args.num_updates):
        progress = update / float(max(args.num_updates - 1, 1))
        epsilon = epsilon_scheduler.value(total_env_steps)
        trade_trainer.set_progress(progress)

        gameplay_transitions, trade_rollouts, stats = rollout_manager.collect(
            dqn_trainer=dqn_trainer,
            epsilon=epsilon,
            trade_policy=trade_policy,
            steps=args.rollout_steps,
        )
        total_env_steps += args.num_envs * args.rollout_steps

        for item in gameplay_transitions:
            replay_buffer.add(
                obs=item["obs"],
                action=item["action"],
                reward=item["reward"],
                next_obs=item["next_obs"],
                done=item["done"],
                action_mask=item["action_mask"],
                next_action_mask=item["next_action_mask"],
            )

        dqn_metrics = {"td_loss": 0.0, "q_mean": 0.0, "target_q_mean": 0.0}
        if len(replay_buffer) >= 512:
            for _ in range(4):
                dqn_metrics = dqn_trainer.update(replay_buffer.sample(256))

        trade_metrics = trade_trainer.update(apply_trade_shaping(trade_rollouts, reward_shaper, progress))

        gameplay_window.append(stats["gameplay_reward_mean"])
        trade_window.append(stats["trade_reward_mean"])

        print(f"\nUpdate {update}")
        print(
            f"rollouts | gameplay={stats['gameplay_rollouts']} trade={stats['trade_rollouts']} "
            f"epsilon={epsilon:.4f} replay={len(replay_buffer)}"
        )
        print(
            f"reward   | gameplay={stats['gameplay_reward_mean']:.4f} (avg={sum(gameplay_window)/len(gameplay_window):.4f}) "
            f"trade={stats['trade_reward_mean']:.4f} (avg={sum(trade_window)/len(trade_window):.4f})"
        )
        print(
            f"trade actions | propose={stats['trade_propose_count']} accept={stats['trade_accept_count']} "
            f"reject={stats['trade_reject_count']} counter={stats['trade_counter_count']} skip={stats['trade_skip_count']}"
        )
        print(
            f"dqn gameplay | td={dqn_metrics['td_loss']:.4f} q={dqn_metrics['q_mean']:.4f} target_q={dqn_metrics['target_q_mean']:.4f}"
        )
        print(
            f"ppo trade    | policy={trade_metrics['trade_policy_loss']:.4f} value={trade_metrics['trade_value_loss']:.4f} "
            f"entropy={trade_metrics['trade_entropy']:.4f} tom={trade_metrics['trade_tom_loss']:.4f}"
        )

        if (update + 1) % 20 == 0:
            path = checkpoint_manager.save(update + 1, dqn_trainer, trade_trainer)
            print(f"saved checkpoint -> {path}")

    print("hybrid v2 training complete")


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
