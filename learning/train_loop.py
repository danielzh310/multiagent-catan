from __future__ import annotations

import argparse
import os
import random
from collections import deque
from typing import Dict, List

import torch

from learning.league.league_manager import LeagueManager
from learning.rewards.reward_shaper import RewardShaper
from learning.trade.trade_labeler import build_batch_need_targets
from learning.unified.unified_policy import UnifiedPolicy
from learning.unified.unified_rollout_manager import UnifiedRolloutManager


class UnifiedPPOTrainer:
    def __init__(
        self,
        policy: UnifiedPolicy,
        device: str = "cpu",
        lr: float = 7.5e-5,
        gamma_start: float = 0.94,
        gamma_end: float = 0.995,
        gae_lambda: float = 0.95,
        clip_param: float = 0.10,
        value_clip_param: float = 0.10,
        value_loss_coef: float = 0.25,
        entropy_coef_start: float = 1.2e-3,
        entropy_coef_end: float = 7.5e-4,
        entropy_hold_fraction: float = 0.75,
        gameplay_entropy_floor: float = 0.35,
        trade_entropy_floor: float = 0.75,
        gameplay_entropy_floor_coef: float = 0.010,
        trade_entropy_floor_coef: float = 0.022,
        tom_loss_coef_start: float = 0.02,
        tom_loss_coef_end: float = 0.06,
        max_grad_norm: float = 0.25,
        ppo_epochs: int = 3,
        mini_batch_size: int = 256,
    ):
        self.policy = policy.to(device)
        self.device = device
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.value_loss_coef = value_loss_coef

        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_hold_fraction = entropy_hold_fraction

        self.gameplay_entropy_floor = gameplay_entropy_floor
        self.trade_entropy_floor = trade_entropy_floor
        self.gameplay_entropy_floor_coef = gameplay_entropy_floor_coef
        self.trade_entropy_floor_coef = trade_entropy_floor_coef

        self.tom_loss_coef_start = tom_loss_coef_start
        self.tom_loss_coef_end = tom_loss_coef_end

        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.progress = 0.0

    def set_progress(self, progress: float) -> None:
        self.progress = max(0.0, min(1.0, float(progress)))

    def gamma(self) -> float:
        return self.gamma_start + (self.gamma_end - self.gamma_start) * self.progress

    def entropy_coef(self) -> float:
        if self.progress <= self.entropy_hold_fraction:
            return self.entropy_coef_start

        scaled = (self.progress - self.entropy_hold_fraction) / max(1.0 - self.entropy_hold_fraction, 1e-8)
        scaled = max(0.0, min(1.0, scaled))
        return self.entropy_coef_start + (self.entropy_coef_end - self.entropy_coef_start) * scaled

    def tom_loss_coef(self) -> float:
        return self.tom_loss_coef_start + (self.tom_loss_coef_end - self.tom_loss_coef_start) * self.progress

    def _stack_obs(self, storage: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "board": torch.cat([x["obs"]["board"] for x in storage], dim=0).to(self.device),
            "self": torch.cat([x["obs"]["self"] for x in storage], dim=0).to(self.device),
            "opponent": torch.cat([x["obs"]["opponent"] for x in storage], dim=0).to(self.device),
            "gameplay_candidates": torch.cat([x["obs"]["gameplay_candidates"] for x in storage], dim=0).to(self.device),
            "gameplay_mask": torch.cat([x["obs"]["gameplay_mask"] for x in storage], dim=0).to(self.device),
            "trade_candidates": torch.cat([x["obs"]["trade_candidates"] for x in storage], dim=0).to(self.device),
            "trade_mask": torch.cat([x["obs"]["trade_mask"] for x in storage], dim=0).to(self.device),
        }

    def _compute_returns_advantages(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        next_value = 0.0
        gae = 0.0
        gamma = self.gamma()

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _trade_targets(self, storage: List[Dict], idxs: List[int]):
        trade_events = []
        for i in idxs:
            env_action = storage[i].get("env_action", {})
            response_type = env_action.get("response_type", "")
            offer = env_action.get("offer", None)
            counter_request = env_action.get("counter_request", None)

            if not response_type:
                if storage[i]["reward"] > 0:
                    response_type = "accept"
                elif storage[i]["reward"] < 0:
                    response_type = "reject"
                else:
                    response_type = ""

            trade_events.append(
                {
                    "response_type": response_type,
                    "offer": offer,
                    "counter_request": counter_request,
                }
            )

        need_targets, need_mask = build_batch_need_targets(trade_events)
        return need_targets.to(self.device), need_mask.to(self.device)

    def _phase_batch_tensors(
        self,
        storage: List[Dict],
        obs: Dict[str, torch.Tensor],
        returns: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
        phase_name: str,
    ):
        idxs = [i for i, x in enumerate(storage) if x["phase"] == phase_name]
        if not idxs:
            return None

        sub_obs = {k: v[idxs] for k, v in obs.items()}
        sub_returns = returns[idxs]
        sub_advantages = advantages[idxs]
        sub_old_values = old_values[idxs]

        if phase_name == "gameplay":
            actions = {
                "gameplay_action": torch.cat(
                    [storage[i]["action"]["gameplay_action"] for i in idxs], dim=0
                ).to(self.device)
            }
            old_log_prob = torch.cat(
                [storage[i]["log_prob"]["gameplay_action"] for i in idxs], dim=0
            ).to(self.device)
            need_targets = None
            need_mask = None
        else:
            actions = {
                "trade_action": torch.cat([storage[i]["action"]["trade_action"] for i in idxs], dim=0).to(self.device),
            }
            old_log_prob = torch.cat([storage[i]["log_prob"]["trade_action"] for i in idxs], dim=0).to(self.device)
            need_targets, need_mask = self._trade_targets(storage, idxs)

        return {
            "idxs": idxs,
            "obs": sub_obs,
            "returns": sub_returns,
            "advantages": sub_advantages,
            "old_values": sub_old_values,
            "actions": actions,
            "old_log_prob": old_log_prob,
            "need_targets": need_targets,
            "need_mask": need_mask,
        }

    def _optimize_phase(self, phase_name: str, phase_batch: Dict | None):
        zero_float = 0.0
        if phase_batch is None:
            return {
                "policy_loss": zero_float,
                "value_loss": zero_float,
                "entropy": zero_float,
                "tom_loss": zero_float,
            }

        obs = phase_batch["obs"]
        returns = phase_batch["returns"]
        advantages = phase_batch["advantages"]
        old_values = phase_batch["old_values"]
        actions = phase_batch["actions"]
        old_log_prob = phase_batch["old_log_prob"]
        need_targets = phase_batch["need_targets"]
        need_mask = phase_batch["need_mask"]

        n = returns.shape[0]
        policy_losses = []
        value_losses = []
        entropies = []
        tom_losses = []

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)

            for start in range(0, n, self.mini_batch_size):
                mb = perm[start : start + self.mini_batch_size]

                mb_obs = {k: v[mb] for k, v in obs.items()}
                mb_returns = returns[mb]
                mb_advantages = advantages[mb]
                mb_old_values = old_values[mb]
                mb_old_log_prob = old_log_prob[mb]

                if phase_name == "gameplay":
                    mb_actions = {"gameplay_action": actions["gameplay_action"][mb]}
                else:
                    mb_actions = {"trade_action": actions["trade_action"][mb]}

                log_prob_dict, entropy, new_values, tom_outputs = self.policy.evaluate_actions(
                    obs=mb_obs,
                    actions=mb_actions,
                    phase=phase_name,
                )

                if phase_name == "gameplay":
                    new_log_prob = log_prob_dict["gameplay_action"]
                else:
                    new_log_prob = log_prob_dict["trade_action"]

                ratio = torch.exp(new_log_prob - mb_old_log_prob)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = new_values.squeeze(-1)
                value_pred_clipped = mb_old_values + (value_pred - mb_old_values).clamp(
                    -self.value_clip_param,
                    self.value_clip_param,
                )

                value_loss_unclipped = (value_pred - mb_returns).pow(2)
                value_loss_clipped = (value_pred_clipped - mb_returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                if phase_name == "trade":
                    mb_need_targets = need_targets[mb]
                    mb_need_mask = need_mask[mb]
                    tom_loss = self.policy.need_predictor.compute_loss(
                        tom_outputs["need_pred"],
                        mb_need_targets,
                        mb_need_mask,
                    )
                    entropy_floor_penalty = self.trade_entropy_floor_coef * torch.relu(
                        torch.tensor(self.trade_entropy_floor, device=self.device) - entropy
                    )
                else:
                    tom_loss = torch.tensor(0.0, device=self.device)
                    entropy_floor_penalty = self.gameplay_entropy_floor_coef * torch.relu(
                        torch.tensor(self.gameplay_entropy_floor, device=self.device) - entropy
                    )

                total_mb_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef() * entropy
                    + entropy_floor_penalty
                    + self.tom_loss_coef() * tom_loss
                )

                self.optimizer.zero_grad()
                total_mb_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))
                tom_losses.append(float(tom_loss.item()))

        return {
            "policy_loss": float(sum(policy_losses) / max(len(policy_losses), 1)),
            "value_loss": float(sum(value_losses) / max(len(value_losses), 1)),
            "entropy": float(sum(entropies) / max(len(entropies), 1)),
            "tom_loss": float(sum(tom_losses) / max(len(tom_losses), 1)),
        }

    def update(self, storage: List[Dict]) -> Dict[str, float]:
        if len(storage) == 0:
            return {
                "gameplay_policy_loss": 0.0,
                "gameplay_value_loss": 0.0,
                "gameplay_entropy": 0.0,
                "trade_policy_loss": 0.0,
                "trade_value_loss": 0.0,
                "trade_entropy": 0.0,
                "tom_loss": 0.0,
                "total_loss": 0.0,
                "entropy_coef": self.entropy_coef(),
                "gamma": self.gamma(),
                "tom_loss_coef": self.tom_loss_coef(),
            }

        obs = self._stack_obs(storage)
        rewards = torch.tensor([x["reward"] for x in storage], dtype=torch.float32, device=self.device)
        old_values = torch.cat([x["value"] for x in storage], dim=0).squeeze(-1).to(self.device)
        dones = torch.tensor([x["done"] for x in storage], dtype=torch.float32, device=self.device)

        returns, advantages = self._compute_returns_advantages(rewards, old_values, dones)

        gp_batch = self._phase_batch_tensors(storage, obs, returns, advantages, old_values, "gameplay")
        tr_batch = self._phase_batch_tensors(storage, obs, returns, advantages, old_values, "trade")

        gp = self._optimize_phase("gameplay", gp_batch)
        tr = self._optimize_phase("trade", tr_batch)

        total_loss = (
            gp["policy_loss"]
            + tr["policy_loss"]
            + self.value_loss_coef * (gp["value_loss"] + tr["value_loss"])
            - self.entropy_coef() * 0.5 * (gp["entropy"] + tr["entropy"])
            + self.tom_loss_coef() * tr["tom_loss"]
        )

        return {
            "gameplay_policy_loss": gp["policy_loss"],
            "gameplay_value_loss": gp["value_loss"],
            "gameplay_entropy": gp["entropy"],
            "trade_policy_loss": tr["policy_loss"],
            "trade_value_loss": tr["value_loss"],
            "trade_entropy": tr["entropy"],
            "tom_loss": tr["tom_loss"],
            "total_loss": float(total_loss),
            "entropy_coef": float(self.entropy_coef()),
            "gamma": float(self.gamma()),
            "tom_loss_coef": float(self.tom_loss_coef()),
        }


def save_checkpoint(policy: UnifiedPolicy, save_dir: str, step: int, prefix: str = "unified_checkpoint") -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_{step}.pt")
    torch.save({"policy": policy.state_dict(), "step": step}, path)
    return path


def compute_rollout_stats(storage: List[Dict]) -> Dict[str, float]:
    gameplay_rewards = [x["reward"] for x in storage if x["phase"] == "gameplay"]
    trade_rewards = [x["reward"] for x in storage if x["phase"] == "trade"]

    counts = {
        "propose": 0,
        "accept": 0,
        "reject": 0,
        "counter": 0,
        "skip": 0,
    }

    for x in storage:
        if x["phase"] != "trade":
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

        if item["phase"] == "trade":
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
    parser = argparse.ArgumentParser(description="Unified PPO training loop")
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=192)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    return parser


def train(args: argparse.Namespace) -> None:
    device = args.device

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    policy = UnifiedPolicy(hidden_dim=args.hidden_dim).to(device)
    trainer = UnifiedPPOTrainer(policy=policy, device=device)
    rollout_manager = UnifiedRolloutManager(num_envs=args.num_envs, device=device)
    league = LeagueManager(checkpoint_dir=args.checkpoint_dir, frozen_ratio=0.2)
    reward_shaper = RewardShaper()

    gameplay_reward_window = deque(maxlen=20)
    trade_reward_window = deque(maxlen=20)
    gameplay_entropy_window = deque(maxlen=20)
    trade_entropy_window = deque(maxlen=20)

    best_trade_reward = float("-inf")
    best_trade_update = -1
    best_trade_score = float("-inf")
    best_trade_score_update = -1

    print(
        f"Starting unified PPO training for {args.num_updates} updates "
        f"(envs={args.num_envs}, rollout_steps={args.rollout_steps}, hidden_dim={args.hidden_dim}, seed={args.seed})"
    )

    for update in range(args.num_updates):
        progress = update / float(max(args.num_updates - 1, 1))
        trainer.set_progress(progress)

        raw_storage = rollout_manager.collect(policy=policy, steps=args.rollout_steps)
        storage = apply_shaped_rewards(raw_storage, reward_shaper, progress=progress)

        stats = compute_rollout_stats(storage)
        metrics = trainer.update(storage)

        gameplay_reward_window.append(stats["gameplay_reward_mean"])
        trade_reward_window.append(stats["trade_reward_mean"])
        gameplay_entropy_window.append(metrics["gameplay_entropy"])
        trade_entropy_window.append(metrics["trade_entropy"])

        avg_gameplay_reward = sum(gameplay_reward_window) / len(gameplay_reward_window)
        avg_trade_reward = sum(trade_reward_window) / len(trade_reward_window)
        avg_gameplay_entropy = sum(gameplay_entropy_window) / len(gameplay_entropy_window)
        avg_trade_entropy = sum(trade_entropy_window) / len(trade_entropy_window)

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
            f"trade rates   | propose={stats['trade_propose_rate']:.3f} "
            f"accept={stats['trade_accept_rate']:.3f} "
            f"reject={stats['trade_reject_rate']:.3f} "
            f"counter={stats['trade_counter_rate']:.3f} "
            f"skip={stats['trade_skip_rate']:.3f}"
        )
        print(
            f"ppo gameplay | policy={metrics['gameplay_policy_loss']:.4f} "
            f"value={metrics['gameplay_value_loss']:.4f} "
            f"entropy={metrics['gameplay_entropy']:.4f} (avg={avg_gameplay_entropy:.4f})"
        )
        print(
            f"ppo trade    | policy={metrics['trade_policy_loss']:.4f} "
            f"value={metrics['trade_value_loss']:.4f} "
            f"entropy={metrics['trade_entropy']:.4f} (avg={avg_trade_entropy:.4f}) "
            f"tom={metrics['tom_loss']:.6f}"
        )
        print(
            f"ppo total    | loss={metrics['total_loss']:.4f} "
            f"entropy_coef={metrics['entropy_coef']:.6f} "
            f"gamma={metrics['gamma']:.5f} "
            f"tom_coef={metrics['tom_loss_coef']:.5f} "
            f"trade_score={trade_score:.4f}"
        )

        if stats["trade_reward_mean"] > best_trade_reward:
            best_trade_reward = stats["trade_reward_mean"]
            best_trade_update = update + 1
            best_path = save_checkpoint(policy, args.checkpoint_dir, update + 1, prefix="unified_best_trade")
            print(f"saved best-trade checkpoint -> {best_path}")

        if trade_score > best_trade_score:
            best_trade_score = trade_score
            best_trade_score_update = update + 1
            score_path = save_checkpoint(policy, args.checkpoint_dir, update + 1, prefix="unified_best_score")
            print(f"saved best-score checkpoint -> {score_path}")

        if (update + 1) % 20 == 0:
            path = save_checkpoint(policy, args.checkpoint_dir, update + 1)
            league.maybe_add_checkpoint(path, update + 1)
            print(f"saved checkpoint -> {path}")

    print(f"best trade reward checkpoint update: {best_trade_update} value={best_trade_reward:.6f}")
    print(f"best trade score checkpoint update: {best_trade_score_update} value={best_trade_score:.6f}")
    print("training complete")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)
