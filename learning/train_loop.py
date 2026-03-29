from __future__ import annotations

import argparse
import os
from collections import defaultdict, deque
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
        lr: float = 3e-4,
        gamma_start: float = 0.92,
        gamma_end: float = 0.995,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef_start: float = 8e-4,
        entropy_coef_end: float = 2.5e-4,
        entropy_hold_fraction: float = 0.35,
        tom_loss_coef: float = 0.08,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy.to(device)
        self.device = device
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.entropy_hold_fraction = entropy_hold_fraction
        self.tom_loss_coef = tom_loss_coef
        self.max_grad_norm = max_grad_norm
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

    def _stack_obs(self, storage: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "board": torch.cat([x["obs"]["board"] for x in storage], dim=0).to(self.device),
            "self": torch.cat([x["obs"]["self"] for x in storage], dim=0).to(self.device),
            "opponent": torch.cat([x["obs"]["opponent"] for x in storage], dim=0).to(self.device),
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

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def _phase_metrics(
        self,
        storage: List[Dict],
        obs: Dict[str, torch.Tensor],
        returns: torch.Tensor,
        advantages: torch.Tensor,
        phase_name: str,
    ) -> Dict[str, torch.Tensor] | None:
        idxs = [i for i, x in enumerate(storage) if x["phase"] == phase_name]
        if not idxs:
            return None

        sub_obs = {k: v[idxs] for k, v in obs.items()}
        sub_returns = returns[idxs]
        sub_advantages = advantages[idxs]

        if phase_name == "gameplay":
            actions = {
                "gameplay_action": torch.cat(
                    [storage[i]["action"]["gameplay_action"] for i in idxs], dim=0
                ).to(self.device)
            }
            old_log_prob = torch.cat(
                [storage[i]["log_prob"]["gameplay_action"] for i in idxs], dim=0
            ).to(self.device)
        else:
            actions = {
                "action_type": torch.cat([storage[i]["action"]["action_type"] for i in idxs], dim=0).to(self.device),
                "target": torch.cat([storage[i]["action"]["target"] for i in idxs], dim=0).to(self.device),
                "offer": torch.cat([storage[i]["action"]["offer"] for i in idxs], dim=0).to(self.device),
                "request": torch.cat([storage[i]["action"]["request"] for i in idxs], dim=0).to(self.device),
            }
            old_log_prob = (
                torch.cat([storage[i]["log_prob"]["action_type"] for i in idxs], dim=0).to(self.device)
                + torch.cat([storage[i]["log_prob"]["target"] for i in idxs], dim=0).to(self.device)
                + torch.cat([storage[i]["log_prob"]["offer"] for i in idxs], dim=0).to(self.device)
                + torch.cat([storage[i]["log_prob"]["request"] for i in idxs], dim=0).to(self.device)
            )

        log_prob_dict, entropy, new_values, tom_outputs = self.policy.evaluate_actions(
            obs=sub_obs,
            actions=actions,
            phase=phase_name,
        )

        if phase_name == "gameplay":
            new_log_prob = log_prob_dict["gameplay_action"]
        else:
            new_log_prob = (
                log_prob_dict["action_type"]
                + log_prob_dict["target"]
                + log_prob_dict["offer"]
                + log_prob_dict["request"]
            )

        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * sub_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * sub_advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = torch.nn.functional.smooth_l1_loss(
            new_values.squeeze(-1),
            sub_returns,
        )

        if phase_name == "trade":
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
            need_targets = need_targets.to(self.device)
            need_mask = need_mask.to(self.device)
            tom_loss = self.policy.need_predictor.compute_loss(
                tom_outputs["need_pred"],
                need_targets,
                need_mask,
            )
        else:
            tom_loss = torch.tensor(0.0, device=self.device)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "tom_loss": tom_loss,
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
            }

        obs = self._stack_obs(storage)
        rewards = torch.tensor([x["reward"] for x in storage], dtype=torch.float32, device=self.device)
        values = torch.cat([x["value"] for x in storage], dim=0).squeeze(-1).to(self.device)
        dones = torch.tensor([x["done"] for x in storage], dtype=torch.float32, device=self.device)

        returns, advantages = self._compute_returns_advantages(rewards, values, dones)

        gp = self._phase_metrics(storage, obs, returns, advantages, "gameplay")
        tr = self._phase_metrics(storage, obs, returns, advantages, "trade")

        zero = torch.tensor(0.0, device=self.device)

        gp_policy_loss = gp["policy_loss"] if gp is not None else zero
        gp_value_loss = gp["value_loss"] if gp is not None else zero
        gp_entropy = gp["entropy"] if gp is not None else zero

        tr_policy_loss = tr["policy_loss"] if tr is not None else zero
        tr_value_loss = tr["value_loss"] if tr is not None else zero
        tr_entropy = tr["entropy"] if tr is not None else zero
        tom_loss = tr["tom_loss"] if tr is not None else zero

        total_policy_loss = gp_policy_loss + tr_policy_loss
        total_value_loss = gp_value_loss + tr_value_loss
        total_entropy = 0.5 * (gp_entropy + tr_entropy)

        total_loss = (
            total_policy_loss
            + self.value_loss_coef * total_value_loss
            - self.entropy_coef() * total_entropy
            + self.tom_loss_coef * tom_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "gameplay_policy_loss": float(gp_policy_loss.item()),
            "gameplay_value_loss": float(gp_value_loss.item()),
            "gameplay_entropy": float(gp_entropy.item()),
            "trade_policy_loss": float(tr_policy_loss.item()),
            "trade_value_loss": float(tr_value_loss.item()),
            "trade_entropy": float(tr_entropy.item()),
            "tom_loss": float(tom_loss.item()),
            "total_loss": float(total_loss.item()),
            "entropy_coef": float(self.entropy_coef()),
            "gamma": float(self.gamma()),
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
) -> List[Dict]:
    shaped_storage = []
    consecutive_skips = 0

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
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    return parser


def train(args: argparse.Namespace) -> None:
    device = args.device

    policy = UnifiedPolicy().to(device)
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

    print(f"Starting unified PPO training for {args.num_updates} updates")

    for update in range(args.num_updates):
        progress = update / float(max(args.num_updates - 1, 1))
        trainer.set_progress(progress)

        raw_storage = rollout_manager.collect(policy=policy, steps=args.rollout_steps)
        storage = apply_shaped_rewards(raw_storage, reward_shaper)

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
            f"gamma={metrics['gamma']:.5f}"
        )

        if stats["trade_reward_mean"] > best_trade_reward:
            best_trade_reward = stats["trade_reward_mean"]
            best_trade_update = update + 1
            best_path = save_checkpoint(policy, args.checkpoint_dir, update + 1, prefix="unified_best_trade")
            print(f"saved best-trade checkpoint -> {best_path}")

        if (update + 1) % 20 == 0:
            path = save_checkpoint(policy, args.checkpoint_dir, update + 1)
            league.maybe_add_checkpoint(path, update + 1)
            print(f"saved checkpoint -> {path}")

    print(f"best trade reward checkpoint update: {best_trade_update} value={best_trade_reward:.6f}")
    print("training complete")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)