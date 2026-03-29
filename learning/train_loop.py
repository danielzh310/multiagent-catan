from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import torch

from learning.dqn.epsilon_scheduler import EpsilonScheduler
from learning.dqn.replay_buffer import ReplayBuffer
from learning.gameplay.build_gameplay_q_model import build_gameplay_q_model
from learning.hybrid.hybrid_checkpoint import HybridCheckpointManager
from learning.hybrid.hybrid_rollout_manager import HybridRolloutManager
from learning.trade.build_trade_model import build_trade_model


STATE_DIM = 64
NUM_GAMEPLAY_ACTIONS = 128


@dataclass
class TradeTrajectoryItem:
    obs: Dict[str, torch.Tensor]
    action: Dict[str, torch.Tensor]
    reward: float
    value: torch.Tensor
    log_prob: Dict[str, torch.Tensor]
    done: bool


class TradeRolloutStorage:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.data: List[TradeTrajectoryItem] = []

    def reset(self) -> None:
        self.data = []

    def _clone_obs(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in obs.items():
            if isinstance(v, dict):
                out[k] = {kk: vv.detach().cpu().clone() for kk, vv in v.items()}
            else:
                out[k] = v.detach().cpu().clone()
        return out

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, torch.Tensor],
        reward: float,
        value: torch.Tensor,
        log_prob: Dict[str, torch.Tensor],
        done: bool,
    ) -> None:
        action_cpu = {}
        for k, v in action.items():
            if torch.is_tensor(v):
                action_cpu[k] = v.detach().cpu().clone()
            else:
                action_cpu[k] = v

        log_prob_cpu = {}
        for k, v in log_prob.items():
            if torch.is_tensor(v):
                log_prob_cpu[k] = v.detach().cpu().clone()
            else:
                log_prob_cpu[k] = v

        self.data.append(
            TradeTrajectoryItem(
                obs=self._clone_obs(obs),
                action=action_cpu,
                reward=float(reward),
                value=value.detach().cpu().clone(),
                log_prob=log_prob_cpu,
                done=bool(done),
            )
        )

    def __len__(self) -> int:
        return len(self.data)


class TradePPOTrainer:
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
    ):
        self.policy = policy
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.tom_loss_coef = tom_loss_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.progress = 0.0

    def set_progress(self, progress: float) -> None:
        self.progress = max(0.0, min(1.0, float(progress)))

    def entropy_coef(self) -> float:
        return self.entropy_coef_start + (
            self.entropy_coef_end - self.entropy_coef_start
        ) * self.progress

    def _stack_obs(self, items: List[TradeTrajectoryItem]):
        return {
            "board": torch.cat([item.obs["board"] for item in items], dim=0).to(self.device),
            "self": torch.cat([item.obs["self"] for item in items], dim=0).to(self.device),
            "opponent": torch.cat([item.obs["opponent"] for item in items], dim=0).to(self.device),
            "trade_history": {
                "proposer_ids": torch.cat([item.obs["trade_history"]["proposer_ids"] for item in items], dim=0).to(self.device),
                "target_ids": torch.cat([item.obs["trade_history"]["target_ids"] for item in items], dim=0).to(self.device),
                "response_types": torch.cat([item.obs["trade_history"]["response_types"] for item in items], dim=0).to(self.device),
                "offers": torch.cat([item.obs["trade_history"]["offers"] for item in items], dim=0).to(self.device),
                "requests": torch.cat([item.obs["trade_history"]["requests"] for item in items], dim=0).to(self.device),
                "accepted_flags": torch.cat([item.obs["trade_history"]["accepted_flags"] for item in items], dim=0).to(self.device),
                "turn_numbers": torch.cat([item.obs["trade_history"]["turn_numbers"] for item in items], dim=0).to(self.device),
            },
        }

    def _stack_action_field(self, items: List[TradeTrajectoryItem], field: str) -> torch.Tensor:
        vals = [item.action[field] for item in items]
        return torch.stack(vals).to(self.device)

    def _stack_log_prob_sum(self, items: List[TradeTrajectoryItem]) -> torch.Tensor:
        out = []
        for item in items:
            total = 0.0
            for v in item.log_prob.values():
                total = total + v.mean()
            out.append(total)
        return torch.stack(out).to(self.device)

    def _compute_returns_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        next_value = 0.0
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, storage: TradeRolloutStorage) -> Dict[str, float]:
        if len(storage) == 0:
            return {
                "trade_policy_loss": 0.0,
                "trade_value_loss": 0.0,
                "trade_entropy": 0.0,
                "trade_entropy_coef": self.entropy_coef(),
                "trade_tom_loss": 0.0,
                "trade_total_loss": 0.0,
            }

        items = storage.data
        obs = self._stack_obs(items)

        actions = {
            "action_type": self._stack_action_field(items, "action_type"),
            "target": self._stack_action_field(items, "target"),
            "offer": self._stack_action_field(items, "offer"),
            "request": self._stack_action_field(items, "request"),
        }

        rewards = torch.tensor([x.reward for x in items], dtype=torch.float32, device=self.device)
        values = torch.cat([x.value for x in items], dim=0).squeeze(-1).to(self.device)
        dones = torch.tensor([x.done for x in items], dtype=torch.float32, device=self.device)
        old_log_probs = self._stack_log_prob_sum(items)

        returns, advantages = self._compute_returns_advantages(rewards, values, dones)

        logits, new_values, tom_outputs = self.policy.forward(obs)
        log_prob_dict, entropy = self.policy.action_heads.evaluate_actions(logits, actions)

        new_log_probs = (
            log_prob_dict["action_type"]
            + log_prob_dict["target"]
            + log_prob_dict["offer"].sum(dim=-1)
            + log_prob_dict["request"].sum(dim=-1)
        )

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        new_values = new_values.squeeze(-1)
        value_loss = torch.nn.functional.smooth_l1_loss(new_values, returns)

        tom_loss = torch.tensor(0.0, device=self.device)
        if hasattr(self.policy, "tom_head") and hasattr(self.policy.tom_head, "compute_loss"):
            if isinstance(tom_outputs, dict):
                zero_targets = {}
                for k, v in tom_outputs.items():
                    zero_targets[k] = torch.zeros_like(v)
                computed = self.policy.tom_head.compute_loss(tom_outputs, zero_targets)
                if torch.is_tensor(computed):
                    tom_loss = computed
                else:
                    tom_loss = torch.tensor(float(computed), dtype=torch.float32, device=self.device)

        ent_coef = self.entropy_coef()
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - ent_coef * entropy
            + self.tom_loss_coef * tom_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        entropy_value = entropy.item() if torch.is_tensor(entropy) else float(entropy)

        return {
            "trade_policy_loss": float(policy_loss.item()),
            "trade_value_loss": float(value_loss.item()),
            "trade_entropy": float(entropy_value),
            "trade_entropy_coef": float(ent_coef),
            "trade_tom_loss": float(tom_loss.item()),
            "trade_total_loss": float(total_loss.item()),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid DQN gameplay + PPO trade training loop")
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--dqn-batch-size", type=int, default=256)
    parser.add_argument("--dqn-updates-per-iter", type=int, default=8)
    parser.add_argument("--warmup-transitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def train(args: argparse.Namespace) -> None:
    device = args.device

    dqn_config = {
        "state_dim": STATE_DIM,
        "num_actions": NUM_GAMEPLAY_ACTIONS,
        "hidden_dim": 256,
        "lr": 1e-3,
        "gamma": 0.99,
        "target_update_freq": 1000,
        "device": device,
    }

    gameplay_trainer = build_gameplay_q_model(dqn_config)
    trade_policy = build_trade_model(device=device)
    trade_trainer = TradePPOTrainer(policy=trade_policy, device=device)

    replay_buffer = ReplayBuffer(
        capacity=args.replay_capacity,
        device=device,
        seed=args.seed,
    )
    epsilon_scheduler = EpsilonScheduler(
        start_epsilon=1.0,
        end_epsilon=0.05,
        decay_steps=100_000,
    )

    env_config: Dict[str, object] = {}
    rollout_manager = HybridRolloutManager(
        num_envs=args.num_envs,
        env_config=env_config,
        device=device,
    )
    trade_storage = TradeRolloutStorage(device=device)
    checkpoint_manager = HybridCheckpointManager(args.checkpoint_dir)

    gameplay_reward_window = deque(maxlen=20)
    trade_reward_window = deque(maxlen=20)

    global_step = 0

    print(f"Starting hybrid training for {args.num_updates} updates")

    for update in range(args.num_updates):
        progress = update / float(max(args.num_updates - 1, 1))
        trade_trainer.set_progress(progress)

        epsilon = epsilon_scheduler.value(global_step)

        gameplay_transitions, trade_storage = rollout_manager.collect(
            dqn_trainer=gameplay_trainer,
            epsilon=epsilon,
            trade_policy=trade_policy,
            rollout_storage=trade_storage,
            steps=args.rollout_steps,
        )

        gameplay_rollouts = len(gameplay_transitions)
        trade_rollouts = len(trade_storage)

        gameplay_reward_mean = 0.0
        if gameplay_rollouts > 0:
            gameplay_reward_mean = sum(x["reward"] for x in gameplay_transitions) / gameplay_rollouts

        trade_reward_mean = 0.0
        if trade_rollouts > 0:
            trade_reward_mean = sum(x.reward for x in trade_storage.data) / trade_rollouts

        gameplay_reward_window.append(gameplay_reward_mean)
        trade_reward_window.append(trade_reward_mean)

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
            global_step += 1

        dqn_metrics = {
            "td_loss": 0.0,
            "q_mean": 0.0,
            "target_q_mean": 0.0,
        }

        if len(replay_buffer) >= max(args.warmup_transitions, args.dqn_batch_size):
            batch_metrics = []
            for _ in range(args.dqn_updates_per_iter):
                batch = replay_buffer.sample(args.dqn_batch_size)
                batch_metrics.append(gameplay_trainer.update(batch))

            dqn_metrics = {
                "td_loss": float(sum(x["td_loss"] for x in batch_metrics) / len(batch_metrics)),
                "q_mean": float(sum(x["q_mean"] for x in batch_metrics) / len(batch_metrics)),
                "target_q_mean": float(sum(x["target_q_mean"] for x in batch_metrics) / len(batch_metrics)),
            }

        trade_metrics = trade_trainer.update(trade_storage)

        avg_gameplay_reward = sum(gameplay_reward_window) / len(gameplay_reward_window)
        avg_trade_reward = sum(trade_reward_window) / len(trade_reward_window)

        print(f"\nUpdate {update}")
        print(f"rollouts | gameplay={gameplay_rollouts} trade={trade_rollouts}")
        print(
            f"reward   | gameplay={gameplay_reward_mean:.4f} (avg={avg_gameplay_reward:.4f}) "
            f"trade={trade_reward_mean:.4f} (avg={avg_trade_reward:.4f})"
        )
        print(f"dqn      | epsilon={epsilon:.4f} td_loss={dqn_metrics['td_loss']:.4f} q_mean={dqn_metrics['q_mean']:.4f}")
        print(
            f"trade    | policy={trade_metrics['trade_policy_loss']:.4f} "
            f"value={trade_metrics['trade_value_loss']:.4f} "
            f"entropy={trade_metrics['trade_entropy']:.4f} "
            f"(coef={trade_metrics['trade_entropy_coef']:.6f}) "
            f"tom={trade_metrics['trade_tom_loss']:.6f}"
        )

        if (update + 1) % 20 == 0:
            path = checkpoint_manager.save(
                step=update + 1,
                dqn_trainer=gameplay_trainer,
                trade_trainer=trade_trainer,
                extra={
                    "epsilon": epsilon,
                    "global_step": global_step,
                    "update": update + 1,
                },
            )
            print(f"saved checkpoint -> {path}")

    print("training complete")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)