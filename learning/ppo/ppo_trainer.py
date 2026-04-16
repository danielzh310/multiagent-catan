from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import Adam


class PPOTrainer:
    def __init__(
        self,
        policy,
        lr,
        gamma,
        gae_lambda,
        clip_ratio,
        value_loss_coef,
        entropy_coef,
        max_grad_norm,
        device,
        num_envs: int = 8,
        rollout_steps: int = 128,
        num_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        self.policy = policy
        self.optimizer = Adam(policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.num_envs = num_envs
        self.rollout_steps = rollout_steps
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size

    def save(self, path: str, step: int):
        torch.save({
            "policy": self.policy.state_dict(),
            "step": step
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        # Handle both old (state_dict only) and new (dict) checkpoint formats
        if "policy" in ckpt:
            self.policy.load_state_dict(ckpt["policy"])
        else:
            self.policy.load_state_dict(ckpt)

    def update(self, storage: List[Dict[str, Any]], next_value: torch.Tensor):
        num_steps = len(storage)
        if num_steps == 0:
            return {}

        # 1. Process storage into tensors and reshape
        def to_tensor(key, dtype=torch.float32):
            return torch.tensor([s[key] for s in storage], dtype=dtype, device=self.device)

        obs_tensors = {k: torch.cat([s["obs"][k] for s in storage]) for k in storage[0]["obs"]}
        action_tensors = {k: torch.cat([s["action"][k] for s in storage]) for k in storage[0]["action"]}
        old_log_prob_tensors = {k: torch.cat([s["log_prob"][k] for s in storage]).detach() for k in storage[0]["log_prob"]}
        value_tensors = torch.cat([s["value"] for s in storage]).detach()
        reward_tensors = to_tensor("reward").view(self.rollout_steps, self.num_envs, 1)
        done_tensors = to_tensor("done").view(self.rollout_steps, self.num_envs, 1)
        phase_list = [s["phase"] for s in storage]

        # 2. Compute advantages and returns (GAE)
        with torch.no_grad():
            advantages = torch.zeros(self.rollout_steps, self.num_envs, 1, device=self.device)
            last_gae_lam = 0
            # Reshape values for GAE calculation
            values_reshaped = value_tensors.view(self.rollout_steps, self.num_envs, 1)

            for t in reversed(range(self.rollout_steps)):
                if t == self.rollout_steps - 1:
                    next_values = next_value
                else:
                    next_values = values_reshaped[t + 1]

                # The done from step t masks the value of step t+1
                next_non_terminal = 1.0 - done_tensors[t]

                delta = reward_tensors[t] + self.gamma * next_values * next_non_terminal - values_reshaped[t]
                advantages[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam

            returns = advantages + values_reshaped
            # Flatten advantages and returns
            advantages = advantages.view(-1, 1)
            returns = returns.view(-1, 1)

        # 3. Update loop
        all_metrics = {
            "gameplay_policy_loss": [], "gameplay_value_loss": [], "gameplay_entropy": [],
            "trade_policy_loss": [], "trade_value_loss": [], "trade_entropy": [],
            "total_loss": []
        }

        for _ in range(self.num_epochs):
            indices = torch.randperm(num_steps)
            for start in range(0, num_steps, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]

                # Get minibatch data
                batch_obs = {k: v[batch_indices] for k, v in obs_tensors.items()}
                batch_actions = {k: v[batch_indices] for k, v in action_tensors.items()}
                batch_old_log_probs = {k: v[batch_indices] for k, v in old_log_prob_tensors.items()}
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_phases = [phase_list[i] for i in batch_indices]

                # Separate by phase
                gp_mask = torch.tensor([p == 'gameplay' for p in batch_phases], device=self.device)
                tr_mask = torch.tensor([p == 'trade' for p in batch_phases], device=self.device)

                # *** FIX: Initialize total_loss as a tensor ***
                total_loss = torch.tensor(0.0, device=self.device)

                # --- Gameplay Loss ---
                if gp_mask.any():
                    log_probs, entropy, values = self.policy.evaluate_actions({k: v[gp_mask] for k,v in batch_obs.items()}, {k: v[gp_mask] for k,v in batch_actions.items()}, 'gameplay')
                    ratio = torch.exp(log_probs["gameplay_action"] - batch_old_log_probs["gameplay_action"][gp_mask])
                    surr1 = ratio * batch_advantages[gp_mask]
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages[gp_mask]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values, batch_returns[gp_mask])
                    gameplay_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                    total_loss += gameplay_loss
                    all_metrics["gameplay_policy_loss"].append(policy_loss.item())
                    all_metrics["gameplay_value_loss"].append(value_loss.item())
                    all_metrics["gameplay_entropy"].append(entropy.item())

                # --- Trade Loss ---
                if tr_mask.any():
                    log_probs, entropy, values = self.policy.evaluate_actions({k: v[tr_mask] for k,v in batch_obs.items()}, {k: v[tr_mask] for k,v in batch_actions.items()}, 'trade')
                    ratio = torch.exp(log_probs["trade_action"] - batch_old_log_probs["trade_action"][tr_mask])
                    surr1 = ratio * batch_advantages[tr_mask]
                    surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages[tr_mask]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(values, batch_returns[tr_mask])
                    trade_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                    total_loss += trade_loss
                    all_metrics["trade_policy_loss"].append(policy_loss.item())
                    all_metrics["trade_value_loss"].append(value_loss.item())
                    all_metrics["trade_entropy"].append(entropy.item())

                if total_loss.item() == 0.0:
                    continue

                all_metrics["total_loss"].append(total_loss.item())
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in all_metrics.items()}
        return avg_metrics