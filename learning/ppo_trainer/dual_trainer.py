from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class DualTrainer:
    """
    Joint PPO-style trainer for:
    - gameplay policy
    - trade policy

    Tuned to avoid overcorrecting the trade side:
    - moderate entropy annealing
    - Huber value loss
    - value clipping
    - lighter auxiliary weighting
    """

    def __init__(
        self,
        gameplay_model,
        trade_model,
        gameplay_lr: float = 3e-4,
        trade_lr: float = 3e-4,
        clip_param: float = 0.2,
        value_loss_coef: float = 0.5,
        gameplay_entropy_coef_start: float = 0.008,
        gameplay_entropy_coef_end: float = 0.0015,
        trade_entropy_coef_start: float = 0.0005,
        trade_entropy_coef_end: float = 0.00005,
        tom_loss_coef: float = 0.02,
        max_grad_norm: float = 0.5,
        value_clip_param: float = 0.2,
        use_value_clipping: bool = True,
        device: str = "cpu",
    ):
        self.gameplay_model = gameplay_model
        self.trade_model = trade_model

        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef

        self.gameplay_entropy_coef_start = gameplay_entropy_coef_start
        self.gameplay_entropy_coef_end = gameplay_entropy_coef_end

        self.trade_entropy_coef_start = trade_entropy_coef_start
        self.trade_entropy_coef_end = trade_entropy_coef_end

        self.tom_loss_coef = tom_loss_coef
        self.max_grad_norm = max_grad_norm

        self.value_clip_param = value_clip_param
        self.use_value_clipping = use_value_clipping

        self.device = device
        self.current_progress = 0.0

        self.gameplay_optimizer = torch.optim.Adam(
            self.gameplay_model.parameters(),
            lr=gameplay_lr,
        )
        self.trade_optimizer = torch.optim.Adam(
            self.trade_model.parameters(),
            lr=trade_lr,
        )

    def set_progress(self, progress: float) -> None:
        self.current_progress = max(0.0, min(1.0, float(progress)))

    def _interp(self, start: float, end: float) -> float:
        return start + (end - start) * self.current_progress

    def current_gameplay_entropy_coef(self) -> float:
        return self._interp(
            self.gameplay_entropy_coef_start,
            self.gameplay_entropy_coef_end,
        )

    def current_trade_entropy_coef(self) -> float:
        return self._interp(
            self.trade_entropy_coef_start,
            self.trade_entropy_coef_end,
        )

    def _compute_value_loss(
        self,
        predicted_values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_value_clipping:
            return nn.functional.smooth_l1_loss(predicted_values, returns)

        value_pred_clipped = old_values + (predicted_values - old_values).clamp(
            -self.value_clip_param,
            self.value_clip_param,
        )

        value_loss_unclipped = nn.functional.smooth_l1_loss(
            predicted_values,
            returns,
            reduction="none",
        )
        value_loss_clipped = nn.functional.smooth_l1_loss(
            value_pred_clipped,
            returns,
            reduction="none",
        )

        return torch.max(value_loss_unclipped, value_loss_clipped).mean()

    def train_gameplay_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if not batch:
            return {
                "gameplay_policy_loss": 0.0,
                "gameplay_value_loss": 0.0,
                "gameplay_entropy": 0.0,
                "gameplay_entropy_coef": self.current_gameplay_entropy_coef(),
                "gameplay_total_loss": 0.0,
            }

        obs = {"flat": batch["obs"]}
        actions = {"action_type": batch["actions"]}
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_log_probs = batch["log_probs"]
        old_values = batch["values"]

        logits, values = self.gameplay_model.forward(obs)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        new_log_probs = dist.log_prob(actions["action_type"])
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = values.squeeze(-1)
        value_loss = self._compute_value_loss(values, old_values, returns)

        entropy_coef = self.current_gameplay_entropy_coef()
        total_loss = policy_loss + self.value_loss_coef * value_loss - entropy_coef * entropy

        self.gameplay_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.gameplay_model.parameters(), self.max_grad_norm)
        self.gameplay_optimizer.step()

        return {
            "gameplay_policy_loss": float(policy_loss.item()),
            "gameplay_value_loss": float(value_loss.item()),
            "gameplay_entropy": float(entropy.item()),
            "gameplay_entropy_coef": float(entropy_coef),
            "gameplay_total_loss": float(total_loss.item()),
        }

    def train_trade_step(
        self,
        batch: Dict[str, torch.Tensor],
        tom_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        if not batch:
            return {
                "trade_policy_loss": 0.0,
                "trade_value_loss": 0.0,
                "trade_entropy": 0.0,
                "trade_entropy_coef": self.current_trade_entropy_coef(),
                "trade_tom_loss": 0.0,
                "trade_total_loss": 0.0,
            }

        obs = batch["obs"]
        actions = batch["actions"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_log_probs = batch["log_probs"]
        old_values = batch["values"]

        logits, values, tom_outputs = self.trade_model.forward(obs)

        log_prob_dict, entropy = self.trade_model.action_heads.evaluate_actions(logits, actions)

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

        values = values.squeeze(-1)
        value_loss = self._compute_value_loss(values, old_values, returns)

        tom_loss = torch.tensor(0.0, device=self.device)
        if tom_targets is not None:
            tom_loss = self.trade_model.tom_head.compute_loss(tom_outputs, tom_targets)

        entropy_coef = self.current_trade_entropy_coef()
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - entropy_coef * entropy
            + self.tom_loss_coef * tom_loss
        )

        self.trade_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.trade_model.parameters(), self.max_grad_norm)
        self.trade_optimizer.step()

        return {
            "trade_policy_loss": float(policy_loss.item()),
            "trade_value_loss": float(value_loss.item()),
            "trade_entropy": float(entropy.item() if torch.is_tensor(entropy) else entropy),
            "trade_entropy_coef": float(entropy_coef),
            "trade_tom_loss": float(tom_loss.item()),
            "trade_total_loss": float(total_loss.item()),
        }

    def train_joint_step(
        self,
        gameplay_batch: Dict[str, torch.Tensor],
        trade_batch: Dict[str, torch.Tensor],
        tom_targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        gameplay_metrics = self.train_gameplay_step(gameplay_batch)
        trade_metrics = self.train_trade_step(trade_batch, tom_targets=tom_targets)

        out = {}
        out.update(gameplay_metrics)
        out.update(trade_metrics)
        return out