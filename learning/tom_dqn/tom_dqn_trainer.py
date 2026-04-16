from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class ToMEnhancedDQNTrainer:
    """
    DQN trainer enhanced with ToM capabilities and global state awareness.
    Includes opponent need prediction training for better trading.
    """

    def __init__(
        self,
        policy: nn.Module,
        target_policy: nn.Module,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 10000,
        batch_size: int = 64,
        device: str = "cpu",
        tom_loss_coef: float = 0.1,
    ):
        self.policy = policy.to(device)
        self.target_policy = target_policy.to(device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.tom_loss_coef = tom_loss_coef

        # Experience replay buffer
        self.buffer = []
        self.buffer_size = buffer_size
        self.buffer_idx = 0

    def _stack_obs(self, storage: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "board": torch.cat([x["obs"]["board"] for x in storage], dim=0).to(self.device),
            "self": torch.cat([x["obs"]["self"] for x in storage], dim=0).to(self.device),
            "opponent": torch.cat([x["obs"]["opponent"] for x in storage], dim=0).to(self.device),
            "global_state": torch.cat([x["obs"].get("global_state", torch.zeros_like(x["obs"]["board"])) for x in storage], dim=0).to(self.device),
            "gameplay_candidates": torch.cat([x["obs"]["gameplay_candidates"] for x in storage], dim=0).to(self.device),
            "gameplay_mask": torch.cat([x["obs"]["gameplay_mask"] for x in storage], dim=0).to(self.device),
            "trade_candidates": torch.cat([x["obs"]["trade_candidates"] for x in storage], dim=0).to(self.device),
            "trade_mask": torch.cat([x["obs"]["trade_mask"] for x in storage], dim=0).to(self.device),
        }

    def store_transition(self, obs: Dict[str, torch.Tensor], action: Dict[str, torch.Tensor],
                        reward: float, next_obs: Dict[str, torch.Tensor], done: bool, phase: str):
        """Store transition in replay buffer."""
        transition = {
            "obs": {k: v.detach().clone() for k, v in obs.items()},
            "action": {k: v.detach().clone() for k, v in action.items()},
            "reward": reward,
            "next_obs": {k: v.detach().clone() for k, v in next_obs.items()},
            "done": done,
            "phase": phase,
        }

        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.buffer_idx] = transition
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

    def sample_batch(self) -> Optional[Dict]:
        """Sample a batch of transitions from replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        indices = torch.randint(0, len(self.buffer), (self.batch_size,))
        batch = {
            "obs": {k: torch.stack([self.buffer[i]["obs"][k] for i in indices]) for k in self.buffer[0]["obs"].keys()},
            "action": {k: torch.stack([self.buffer[i]["action"][k] for i in indices]) for k in self.buffer[0]["action"].keys()},
            "reward": torch.tensor([self.buffer[i]["reward"] for i in indices], dtype=torch.float32, device=self.device),
            "next_obs": {k: torch.stack([self.buffer[i]["next_obs"][k] for i in indices]) for k in self.buffer[0]["next_obs"].keys()},
            "done": torch.tensor([self.buffer[i]["done"] for i in indices], dtype=torch.float32, device=self.device),
            "phase": [self.buffer[i]["phase"] for i in indices],
        }
        return batch

    def update(self) -> Dict[str, float]:
        """Update policy using a batch from replay buffer."""
        batch = self.sample_batch()
        if batch is None:
            return {"loss": 0.0, "tom_loss": 0.0}

        # Separate gameplay and trade transitions
        gameplay_mask = torch.tensor([p == "gameplay" for p in batch["phase"]], device=self.device)
        trade_mask = torch.tensor([p == "trade" for p in batch["phase"]], device=self.device)

        total_loss = 0.0
        tom_loss_total = 0.0

        # Update gameplay transitions
        if gameplay_mask.any():
            gp_loss = self._update_phase(batch, "gameplay", gameplay_mask)
            total_loss += gp_loss

        # Update trade transitions with ToM
        if trade_mask.any():
            tr_loss, tom_loss = self._update_trade_phase(batch, trade_mask)
            total_loss += tr_loss
            tom_loss_total += tom_loss

        # Soft update target network
        self._soft_update_target()

        return {
            "loss": total_loss.item(),
            "tom_loss": tom_loss_total.item() if isinstance(tom_loss_total, torch.Tensor) else tom_loss_total,
        }

    def _update_phase(self, batch: Dict, phase: str, mask: torch.Tensor) -> torch.Tensor:
        """Update for a specific phase (gameplay or trade without ToM)."""
        obs = {k: v[mask] for k, v in batch["obs"].items()}
        actions = {k: v[mask] for k, v in batch["action"].items()}
        rewards = batch["reward"][mask]
        next_obs = {k: v[mask] for k, v in batch["next_obs"].items()}
        dones = batch["done"][mask]

        # Current Q values
        _, current_q_values, _, _ = self.policy.evaluate_actions(obs, actions, phase)

        # Target Q values
        with torch.no_grad():
            if phase == "gameplay":
                next_actions, _ = self.target_policy.get_gameplay_action(next_obs, epsilon=0.0)
                _, next_q_values, _, _ = self.target_policy.evaluate_actions(next_obs, next_actions, phase)
            else:
                next_actions, _ = self.target_policy.get_trade_action(next_obs, epsilon=0.0)
                _, next_q_values, _, _ = self.target_policy.evaluate_actions(next_obs, next_actions, phase)

            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.functional.mse_loss(current_q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def _update_trade_phase(self, batch: Dict, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update trade phase with ToM supervision."""
        obs = {k: v[mask] for k, v in batch["obs"].items()}
        actions = {k: v[mask] for k, v in batch["action"].items()}
        rewards = batch["reward"][mask]
        next_obs = {k: v[mask] for k, v in batch["next_obs"].items()}
        dones = batch["done"][mask]

        # Current Q values and ToM predictions
        _, current_q_values, _, tom_outputs = self.policy.evaluate_actions(obs, actions, "trade")

        # Target Q values
        with torch.no_grad():
            next_actions, _ = self.target_policy.get_trade_action(next_obs, epsilon=0.0)
            _, next_q_values, _, _ = self.target_policy.evaluate_actions(next_obs, next_actions, "trade")
            target_q = rewards + self.gamma * next_q_values * (1 - dones)

        # Q-learning loss
        q_loss = nn.functional.mse_loss(current_q_values, target_q)

        # ToM supervision loss (if available)
        tom_loss = torch.tensor(0.0, device=self.device)
        # Note: ToM loss would require target labels from trade events
        # This is simplified - in practice you'd need trade event supervision

        total_loss = q_loss + self.tom_loss_coef * tom_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return q_loss, tom_loss

    def _soft_update_target(self):
        """Soft update target network parameters."""
        for target_param, policy_param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str, step: int):
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "target_policy_state_dict": self.target_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.target_policy.load_state_dict(checkpoint["target_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])