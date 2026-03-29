from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from learning.dqn.q_network import QNetwork


class DQNTrainer:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        device: str = "cpu",
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        hidden_dim: int = 256,
    ):
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.q_network = QNetwork(state_dim, num_actions, hidden_dim).to(device)
        self.target_q_network = QNetwork(state_dim, num_actions, hidden_dim).to(device)

        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.train_step_count = 0

    def compute_q_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.q_network(obs)

    def compute_target_q_values(self, next_obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_q_network(next_obs)

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        action_masks = batch["action_masks"]
        next_action_masks = batch["next_action_masks"]

        q_values = self.q_network(obs)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_q_values = self.target_q_network(next_obs)
            next_q_values[next_action_masks == 0] = -1e9
            max_next_q = next_q_values.max(dim=1)[0]

            target_q = rewards + self.gamma * (1.0 - dones) * max_next_q

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1

        if self.train_step_count % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        return {
            "td_loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

    def act(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        epsilon: float,
    ) -> int:
        return self.q_network.select_action(obs, action_mask, epsilon)

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])