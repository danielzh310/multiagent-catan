from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

from .dqn_policy import DQNBaselinePolicy


class DQNTrainer:
    def __init__(
        self,
        board_dim: int = 64,
        self_dim: int = 64,
        opponent_dim: int = 64,
        hidden_dim: int = 192,
        resources: int = 5,
        device: str = "cpu",
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
    ):
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq

        self.policy = DQNBaselinePolicy(
            board_dim=board_dim,
            self_dim=self_dim,
            opponent_dim=opponent_dim,
            hidden_dim=hidden_dim,
            resources=resources,
            device=device,
        ).to(device)

        self.target_policy = DQNBaselinePolicy(
            board_dim=board_dim,
            self_dim=self_dim,
            opponent_dim=opponent_dim,
            hidden_dim=hidden_dim,
            resources=resources,
            device=device,
        ).to(device)

        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.train_step_count = 0

    def compute_q_values(self, obs: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        if phase == "gameplay":
            trunk = self.policy.encode(obs)
            return self.policy.gameplay_head(
                trunk=trunk,
                gameplay_candidates=obs["gameplay_candidates"],
                gameplay_mask=obs["gameplay_mask"]
            )
        else:
            trunk = self.policy.encode(obs)
            return self.policy.trade_head(trunk)

    def compute_target_q_values(self, next_obs: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        with torch.no_grad():
            if phase == "gameplay":
                trunk = self.target_policy.encode(next_obs)
                q_values = self.target_policy.gameplay_head(
                    trunk=trunk,
                    gameplay_candidates=next_obs["gameplay_candidates"],
                    gameplay_mask=next_obs["gameplay_mask"]
                )
                return q_values
            else:
                trunk = self.target_policy.encode(next_obs)
                return self.target_policy.trade_head(trunk)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]
        phases = batch["phases"]

        # Handle both gameplay and trade phases
        q_values = []
        target_q_values = []

        for i in range(len(phases)):
            phase = phases[i]
            obs_i = {k: v[i:i+1] for k, v in obs.items()}
            next_obs_i = {k: v[i:i+1] for k, v in next_obs.items()}

            q_vals = self.compute_q_values(obs_i, phase)
            target_q_vals = self.compute_target_q_values(next_obs_i, phase)

            if phase == "gameplay":
                action_idx = actions["gameplay_action"][i]
                q_val = q_vals[0, action_idx]
                max_next_q = target_q_vals.max().item()
            else:
                action_idx = actions["trade_action"][i]
                q_val = q_vals[0, action_idx]
                max_next_q = target_q_vals.max().item()

            q_values.append(q_val)
            target_q = rewards[i] + self.gamma * (1.0 - dones[i]) * max_next_q
            target_q_values.append(target_q)

        q_values = torch.stack(q_values)
        target_q_values = torch.tensor(target_q_values, device=self.device)

        loss = self.loss_fn(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1

        if self.train_step_count % self.target_update_freq == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        return {
            "td_loss": loss.item(),
            "q_mean": q_values.mean().item(),
            "target_q_mean": target_q_values.mean().item(),
        }

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        phase: str,
        epsilon: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        if phase == "gameplay":
            action, _ = self.policy.get_gameplay_action(obs, epsilon)
            return action
        else:
            action, _ = self.policy.get_trade_action(obs, epsilon)
            return action

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target_policy": self.target_policy.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.target_policy.load_state_dict(checkpoint["target_policy"])