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
            return self.policy.trade_head(
                trunk=trunk,
                trade_candidates=obs["trade_candidates"],
                trade_mask=obs["trade_mask"],
            )

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
                return self.target_policy.trade_head(
                    trunk=trunk,
                    trade_candidates=next_obs["trade_candidates"],
                    trade_mask=next_obs["trade_mask"],
                )

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update the DQN policy using a batch of experiences.
        This method processes gameplay and trade phases separately in a vectorized manner
        and uses Double DQN for calculating the target Q-values.
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"]
        dones = batch["dones"].to(self.device)
        phases = batch["phases"]

        total_loss = torch.tensor(0.0, device=self.device)
        all_q_values = []
        all_target_q_values = []

        for phase_name in ["gameplay", "trade"]:
            if phase_name not in actions or actions[phase_name].numel() == 0:
                continue

            indices = torch.tensor([i for i, p in enumerate(phases) if p == phase_name], device=self.device, dtype=torch.long)
            if indices.numel() == 0:
                continue

            phase_obs = {k: v[indices] for k, v in obs.items()}
            phase_next_obs = {k: v[indices] for k, v in next_obs.items()}
            phase_rewards = rewards[indices]
            phase_dones = dones[indices]
            phase_actions = actions[phase_name].to(self.device).unsqueeze(1)

            # 1. Get Q-values for the actions that were actually taken
            current_q_vals_all = self.compute_q_values(phase_obs, phase_name)
            current_q_vals = current_q_vals_all.gather(1, phase_actions)

            # 2. Compute target Q-values using Double DQN
            with torch.no_grad():
                # Select best action using the online network
                next_q_vals_online = self.compute_q_values(phase_next_obs, phase_name)
                best_next_actions = next_q_vals_online.argmax(dim=1, keepdim=True)

                # Evaluate that action using the target network
                next_q_vals_target = self.compute_target_q_values(phase_next_obs, phase_name)
                max_next_q = next_q_vals_target.gather(1, best_next_actions)

                # Compute the TD target
                td_target = phase_rewards.unsqueeze(1) + self.gamma * (1.0 - phase_dones.unsqueeze(1)) * max_next_q

            loss = self.loss_fn(current_q_vals, td_target)
            total_loss += loss

            all_q_values.append(current_q_vals.detach())
            all_target_q_values.append(td_target.detach())

        if total_loss.item() == 0.0:
            return {"td_loss": 0.0, "q_mean": 0.0, "target_q_mean": 0.0}

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        return {
            "td_loss": total_loss.item(),
            "q_mean": torch.cat(all_q_values).mean().item() if all_q_values else 0.0,
            "target_q_mean": torch.cat(all_target_q_values).mean().item() if all_target_q_values else 0.0,
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