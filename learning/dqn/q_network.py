from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def masked_q_values(
        self,
        q_values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        masked_q = q_values.clone()
        masked_q[action_mask == 0] = -1e9
        return masked_q

    def select_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        epsilon: float,
    ) -> int:
        if action_mask.dim() == 2:
            action_mask = action_mask.squeeze(0)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        if torch.rand(1).item() < epsilon:
            valid_actions = torch.nonzero(action_mask > 0, as_tuple=False).squeeze(-1)

            if valid_actions.numel() == 0:
                return 0

            idx = torch.randint(valid_actions.numel(), (1,)).item()
            return int(valid_actions[idx].item())

        with torch.no_grad():
            q_values = self.forward(obs)

            if q_values.dim() == 2 and q_values.size(0) == 1:
                q_values = q_values.squeeze(0)

            masked_q = q_values.clone()
            masked_q[action_mask == 0] = -1e9

            return int(torch.argmax(masked_q).item())