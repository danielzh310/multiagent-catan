from __future__ import annotations

import torch
import torch.nn as nn


class OpponentNeedPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_resources: int = 5,
    ):
        super().__init__()
        self.num_resources = num_resources

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_resources),
        )

    def forward(self, opponent_embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.net(opponent_embeddings)
        return torch.softmax(logits, dim=-1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        eps = 1e-8
        loss = -(targets * torch.log(predictions + eps)).sum(dim=-1)

        if mask is not None:
            loss = loss * mask
            denom = mask.sum().clamp_min(1.0)
            return loss.sum() / denom

        return loss.mean()