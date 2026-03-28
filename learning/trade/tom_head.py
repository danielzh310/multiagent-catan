from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class ToMHead(nn.Module):
    """
    Theory-of-Mind head.

    Predicts opponent preferences / acceptance likelihood.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.acceptance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.resource_pref_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        acceptance_logits = self.acceptance_head(features)
        resource_pref_logits = self.resource_pref_head(features)

        return {
            "acceptance_logit": acceptance_logits.squeeze(-1),
            "resource_pref_logits": resource_pref_logits,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        loss = 0.0

        if "acceptance_target" in targets:
            target = targets["acceptance_target"].float()
            pred = outputs["acceptance_logit"]
            loss += nn.functional.binary_cross_entropy_with_logits(pred, target)

        if "resource_pref_target" in targets:
            target = targets["resource_pref_target"]
            pred = outputs["resource_pref_logits"]
            loss += nn.functional.cross_entropy(pred, target)

        return loss