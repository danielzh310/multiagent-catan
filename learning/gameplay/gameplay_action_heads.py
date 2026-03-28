from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class GameplayActionHeads(nn.Module):
    """
    Action heads for gameplay decisions.

    This separates action-type selection from parameter selection
    so the model can scale later if needed.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.action_type_head = nn.Linear(hidden_dim, 8)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_type_logits = self.action_type_head(features)

        return {
            "action_type": action_type_logits
        }

    def apply_action_mask(
        self,
        logits: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        masked_logits = {}

        for key in logits:
            if key in masks:
                masked_logits[key] = logits[key] + torch.log(masks[key] + 1e-8)
            else:
                masked_logits[key] = logits[key]

        return masked_logits

    def sample_action(
        self,
        logits: Dict[str, torch.Tensor],
        deterministic: bool = False
    ):
        action_dict = {}
        log_probs = {}

        probs = torch.softmax(logits["action_type"], dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8)

        action_dict["action_type"] = action
        log_probs["action_type"] = log_prob

        return action_dict, log_probs

    def evaluate_actions(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor]
    ):
        probs = torch.softmax(logits["action_type"], dim=-1)
        dist = torch.distributions.Categorical(probs)

        action = actions["action_type"]

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy