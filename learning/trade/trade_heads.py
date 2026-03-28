from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class TradeActionHeads(nn.Module):
    """
    Action heads for trade decisions.

    Outputs:
    - action type (propose / accept / reject / counter / skip)
    - target player
    - offer vector
    - request vector
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.action_type_head = nn.Linear(hidden_dim, 5)
        self.target_head = nn.Linear(hidden_dim, 4)

        self.offer_head = nn.Linear(hidden_dim, 5)
        self.request_head = nn.Linear(hidden_dim, 5)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "action_type": self.action_type_head(features),
            "target": self.target_head(features),
            "offer": self.offer_head(features),
            "request": self.request_head(features),
        }

    def apply_action_mask(
        self,
        logits: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        masked = {}

        for key in logits:
            if key in masks:
                masked[key] = logits[key] + torch.log(masks[key] + 1e-8)
            else:
                masked[key] = logits[key]

        return masked

    def sample_action(
        self,
        logits: Dict[str, torch.Tensor],
        deterministic: bool = False
    ):
        action_dict = {}
        log_probs = {}

        # action type
        probs = torch.softmax(logits["action_type"], dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8)

        action_dict["action_type"] = action
        log_probs["action_type"] = log_prob

        # target
        probs = torch.softmax(logits["target"], dim=-1)
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8)

        action_dict["target"] = action
        log_probs["target"] = log_prob

        # offer (treated as multi-label via sigmoid)
        offer_logits = logits["offer"]
        offer_probs = torch.sigmoid(offer_logits)
        offer_sample = (offer_probs > 0.5).float()

        action_dict["offer"] = offer_sample
        log_probs["offer"] = torch.log(offer_probs + 1e-8)

        # request (same treatment)
        request_logits = logits["request"]
        request_probs = torch.sigmoid(request_logits)
        request_sample = (request_probs > 0.5).float()

        action_dict["request"] = request_sample
        log_probs["request"] = torch.log(request_probs + 1e-8)

        return action_dict, log_probs

    def evaluate_actions(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor]
    ):
        log_probs = {}
        entropy = 0.0

        # action type
        probs = torch.softmax(logits["action_type"], dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = actions["action_type"]
        log_probs["action_type"] = dist.log_prob(action)
        entropy += dist.entropy().mean()

        # target
        probs = torch.softmax(logits["target"], dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = actions["target"]
        log_probs["target"] = dist.log_prob(action)
        entropy += dist.entropy().mean()

        # offer
        offer_logits = logits["offer"]
        offer_probs = torch.sigmoid(offer_logits)
        log_probs["offer"] = actions["offer"] * torch.log(offer_probs + 1e-8)

        # request
        request_logits = logits["request"]
        request_probs = torch.sigmoid(request_logits)
        log_probs["request"] = actions["request"] * torch.log(request_probs + 1e-8)

        return log_probs, entropy