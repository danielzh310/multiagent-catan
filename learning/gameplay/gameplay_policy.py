from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class GameplayPolicy(nn.Module):
    """
    Gameplay policy for non-trade actions.

    This model handles:
    - settlement / road / city decisions
    - end turn logic
    - general progression decisions

    It follows an actor-critic structure.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.device = device
        self.use_lstm = False

        input_dim = 64
        hidden_dim = 128

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim, 8)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = obs["flat"]

        features = self.encoder(x)

        logits = self.policy_head(features)
        value = self.value_head(features)

        return logits, value

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        action_masks: Optional[torch.Tensor] = None,
        hidden_state=None,
        done_mask=None,
        deterministic: bool = False,
    ):
        logits, value = self.forward(obs)

        if action_masks is not None:
            logits = logits + torch.log(action_masks + 1e-8)

        probs = torch.softmax(logits, dim=-1)

        if deterministic:
            action_idx = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()

        action_log_prob = torch.log(probs.gather(-1, action_idx.unsqueeze(-1)) + 1e-8)

        action_dict = {
            "action_type": action_idx
        }

        return value, action_dict, action_log_prob, hidden_state, None

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        action_dict: Dict[str, torch.Tensor],
        action_masks: Optional[torch.Tensor] = None,
    ):
        logits, value = self.forward(obs)

        if action_masks is not None:
            logits = logits + torch.log(action_masks + 1e-8)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        action_idx = action_dict["action_type"]

        action_log_probs = dist.log_prob(action_idx)
        entropy = dist.entropy()

        return value, action_log_probs, entropy