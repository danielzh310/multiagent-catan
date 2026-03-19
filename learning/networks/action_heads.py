"""
Multi-head action output module.

This module:
- produces distributions for each action type
- handles masking
- supports autoregressive-style decisions (type → target)

This design allows flexible, structured action spaces for Catan.
"""

import torch
import torch.nn as nn

from learning.probability_heads import CategoricalHead
from learning.networks.network_utils import init_linear


class ActionHeads(nn.Module):
    """
    Multi-head action module.

    Heads:
    - action type
    - settlement location
    - road connection
    - city upgrade
    - robber movement
    - trade decisions (basic)
    """

    def __init__(
        self,
        input_dim,
        num_action_types,
        num_vertices,
        num_connections,
        num_tiles,
    ):
        super().__init__()

        hidden_dim = 256

        self.shared = nn.Sequential(
            init_linear(nn.Linear(input_dim, hidden_dim), gain=1.414),
            nn.ReLU(),
        )

        self.action_type_head = CategoricalHead(hidden_dim, num_action_types)
        self.settlement_head = CategoricalHead(hidden_dim, num_vertices)
        self.road_head = CategoricalHead(hidden_dim, num_connections)
        self.city_head = CategoricalHead(hidden_dim, num_vertices)
        self.robber_head = CategoricalHead(hidden_dim, num_tiles)

        self.trade_head = CategoricalHead(hidden_dim, 2)  # accept / reject

    def forward(self, x, action_masks=None):
        """
        x: (B, input_dim)

        action_masks: dict of masks per head
        """
        h = self.shared(x)

        outputs = {}

        outputs["action_type"] = self.action_type_head(
            h,
            mask=self._get_mask(action_masks, "action_type"),
        )

        outputs["settlement"] = self.settlement_head(
            h,
            mask=self._get_mask(action_masks, "settlement"),
        )

        outputs["road"] = self.road_head(
            h,
            mask=self._get_mask(action_masks, "road"),
        )

        outputs["city"] = self.city_head(
            h,
            mask=self._get_mask(action_masks, "city"),
        )

        outputs["robber"] = self.robber_head(
            h,
            mask=self._get_mask(action_masks, "robber"),
        )

        outputs["trade"] = self.trade_head(
            h,
            mask=self._get_mask(action_masks, "trade"),
        )

        return outputs

    def sample(self, outputs):
        """
        Sample an action dictionary from distributions.
        """
        action = {}

        action_type = outputs["action_type"].sample()
        action["action_type"] = action_type

        action["settlement"] = outputs["settlement"].sample()
        action["road"] = outputs["road"].sample()
        action["city"] = outputs["city"].sample()
        action["robber"] = outputs["robber"].sample()
        action["trade"] = outputs["trade"].sample()

        return action

    def log_probs(self, outputs, actions):
        """
        Compute log-probabilities of a sampled action.
        """
        log_probs = []

        log_probs.append(outputs["action_type"].log_probs(actions["action_type"]))
        log_probs.append(outputs["settlement"].log_probs(actions["settlement"]))
        log_probs.append(outputs["road"].log_probs(actions["road"]))
        log_probs.append(outputs["city"].log_probs(actions["city"]))
        log_probs.append(outputs["robber"].log_probs(actions["robber"]))
        log_probs.append(outputs["trade"].log_probs(actions["trade"]))

        return torch.sum(torch.stack(log_probs), dim=0)

    def entropy(self, outputs):
        """
        Compute entropy across all heads.
        """
        entropies = []

        for key in outputs:
            entropies.append(outputs[key].entropy())

        return torch.sum(torch.stack(entropies), dim=0)

    def mode(self, outputs):
        """
        Deterministic action (argmax).
        """
        action = {}

        action["action_type"] = outputs["action_type"].mode()
        action["settlement"] = outputs["settlement"].mode()
        action["road"] = outputs["road"].mode()
        action["city"] = outputs["city"].mode()
        action["robber"] = outputs["robber"].mode()
        action["trade"] = outputs["trade"].mode()

        return action

    def _get_mask(self, masks, key):
        if masks is None:
            return None
        return masks.get(key, None)