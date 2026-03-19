"""
Action distribution helpers.

These wrap PyTorch distributions so the policy code can use:
- sample()
- log_probs()
- entropy()
- mode()

in a consistent way.
"""

import torch
import torch.nn as nn

from learning.common_utils import AddBias, init_module


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        probs = self.probs.masked_fill(self.probs <= 0, 1.0)
        return -1.0 * probs.mul(probs.log()).sum(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.linear = init_module(
            nn.Linear(input_dim, output_dim),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01,
        )

    def forward(self, x, mask=None):
        logits = self.linear(x)

        if mask is not None:
            logits = logits + torch.log(mask)

        return FixedCategorical(logits=logits)


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


class DiagonalGaussianHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.mean_layer = init_module(
            nn.Linear(input_dim, output_dim),
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=1.0,
        )

        initial_log_std = -0.693471  # exp(...) ~ 0.5
        self.log_std = AddBias(initial_log_std * torch.ones(output_dim))

    def forward(self, x, mask=None):
        action_mean = self.mean_layer(x)

        zeros = torch.zeros_like(action_mean)
        action_log_std = self.log_std(zeros)

        return FixedNormal(action_mean, action_log_std.exp())