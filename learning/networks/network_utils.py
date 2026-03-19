"""
Small neural-network utility helpers.

This file keeps shared model utilities in one place.
"""

import copy
import torch
import torch.nn as nn


def clone_module(module, num_copies):
    """Return a ModuleList of deep-copied modules."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_copies)])


def init_linear(layer, gain=1.0):
    """Orthogonal init for a linear layer."""
    if not isinstance(layer, nn.Linear):
        raise TypeError("init_linear expects nn.Linear")

    nn.init.orthogonal_(layer.weight, gain=gain)
    if layer.bias is not None:
        nn.init.constant_(layer.bias, 0.0)

    return layer


class ValueNormalizer(nn.Module):
    """
    Simple running value normalizer placeholder.

    This version stores mean/std parameters that can be used
    to normalize and denormalize value targets.
    """

    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean), dtype=torch.float32))
        self.register_buffer("std", torch.tensor(float(std), dtype=torch.float32))

    def normalize(self, values):
        """Normalize values."""
        return (values - self.mean) / (self.std + 1e-6)

    def denormalize(self, values):
        """Undo normalization."""
        return self.mean + values * self.std

    def set_stats(self, mean, std):
        """Update stored statistics."""
        self.mean.fill_(float(mean))
        self.std.fill_(float(std))