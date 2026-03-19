"""
Common utility helpers used across the learning stack.
"""

import torch
import torch.nn as nn


def init_module(module, weight_init, bias_init, gain=1.0):
    """
    Initialize a module with the provided functions.
    """
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class AddBias(nn.Module):
    """
    Small helper module for learned log-std style parameters.
    """

    def __init__(self, bias):
        super().__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def get_render_fn(env):
    """
    Recursively find a render function from nested env wrappers.
    """
    if hasattr(env, "envs"):
        return env.envs[0].render
    if hasattr(env, "venv"):
        return get_render_fn(env.venv)
    if hasattr(env, "env"):
        return get_render_fn(env.env)

    return None