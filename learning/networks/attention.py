"""
Multi-head self-attention module used by the board and state encoders.

This is a clean project-specific version:
- batch-first inputs
- optional mask support
- simple residual-friendly output projection
"""

import math
import torch
import torch.nn as nn

from .network_utils import clone_module, init_linear


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention.

    Expected input shape:
        (batch_size, seq_len, model_dim)
    """

    def __init__(self, model_dim, num_heads):
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads

        self.qkv_layers = clone_module(nn.Linear(model_dim, model_dim), 3)
        self.out_proj = nn.Linear(model_dim, model_dim)

        for layer in self.qkv_layers:
            init_linear(layer)
        init_linear(self.out_proj)

        self.softmax = nn.Softmax(dim=-1)

    def _split_heads(self, x):
        """
        Convert:
            (B, T, D)
        into:
            (B, H, T, Hd)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _combine_heads(self, x):
        """
        Convert:
            (B, H, T, Hd)
        into:
            (B, T, D)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.model_dim)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value:
            tensors of shape (B, T, D)

        mask:
            optional tensor broadcastable to attention score shape
        """
        q = self._split_heads(self.qkv_layers[0](query))
        k = self._split_heads(self.qkv_layers[1](key))
        v = self._split_heads(self.qkv_layers[2](value))

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            # mask should be 1 for valid, 0 for invalid
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = self.softmax(scores)
        attn_output = torch.matmul(attn_weights, v)

        combined = self._combine_heads(attn_output)
        return self.out_proj(combined)