"""
Board encoder for tile-level board state.

This module:
- projects per-tile features into a model space
- applies stacked self-attention blocks
- returns a flattened board representation for downstream policy/value heads
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .network_utils import clone_module, init_linear


class FeedForwardBlock(nn.Module):
    def __init__(self, model_dim, hidden_mult=2):
        super().__init__()
        hidden_dim = model_dim * hidden_mult

        self.fc1 = init_linear(nn.Linear(model_dim, hidden_dim), gain=1.414)
        self.fc2 = init_linear(nn.Linear(hidden_dim, model_dim), gain=1.414)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class EncoderBlock(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()

        self.attn = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads)
        self.ffn = FeedForwardBlock(model_dim=model_dim, hidden_mult=2)

        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x, mask=None):
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + attn_out

        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


class BoardEncoder(nn.Module):
    """
    Encodes all board tiles into a single board representation.

    Input:
        tile_features: (B, num_tiles, tile_feature_dim)

    Output:
        flattened board embedding: (B, num_tiles * out_proj_dim)
    """

    def __init__(
        self,
        tile_feature_dim,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        out_proj_dim=24,
    ):
        super().__init__()

        self.tile_feature_dim = tile_feature_dim
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.out_proj_dim = out_proj_dim

        self.input_proj = init_linear(nn.Linear(tile_feature_dim, model_dim), gain=1.414)
        self.encoder_blocks = clone_module(EncoderBlock(model_dim, num_heads), num_layers)

        self.out_proj = init_linear(nn.Linear(model_dim, out_proj_dim), gain=1.414)

        self.norm_in = nn.LayerNorm(model_dim)
        self.norm_out = nn.LayerNorm(out_proj_dim)
        self.relu = nn.ReLU()

    def forward(self, tile_features, mask=None):
        """
        tile_features:
            (B, T, F)

        mask:
            optional attention mask
        """
        x = self.input_proj(tile_features)
        x = self.relu(self.norm_in(x))

        for block in self.encoder_blocks:
            x = block(x, mask=mask)

        x = self.out_proj(x)
        x = self.relu(self.norm_out(x))

        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)