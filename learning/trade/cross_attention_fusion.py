from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """
    Fuses multiple input streams using cross-attention.

    Inputs:
    - board embedding
    - self embedding
    - opponent embedding
    - trade history embedding
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()

        self.embed_dim = embed_dim

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        board_emb: torch.Tensor,
        self_emb: torch.Tensor,
        opponent_emb: torch.Tensor,
        trade_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Each input is expected shape: [B, D]
        """

        # stack into sequence
        x = torch.stack(
            [board_emb, self_emb, opponent_emb, trade_emb],
            dim=1,
        )  # [B, 4, D]

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        attn_out, _ = self.attn(Q, K, V)

        fused = attn_out.mean(dim=1)

        return self.output_proj(fused)