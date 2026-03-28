from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class TradeHistoryEncoder(nn.Module):
    """
    Encodes recent trade interactions into a fixed-size embedding.

    Expected sequence inputs:
    - proposer_ids:      [B, T]
    - target_ids:        [B, T]
    - response_types:    [B, T]
    - offers:            [B, T, 5]
    - requests:          [B, T, 5]
    - accepted_flags:    [B, T]
    - turn_numbers:      [B, T]
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.response_embed = nn.Embedding(4, embed_dim)
        self.player_embed = nn.Embedding(4, embed_dim)

        # proposer_emb (64) + target_emb (64) + response_emb (64)
        # offers (5) + requests (5) + accepted (1) + turns (1)
        # total = 64*3 + 12 = 204
        self.input_proj = nn.Linear(embed_dim * 3 + 12, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, seq_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        proposer = seq_dict["proposer_ids"].long()
        target = seq_dict["target_ids"].long()
        response = seq_dict["response_types"].long()

        offers = seq_dict["offers"].float()
        requests = seq_dict["requests"].float()
        accepted = seq_dict["accepted_flags"].float()
        turns = seq_dict["turn_numbers"].float()

        proposer_emb = self.player_embed(proposer)
        target_emb = self.player_embed(target)
        response_emb = self.response_embed(response)

        numeric = torch.cat(
            [
                offers,
                requests,
                accepted.unsqueeze(-1),
                turns.unsqueeze(-1),
            ],
            dim=-1,
        )

        x = torch.cat(
            [
                proposer_emb,
                target_emb,
                response_emb,
                numeric,
            ],
            dim=-1,
        )

        x = self.input_proj(x)

        _, h = self.gru(x)

        return h.squeeze(0)