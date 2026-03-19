"""
Player-state encoders.

This file contains:
- one encoder for the acting player
- one encoder for the other three players
- simple development-card sequence pooling using attention
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from learning.networks.attention import MultiHeadAttention
from learning.networks.network_utils import init_linear


class CurrentPlayerEncoder(nn.Module):
    """
    Encodes the current player's structured state.

    Inputs:
    - main player feature vector
    - hidden dev cards
    - played dev cards
    """

    def __init__(
        self,
        main_input_dim,
        dev_card_embed_dim,
        dev_card_model_dim,
        proj_dev_card_dim,
    ):
        super().__init__()

        self.main_fc = init_linear(nn.Linear(main_input_dim, 256), gain=1.414)

        self.dev_card_embedding = nn.Embedding(6, dev_card_embed_dim)
        self.hidden_card_attn = MultiHeadAttention(dev_card_model_dim, num_heads=4)
        self.played_card_attn = MultiHeadAttention(dev_card_model_dim, num_heads=4)

        self.hidden_proj = init_linear(nn.Linear(dev_card_model_dim, proj_dev_card_dim), gain=1.414)
        self.played_proj = init_linear(nn.Linear(dev_card_model_dim, proj_dev_card_dim), gain=1.414)

        self.final_fc = init_linear(nn.Linear(256 + 2 * proj_dev_card_dim, 128), gain=1.414)

        self.norm_main = nn.LayerNorm(256)
        self.norm_hidden = nn.LayerNorm(proj_dev_card_dim)
        self.norm_played = nn.LayerNorm(proj_dev_card_dim)
        self.norm_final = nn.LayerNorm(128)

        self.relu = nn.ReLU()

    def _encode_card_sequence(self, card_seq, attn_layer, proj_layer, norm_layer):
        """
        Encode one dev-card sequence.

        Supports:
        - padded tensor shape (B, T)
        - list of variable-length tensors
        """
        if isinstance(card_seq, list):
            lengths = [len(x) if len(x) > 0 else 1 for x in card_seq]
            padded = pad_sequence(card_seq, batch_first=True).long()
        else:
            padded = card_seq.long()
            lengths = (padded.shape[-1] - (padded == 0).sum(dim=-1)).cpu().tolist()
            lengths = [max(1, int(x)) for x in lengths]

        embeds = self.dev_card_embedding(padded)

        batch_size, seq_len, _ = embeds.shape
        mask = torch.zeros(batch_size, 1, 1, seq_len, device=embeds.device)

        for b in range(batch_size):
            mask[b, :, :, :lengths[b]] = 1.0

        reps = attn_layer(embeds, embeds, embeds, mask=mask)

        token_mask = mask.squeeze(1).transpose(-1, -2).bool()
        reps = reps.masked_fill(~token_mask.expand_as(reps), 0.0)

        pooled = reps.sum(dim=1)
        pooled = proj_layer(pooled)
        pooled = self.relu(norm_layer(pooled))
        return pooled

    def forward(self, main_input, hidden_dev_cards, played_dev_cards):
        main_out = self.main_fc(main_input)
        main_out = self.relu(self.norm_main(main_out))

        hidden_out = self._encode_card_sequence(
            hidden_dev_cards,
            self.hidden_card_attn,
            self.hidden_proj,
            self.norm_hidden,
        )

        played_out = self._encode_card_sequence(
            played_dev_cards,
            self.played_card_attn,
            self.played_proj,
            self.norm_played,
        )

        final_input = torch.cat([main_out, hidden_out, played_out], dim=-1)
        final_out = self.final_fc(final_input)
        final_out = self.relu(self.norm_final(final_out))
        return final_out


class OtherPlayerEncoder(nn.Module):
    """
    Encodes a non-acting player's visible state.

    Inputs:
    - main player feature vector
    - played dev cards
    """

    def __init__(
        self,
        main_input_dim,
        dev_card_embed_dim,
        dev_card_model_dim,
        proj_dev_card_dim,
    ):
        super().__init__()

        self.main_fc = init_linear(nn.Linear(main_input_dim, 256), gain=1.414)

        self.dev_card_embedding = nn.Embedding(6, dev_card_embed_dim)
        self.played_card_attn = MultiHeadAttention(dev_card_model_dim, num_heads=4)
        self.played_proj = init_linear(nn.Linear(dev_card_model_dim, proj_dev_card_dim), gain=1.414)

        self.final_fc = init_linear(nn.Linear(256 + proj_dev_card_dim, 128), gain=1.414)

        self.norm_main = nn.LayerNorm(256)
        self.norm_played = nn.LayerNorm(proj_dev_card_dim)
        self.norm_final = nn.LayerNorm(128)

        self.relu = nn.ReLU()

    def _encode_played_cards(self, played_dev_cards):
        if isinstance(played_dev_cards, list):
            lengths = [len(x) if len(x) > 0 else 1 for x in played_dev_cards]
            padded = pad_sequence(played_dev_cards, batch_first=True).long()
        else:
            padded = played_dev_cards.long()
            lengths = (padded.shape[-1] - (padded == 0).sum(dim=-1)).cpu().tolist()
            lengths = [max(1, int(x)) for x in lengths]

        embeds = self.dev_card_embedding(padded)

        batch_size, seq_len, _ = embeds.shape
        mask = torch.zeros(batch_size, 1, 1, seq_len, device=embeds.device)

        for b in range(batch_size):
            mask[b, :, :, :lengths[b]] = 1.0

        reps = self.played_card_attn(embeds, embeds, embeds, mask=mask)

        token_mask = mask.squeeze(1).transpose(-1, -2).bool()
        reps = reps.masked_fill(~token_mask.expand_as(reps), 0.0)

        pooled = reps.sum(dim=1)
        pooled = self.played_proj(pooled)
        pooled = self.relu(self.norm_played(pooled))
        return pooled

    def forward(self, main_input, played_dev_cards):
        main_out = self.main_fc(main_input)
        main_out = self.relu(self.norm_main(main_out))

        played_out = self._encode_played_cards(played_dev_cards)

        final_input = torch.cat([main_out, played_out], dim=-1)
        final_out = self.final_fc(final_input)
        final_out = self.relu(self.norm_final(final_out))
        return final_out