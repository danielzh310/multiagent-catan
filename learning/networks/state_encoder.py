"""
State encoder for the full Catan observation.

This module combines:
- board / tile encoding
- current player encoding
- three opponent encodings

The output is a single latent vector used by the policy and value heads.
"""

import torch
import torch.nn as nn

from learning.networks.board_encoder import BoardEncoder
from learning.networks.player_encoder import CurrentPlayerEncoder, OtherPlayerEncoder
from learning.networks.network_utils import init_linear


class StateEncoder(nn.Module):
    """
    Encodes a structured Catan observation into one feature vector.
    """

    def __init__(
        self,
        tile_feature_dim,
        current_player_dim,
        other_player_dim,
        dev_card_embed_dim=16,
        dev_card_model_dim=16,
        board_model_dim=64,
        board_num_heads=4,
        board_num_layers=2,
        board_out_proj_dim=24,
        obs_out_dim=512,
        proj_dev_card_dim=24,
    ):
        super().__init__()

        self.board_encoder = BoardEncoder(
            tile_feature_dim=tile_feature_dim,
            model_dim=board_model_dim,
            num_heads=board_num_heads,
            num_layers=board_num_layers,
            out_proj_dim=board_out_proj_dim,
        )

        self.current_player_encoder = CurrentPlayerEncoder(
            main_input_dim=current_player_dim,
            dev_card_embed_dim=dev_card_embed_dim,
            dev_card_model_dim=dev_card_model_dim,
            proj_dev_card_dim=proj_dev_card_dim,
        )

        self.other_player_encoder = OtherPlayerEncoder(
            main_input_dim=other_player_dim,
            dev_card_embed_dim=dev_card_embed_dim,
            dev_card_model_dim=dev_card_model_dim,
            proj_dev_card_dim=proj_dev_card_dim,
        )

        # board contributes num_tiles * board_out_proj_dim
        # players contribute 4 * 128
        self.final_fc = init_linear(
            nn.Linear(19 * board_out_proj_dim + 4 * 128, obs_out_dim),
            gain=1.414,
        )

        self.norm = nn.LayerNorm(obs_out_dim)
        self.relu = nn.ReLU()

    def forward(self, obs):
        """
        Expected obs keys:
        - tile_features
        - current_player_main
        - current_player_hidden_dev
        - current_player_played_dev
        - next_player_main
        - next_player_played_dev
        - next_next_player_main
        - next_next_player_played_dev
        - next_next_next_player_main
        - next_next_next_player_played_dev
        """
        board_out = self.board_encoder(obs["tile_features"])

        current_out = self.current_player_encoder(
            obs["current_player_main"],
            obs["current_player_hidden_dev"],
            obs["current_player_played_dev"],
        )

        next_out = self.other_player_encoder(
            obs["next_player_main"],
            obs["next_player_played_dev"],
        )

        next_next_out = self.other_player_encoder(
            obs["next_next_player_main"],
            obs["next_next_player_played_dev"],
        )

        next_next_next_out = self.other_player_encoder(
            obs["next_next_next_player_main"],
            obs["next_next_next_player_played_dev"],
        )

        final_input = torch.cat(
            [board_out, current_out, next_out, next_next_out, next_next_next_out],
            dim=-1,
        )

        out = self.final_fc(final_input)
        out = self.relu(self.norm(out))
        return out