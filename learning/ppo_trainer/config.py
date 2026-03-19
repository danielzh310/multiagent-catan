"""
Build and configure the main Catan policy network.

This file assembles:
- state encoder
- action heads
- policy/value network
"""

from learning.networks.state_encoder import StateEncoder
from learning.networks.action_heads import ActionHeads
from learning.networks.policy_network import CatanPolicyNetwork


DEFAULT_TILE_FEATURE_DIM = 16
DEFAULT_CURRENT_PLAYER_DIM = 32
DEFAULT_OTHER_PLAYER_DIM = 24

DEFAULT_BOARD_MODEL_DIM = 64
DEFAULT_BOARD_NUM_HEADS = 4
DEFAULT_BOARD_NUM_LAYERS = 2
DEFAULT_BOARD_OUT_PROJ_DIM = 24

DEFAULT_DEV_CARD_EMBED_DIM = 16
DEFAULT_DEV_CARD_MODEL_DIM = 16
DEFAULT_PROJ_DEV_CARD_DIM = 24

DEFAULT_LATENT_DIM = 512
DEFAULT_USE_LSTM = False
DEFAULT_LSTM_DIM = 256

DEFAULT_NUM_ACTION_TYPES = 9
DEFAULT_NUM_VERTICES = 54
DEFAULT_NUM_CONNECTIONS = 72
DEFAULT_NUM_TILES = 19


def build_model(
    tile_feature_dim=DEFAULT_TILE_FEATURE_DIM,
    current_player_dim=DEFAULT_CURRENT_PLAYER_DIM,
    other_player_dim=DEFAULT_OTHER_PLAYER_DIM,
    board_model_dim=DEFAULT_BOARD_MODEL_DIM,
    board_num_heads=DEFAULT_BOARD_NUM_HEADS,
    board_num_layers=DEFAULT_BOARD_NUM_LAYERS,
    board_out_proj_dim=DEFAULT_BOARD_OUT_PROJ_DIM,
    dev_card_embed_dim=DEFAULT_DEV_CARD_EMBED_DIM,
    dev_card_model_dim=DEFAULT_DEV_CARD_MODEL_DIM,
    proj_dev_card_dim=DEFAULT_PROJ_DEV_CARD_DIM,
    latent_dim=DEFAULT_LATENT_DIM,
    use_lstm=DEFAULT_USE_LSTM,
    lstm_dim=DEFAULT_LSTM_DIM,
    num_action_types=DEFAULT_NUM_ACTION_TYPES,
    num_vertices=DEFAULT_NUM_VERTICES,
    num_connections=DEFAULT_NUM_CONNECTIONS,
    num_tiles=DEFAULT_NUM_TILES,
    normalize_values=True,
):
    """
    Build the full model used by training and evaluation.
    """

    state_encoder = StateEncoder(
        tile_feature_dim=tile_feature_dim,
        current_player_dim=current_player_dim,
        other_player_dim=other_player_dim,
        dev_card_embed_dim=dev_card_embed_dim,
        dev_card_model_dim=dev_card_model_dim,
        board_model_dim=board_model_dim,
        board_num_heads=board_num_heads,
        board_num_layers=board_num_layers,
        board_out_proj_dim=board_out_proj_dim,
        obs_out_dim=latent_dim,
        proj_dev_card_dim=proj_dev_card_dim,
    )

    action_heads = ActionHeads(
        input_dim=latent_dim + (lstm_dim if use_lstm else 0),
        num_action_types=num_action_types,
        num_vertices=num_vertices,
        num_connections=num_connections,
        num_tiles=num_tiles,
    )

    model = CatanPolicyNetwork(
        state_encoder=state_encoder,
        action_heads=action_heads,
        latent_dim=latent_dim,
        use_lstm=use_lstm,
        lstm_dim=lstm_dim,
        normalize_values=normalize_values,
    )

    return model