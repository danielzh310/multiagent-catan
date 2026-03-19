"""
Simulation worker for forward search.

Runs rollout simulations starting from a candidate action.
"""

import copy
import torch


def simulate_action_sequence(env, action, policy, depth=5, device="cpu"):
    """
    Simulate future trajectory after taking an action.

    Returns:
        cumulative reward estimate
    """

    sim_env = copy.deepcopy(env)

    total_reward = 0.0

    # Apply initial action
    decoded_action = decode_action(sim_env, action)
    obs, reward, done, _ = sim_env.step(decoded_action)

    total_reward += float(reward)

    if done:
        return total_reward

    hidden_state = None

    for _ in range(depth):
        obs_tensor = convert_obs(obs, device)

        action_mask = build_action_mask(sim_env)

        with torch.no_grad():
            _, action_outputs, _ = policy.forward(
                obs=obs_tensor,
                action_masks=action_mask,
                hidden_state=hidden_state,
                done_mask=torch.ones(1, 1, device=device),
            )

            next_action = policy.action_heads.sample(action_outputs)

        decoded = decode_action(sim_env, next_action)
        obs, reward, done, _ = sim_env.step(decoded)

        total_reward += float(reward)

        if done:
            break

    return total_reward

def convert_obs(obs, device):
    # minimal placeholder (same shape as training)
    return {
        "tile_features": torch.zeros(1, 19, 16, device=device),
        "current_player_main": torch.zeros(1, 32, device=device),
        "current_player_hidden_dev": [torch.tensor([0], device=device)],
        "current_player_played_dev": [torch.tensor([0], device=device)],
        "next_player_main": torch.zeros(1, 24, device=device),
        "next_player_played_dev": [torch.tensor([0], device=device)],
        "next_next_player_main": torch.zeros(1, 24, device=device),
        "next_next_player_played_dev": [torch.tensor([0], device=device)],
        "next_next_next_player_main": torch.zeros(1, 24, device=device),
        "next_next_next_player_played_dev": [torch.tensor([0], device=device)],
    }


def build_action_mask(env):
    legal_actions = env.get_legal_actions()

    mask = {
        "action_type": torch.ones(1, 9),
        "settlement": torch.ones(1, 54),
        "road": torch.ones(1, 72),
        "city": torch.ones(1, 54),
        "robber": torch.ones(1, 19),
        "trade": torch.ones(1, 2),
    }

    return mask


def decode_action(env, action_dict):
    legal = env.get_legal_actions()

    if len(legal) == 0:
        return None

    return legal[0]