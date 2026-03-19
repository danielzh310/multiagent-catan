import argparse
import os
import random
import sys

import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.constants import ActionType, PlayerId
from environment.catan_env import CatanEnv
from learning.networks.build_model import build_model
from learning.search_policy.policy import ForwardSearchPolicy


PLAYER_NAME_MAP = {
    "white": PlayerId.WHITE,
    "blue": PlayerId.BLUE,
    "orange": PlayerId.ORANGE,
    "red": PlayerId.RED,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--human", type=str, default="", choices=["", "white", "blue", "orange", "red"])
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--num-simulations", type=int, default=16)
    parser.add_argument("--rollout-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=500)
    return parser.parse_args()


def load_checkpoint_into_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        try:
            model.load_state_dict(checkpoint)
            return
        except Exception:
            pass

        for key in ["model_state_dict", "state_dict", "policy_state_dict", "actor_critic"]:
            if key in checkpoint:
                model.load_state_dict(checkpoint[key])
                return

    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) > 0:
        try:
            model.load_state_dict(checkpoint[0])
            return
        except Exception:
            pass

    raise ValueError(f"Could not load checkpoint format from: {checkpoint_path}")


def build_agent_mask(env, device):
    legal_actions = env.get_legal_actions()

    action_type_mask = torch.zeros(1, 9, dtype=torch.float32, device=device)
    settlement_mask = torch.ones(1, 54, dtype=torch.float32, device=device) * 1e-8
    road_mask = torch.ones(1, 72, dtype=torch.float32, device=device) * 1e-8
    city_mask = torch.ones(1, 54, dtype=torch.float32, device=device) * 1e-8
    robber_mask = torch.ones(1, 19, dtype=torch.float32, device=device) * 1e-8
    trade_mask = torch.ones(1, 2, dtype=torch.float32, device=device)

    for action in legal_actions:
        action_type = action["type"]
        action_type_mask[0, int(action_type)] = 1.0

        if action_type == ActionType.BUILD_SETTLEMENT:
            settlement_mask[0, action["vertex_id"]] = 1.0
        elif action_type == ActionType.BUILD_ROAD:
            road_mask[0, action["connection_id"]] = 1.0
        elif action_type == ActionType.BUILD_CITY:
            city_mask[0, action["vertex_id"]] = 1.0
        elif action_type == ActionType.MOVE_ROBBER:
            robber_mask[0, action["tile_id"]] = 1.0

    return {
        "action_type": action_type_mask,
        "settlement": settlement_mask,
        "road": road_mask,
        "city": city_mask,
        "robber": robber_mask,
        "trade": trade_mask,
    }


def obs_to_model_input(obs, device):
    tile_features = []
    for tile in obs["board"]["tiles"]:
        resource = tile["resource"]
        number = tile["number"] if tile["number"] is not None else 0
        has_robber = 1.0 if tile["has_robber"] else 0.0

        resource_one_hot = [0.0] * 6
        if resource is not None:
            resource_idx = int(resource)
            if 0 <= resource_idx < len(resource_one_hot):
                resource_one_hot[resource_idx] = 1.0

        base = resource_one_hot + [float(number), has_robber]
        padding = [0.0] * (16 - len(base))
        tile_features.append(base + padding)

    player = obs["player"]
    resources = player["resources"]
    current_player_vec = [
        float(resources.get(0, 0)),
        float(resources.get(1, 0)),
        float(resources.get(2, 0)),
        float(resources.get(3, 0)),
        float(resources.get(4, 0)),
        float(resources.get(5, 0)),
        float(player["victory_points"]),
        float(len(player["roads"])),
        float(len(player["dev_cards"])),
        float(player["dev_victory_points"]),
    ]
    while len(current_player_vec) < 32:
        current_player_vec.append(0.0)

    current_player = obs["game"]["current_player"]
    order = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]
    try:
        idx = order.index(current_player)
    except ValueError:
        idx = 0

    def other_player_vec(offset):
        target = order[(idx + offset + 1) % 4]
        summary = obs["players"][target]
        vec = [
            float(summary["victory_points"]),
            float(len(summary["roads"])),
            float(summary["dev_card_count"]),
            float(summary["dev_victory_points"]),
            float(summary["building_count"]),
        ]
        while len(vec) < 24:
            vec.append(0.0)
        return vec[:24]

    return {
        "tile_features": torch.tensor(tile_features, dtype=torch.float32, device=device).unsqueeze(0),
        "current_player_main": torch.tensor(current_player_vec[:32], dtype=torch.float32, device=device).unsqueeze(0),
        "current_player_hidden_dev": [torch.tensor([0], dtype=torch.long, device=device)],
        "current_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
        "next_player_main": torch.tensor(other_player_vec(0), dtype=torch.float32, device=device).unsqueeze(0),
        "next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
        "next_next_player_main": torch.tensor(other_player_vec(1), dtype=torch.float32, device=device).unsqueeze(0),
        "next_next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
        "next_next_next_player_main": torch.tensor(other_player_vec(2), dtype=torch.float32, device=device).unsqueeze(0),
        "next_next_next_player_played_dev": [torch.tensor([0], dtype=torch.long, device=device)],
    }


def decode_action(env, action_dict):
    legal_actions = env.get_legal_actions()
    if len(legal_actions) == 0:
        return None

    action_type_idx = int(action_dict["action_type"].squeeze().cpu().item())
    candidates = [action for action in legal_actions if int(action["type"]) == action_type_idx]

    if len(candidates) == 0:
        return legal_actions[-1]

    chosen = candidates[0]
    action_type = chosen["type"]

    if action_type == ActionType.BUILD_SETTLEMENT:
        vertex_id = int(action_dict["settlement"].squeeze().cpu().item())
        for candidate in candidates:
            if candidate.get("vertex_id") == vertex_id:
                return candidate
        return chosen

    if action_type == ActionType.BUILD_ROAD:
        connection_id = int(action_dict["road"].squeeze().cpu().item())
        for candidate in candidates:
            if candidate.get("connection_id") == connection_id:
                return candidate
        return chosen

    if action_type == ActionType.BUILD_CITY:
        vertex_id = int(action_dict["city"].squeeze().cpu().item())
        for candidate in candidates:
            if candidate.get("vertex_id") == vertex_id:
                return candidate
        return chosen

    if action_type == ActionType.MOVE_ROBBER:
        tile_id = int(action_dict["robber"].squeeze().cpu().item())
        for candidate in candidates:
            if candidate.get("tile_id") == tile_id:
                return candidate
        return chosen

    return chosen


def print_board_summary(obs):
    print("\nBoard summary:")
    print(f"  Turn: {obs['game']['turn_number']}")
    current_player = obs["game"]["current_player"]
    print(f"  Current player: {current_player.name if hasattr(current_player, 'name') else current_player}")
    print(f"  Dice: {obs['game']['dice']}")
    print(f"  Initial placement phase: {obs['game']['initial_placement_phase']}")
    print(f"  Robber pending: {obs['game']['robber_pending']}")


def print_player_summary(obs):
    print("\nPlayer summary:")
    for player_id, summary in obs["players"].items():
        name = player_id.name if hasattr(player_id, "name") else str(player_id)
        print(
            f"  {name}: VP={summary['victory_points']}, "
            f"roads={len(summary['roads'])}, "
            f"dev_cards={summary['dev_card_count']}, "
            f"buildings={summary['building_count']}"
        )


def print_legal_actions(legal_actions):
    print("\nLegal actions:")
    for idx, action in enumerate(legal_actions):
        print(f"  [{idx}] {action}")


def choose_human_action(env):
    legal_actions = env.get_legal_actions()

    if len(legal_actions) == 0:
        return None

    while True:
        print_legal_actions(legal_actions)
        raw = input("\nChoose action index: ").strip()

        try:
            idx = int(raw)
        except ValueError:
            print("Invalid input. Enter an integer.")
            continue

        if 0 <= idx < len(legal_actions):
            return legal_actions[idx]

        print("Index out of range.")


def choose_agent_action(env, model, device, hidden_state=None, use_search=False, search_policy=None):
    obs = env.get_observation()
    model_obs = obs_to_model_input(obs, device)
    action_mask = build_agent_mask(env, device)

    if use_search and search_policy is not None:
        action_dict = search_policy.act(
            env=env,
            obs=model_obs,
            hidden_state=hidden_state,
            action_mask=action_mask,
        )
    else:
        with torch.no_grad():
            _, action_dict, _, _, _ = model.act(
                obs=model_obs,
                action_masks=action_mask,
                hidden_state=hidden_state,
                done_mask=torch.ones(1, 1, device=device),
                deterministic=False,
            )

    return decode_action(env, action_dict)


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    env = CatanEnv(seed=args.seed)
    env.reset()

    model = build_model()
    model.to(device)
    model.eval()

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        load_checkpoint_into_model(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided. Using randomly initialized model.")

    search_policy = None
    if args.search:
        search_policy = ForwardSearchPolicy(
            base_policy=model,
            num_simulations=args.num_simulations,
            rollout_depth=args.rollout_depth,
            device=str(device),
        )

    human_player = PLAYER_NAME_MAP.get(args.human, None) if args.human else None

    print("\n=== Interactive Catan Launch ===")
    if human_player is None:
        print("Mode: all-agent autoplay")
    else:
        print(f"Mode: human controls {human_player.name}")
    print(f"Forward search enabled: {args.search}")
    print("================================")

    turn_counter = 0

    while env.engine.winner is None and turn_counter < args.max_turns:
        obs = env.get_observation()
        current_player = env.get_current_player_id()

        print_board_summary(obs)
        print_player_summary(obs)

        if human_player is not None and current_player == human_player:
            print(f"\nIt is your turn: {current_player.name}")
            action = choose_human_action(env)
        else:
            print(f"\nAgent turn: {current_player.name}")
            action = choose_agent_action(
                env=env,
                model=model,
                device=device,
                hidden_state=None,
                use_search=args.search,
                search_policy=search_policy,
            )
            print(f"Agent selected: {action}")

        if action is None:
            print("No valid action available. Ending.")
            break

        _, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        turn_counter += 1

        if done:
            break

    print("\n=== Game Over ===")
    if env.engine.winner is None:
        print("No winner determined before max turn limit.")
    else:
        print(f"Winner: {env.engine.winner.name}")
    print("=================")


if __name__ == "__main__":
    main()