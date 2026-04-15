# python learning/dqn/train.py --num-updates 50000 --hidden-dim 192

from __future__ import annotations

import argparse
import os
import sys
import random
from collections import deque
from typing import Dict, List, Any

import torch

# allow running from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.constants import Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv
from learning.dqn.dqn_policy import DQNBaselinePolicy
from learning.dqn.dqn_trainer import DQNTrainer
from learning.dqn.epsilon_scheduler import EpsilonScheduler
from learning.dqn.replay_buffer import ReplayBuffer

# Constants for observation/action feature dimensions
MAX_GAMEPLAY_ACTIONS = 256
GAMEPLAY_FEATURE_DIM = 40
MAX_TRADE_ACTIONS = 128
TRADE_FEATURE_DIM = 32


def save_checkpoint(trainer: DQNTrainer, save_dir: str, step: int, prefix: str = "dqn_checkpoint") -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{prefix}_{step}.pt")
    trainer.save(path)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DQN baseline training for Catan")
    parser.add_argument("--num-updates", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--board-dim", type=int, default=64)
    parser.add_argument("--self-dim", type=int, default=64)
    parser.add_argument("--opponent-dim", type=int, default=64)
    parser.add_argument("--resources", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=1000, help="Checkpoint save frequency")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    return parser


def _resource_slot(resource_value: Any) -> int:
    if resource_value is None:
        return -1
    try:
        return int(Resource(resource_value))
    except (ValueError, TypeError):
        return -1


def _dev_card_slot(card_value: Any) -> int:
    if card_value is None:
        return -1
    try:
        return int(card_value)
    except (ValueError, TypeError):
        return -1


def _encode_gameplay_action(action: Dict[str, Any], env: CatanEnv) -> List[float]:
    features = [0.0] * GAMEPLAY_FEATURE_DIM
    action_type = action.get("type", "")
    action_types = {
        "build_settlement": 0, "build_road": 1, "build_city": 2, "buy_dev_card": 3,
        "play_dev_card": 4, "bank_trade": 5, "move_robber": 6, "discard_cards": 7,
        "end_main_action": 8, "end_turn": 9, "roll": 10, "skip_trade": 11,
    }
    action_type_idx = action_types.get(action_type)
    if action_type_idx is not None and action_type_idx < 12:
        features[action_type_idx] = 1.0

    current_player = env.get_current_player_id()
    player = env.engine.players[current_player]
    victory_points = float(player.update_victory_points())

    features[12] = victory_points / 10.0
    features[13] = float(player.n_settlements) / 5.0
    features[14] = float(player.n_cities) / 4.0
    features[15] = float(player.n_roads) / 15.0

    if "vertex" in action: features[16] = float(action["vertex"]) / 64.0
    if "connection" in action: features[17] = float(action["connection"]) / 128.0
    if "tile" in action: features[18] = float(action["tile"]) / 19.0
    if "connection_1" in action and action["connection_1"] is not None: features[19] = float(action["connection_1"]) / 128.0
    if "connection_2" in action and action["connection_2"] is not None: features[20] = float(action["connection_2"]) / 128.0

    if (give_slot := _resource_slot(action.get("give"))) >= 0: features[21 + give_slot] = 1.0
    if (receive_slot := _resource_slot(action.get("receive"))) >= 0: features[26 + receive_slot] = 1.0
    if (resource_slot := _resource_slot(action.get("resource"))) >= 0: features[31] = float(resource_slot + 1) / 5.0
    if (resource_1_slot := _resource_slot(action.get("resource_1"))) >= 0: features[32] = float(resource_1_slot + 1) / 5.0
    if (resource_2_slot := _resource_slot(action.get("resource_2"))) >= 0: features[33] = float(resource_2_slot + 1) / 5.0
    if (card_slot := _dev_card_slot(action.get("card"))) >= 0: features[34] = float(card_slot + 1) / 5.0
    if (rate := action.get("rate")) is not None: features[35] = float(rate) / 4.0
    if (required := action.get("required")) is not None: features[36] = float(required) / 8.0
    if isinstance(resources_to_discard := action.get("resources"), dict):
        features[37] = sum(float(v) for v in resources_to_discard.values()) / 8.0
        features[38] = sum(1 for v in resources_to_discard.values() if int(v) > 0) / 5.0
    if action.get("type") == "play_dev_card": features[39] = 1.0

    return features


def _build_gameplay_candidates(env: CatanEnv, legal_actions: List[Dict[str, Any]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.zeros((1, MAX_GAMEPLAY_ACTIONS, GAMEPLAY_FEATURE_DIM), dtype=torch.float32, device=device)
    mask = torch.zeros((1, MAX_GAMEPLAY_ACTIONS), dtype=torch.bool, device=device)
    capped_actions = legal_actions[:MAX_GAMEPLAY_ACTIONS]
    for idx, action in enumerate(capped_actions):
        candidates[0, idx] = torch.tensor(_encode_gameplay_action(action, env), dtype=torch.float32, device=device)
        mask[0, idx] = True
    if not capped_actions: mask[0, 0] = True
    return candidates, mask


def _encode_trade_action(action: Dict[str, Any], env: CatanEnv) -> List[float]:
    features = [0.0] * TRADE_FEATURE_DIM
    action_type = action.get("type", "")
    action_types = {"skip_trade": 0, "propose_trade": 1, "accept_trade": 2, "reject_trade": 3, "counter_trade": 4}
    if (action_type_idx := action_types.get(action_type)) is not None: features[action_type_idx] = 1.0

    player = env.engine.players[env.get_current_player_id()]
    features[5] = float(player.update_victory_points()) / 10.0
    features[6] = float(sum(int(v) for v in player.resources.values())) / 20.0

    if (pending := env.engine.trade_manager.get_pending_trade()) is not None:
        features[7] = 1.0
        features[8] = float(pending.counter_count) / 3.0
        features[9] = float(int(pending.proposer)) / 3.0
        features[10] = float(int(pending.target)) / 3.0

    if (target := action.get("target")) is not None: features[11 + int(target)] = 1.0

    offer = action.get("offer") or action.get("counter_offer") or {}
    request = action.get("request") or action.get("counter_request") or {}
    for res, amt in offer.items():
        if (slot := _resource_slot(res)) >= 0: features[15 + slot] = float(amt)
    for res, amt in request.items():
        if (slot := _resource_slot(res)) >= 0: features[20 + slot] = float(amt)

    features[25] = float(sum(int(v) for v in offer.values())) / 4.0
    features[26] = float(sum(int(v) for v in request.values())) / 4.0

    if (response_type := action.get("response_type", "")) == "accept": features[27] = 1.0
    elif response_type == "reject": features[28] = 1.0
    elif response_type == "counter": features[29] = 1.0

    if action_type == "counter_trade": features[30] = 1.0
    if action_type == "propose_trade": features[31] = 1.0

    return features


def _build_trade_candidates(env: CatanEnv, legal_actions: List[Dict[str, Any]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    candidates = torch.zeros((1, MAX_TRADE_ACTIONS, TRADE_FEATURE_DIM), dtype=torch.float32, device=device)
    mask = torch.zeros((1, MAX_TRADE_ACTIONS), dtype=torch.bool, device=device)
    capped_actions = legal_actions[:MAX_TRADE_ACTIONS]
    for idx, action in enumerate(capped_actions):
        candidates[0, idx] = torch.tensor(_encode_trade_action(action, env), dtype=torch.float32, device=device)
        mask[0, idx] = True
    if not capped_actions: mask[0, 0] = True
    return candidates, mask


def build_policy_obs(env: CatanEnv, raw_obs: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
    player_state = raw_obs["player"]
    other_players = [v for k, v in raw_obs["players"].items() if k != raw_obs["game"]["current_player"]]

    def to_vec(state: dict) -> List[float]:
        roads_val = state.get("roads", 0)
        if isinstance(roads_val, list):
            roads_val = len(roads_val)

        dev_cards_val = state.get("dev_cards", 0)
        if isinstance(dev_cards_val, list):
            dev_cards_val = len(dev_cards_val)

        vec = [
            float(state.get("resources", {}).get("WOOD", 0)), float(state.get("resources", {}).get("BRICK", 0)),
            float(state.get("resources", {}).get("SHEEP", 0)), float(state.get("resources", {}).get("WHEAT", 0)),
            float(state.get("resources", {}).get("ORE", 0)), float(state.get("victory_points", 0)),
            float(state.get("num_settlements", 0)), float(state.get("num_cities", 0)),
            float(roads_val), float(state.get("bonus_vp", 0)),
            float(state.get("dev_victory_points", 0)), float(dev_cards_val),
            float(state.get("played_knights", 0)), float(state.get("revealed_vp_cards", 0)),
        ]
        return (vec + [0.0] * 64)[:64]

    self_vec = torch.tensor([to_vec(player_state)], dtype=torch.float32, device=device)

    op_vec = [0.0] * 64
    if other_players:
        sum_vec = [sum(col) for col in zip(*(to_vec(opp) for opp in other_players))]
        op_vec = [x / len(other_players) for x in sum_vec]

    board_vec = torch.zeros((1, 64), dtype=torch.float32, device=device)
    game = raw_obs.get("game", {})
    board_vec[0, 0] = float(game.get("turn_number", 0))
    board_vec[0, 1] = float(int(env.get_current_player_id()))
    board_vec[0, 2] = float(env.get_phase().value)
    board_vec[0, 3] = 1.0 if game.get("enable_trading", True) else 0.0
    last_roll = game.get("last_roll")
    board_vec[0, 4] = float(last_roll if last_roll is not None else 0.0)
    board_vec[0, 5] = 1.0 if game.get("robber_pending", False) else 0.0
    if (pending_trade := raw_obs.get("trade")) is not None:
        board_vec[0, 6] = 1.0
        board_vec[0, 7] = float(pending_trade.counter_count)
    if (robber_event := game.get("last_robber_event")) is not None:
        board_vec[0, 8] = 1.0 if robber_event.get("rolled_seven", False) else 0.0
        board_vec[0, 9] = 1.0 if robber_event.get("stolen_from") is not None else 0.0
        board_vec[0, 10] = sum(sum(res.values()) for _, res in robber_event.get("discarded", {}).items())

    legal_actions = raw_obs.get("legal_actions", [])
    board_vec[0, 11] = float(len(legal_actions))

    gameplay_candidates, gameplay_mask = _build_gameplay_candidates(env, legal_actions, device)
    trade_candidates, trade_mask = _build_trade_candidates(env, legal_actions, device)

    return {
        "board": board_vec,
        "self": self_vec,
        "opponent": torch.tensor([op_vec], dtype=torch.float32, device=device),
        "gameplay_candidates": gameplay_candidates,
        "gameplay_mask": gameplay_mask,
        "trade_candidates": trade_candidates,
        "trade_mask": trade_mask,
    }


def _resolve_discard_action(action: Dict[str, Any]) -> Dict[str, Any]:
    required = int(action.get("required", 0))
    available = action.get("available", {})
    ordered_resources = sorted(available.items(), key=lambda item: (-int(item[1]), item[0]))
    resources_to_discard: Dict[Resource, int] = {r: 0 for r in Resource}
    remaining = required
    for resource_name, count in ordered_resources:
        if remaining <= 0: break
        take = min(int(count), remaining)
        if take > 0:
            resources_to_discard[Resource[resource_name]] = take
            remaining -= take
    if remaining > 0: return action
    resolved = dict(action)
    resolved["resources"] = resources_to_discard
    return resolved


def decode_gameplay_action(action_idx: int, env: CatanEnv) -> dict:
    legal_actions = env.get_legal_actions()
    if env.get_phase() == TurnPhase.END_TURN: return {"type": "end_turn"}
    if not legal_actions: return {"type": "end_main_action"}
    mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
    chosen_action = legal_actions[mapped_idx]
    if chosen_action.get("type") == "discard_cards" and "resources" not in chosen_action:
        return _resolve_discard_action(chosen_action)
    return chosen_action


def decode_trade_action(action_idx: int, env: CatanEnv) -> dict:
    legal_actions = env.get_legal_actions()
    if not legal_actions:
        return {"type": "reject_trade", "response_type": "reject"} if env.get_phase() == TurnPhase.TRADE_RESPOND else {"type": "skip_trade"}
    mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
    return legal_actions[mapped_idx]


def stack_items(items: List[Any]) -> Any:
    if isinstance(items[0], dict):
        res = {}
        for k in items[0].keys():
            res[k] = stack_items([item[k] for item in items])
        return res
    else:
        return torch.cat(items, dim=0)


def collect_rollout(
    env,
    trainer: DQNTrainer,
    epsilon: float,
    device: str,
) -> List[Dict[str, Any]]:
    """Collects experience by running the policy in the environment."""
    rollout_data = []
    obs = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 1000:  # Safety limit
        # Build policy observation from raw environment observation
        obs_tensor = build_policy_obs(env, obs, device)

        # Determine phase
        phase = "trade" if env.get_phase() in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND) else "gameplay"

        # Get action from policy
        with torch.no_grad():
            action_idx_dict = trainer.policy.act(obs_tensor, phase, epsilon)
            if phase == "gameplay":
                gameplay_action_idx = int(action_idx_dict["gameplay_action"].detach().cpu().item())
                env_action = decode_gameplay_action(gameplay_action_idx, env)
            else:
                trade_action_idx = int(action_idx_dict["trade_action"].detach().cpu().item())
                env_action = decode_trade_action(trade_action_idx, env)

        # Take action in environment
        next_obs, reward, done, info = env.step(env_action)
        next_obs_tensor = build_policy_obs(env, next_obs, device)

        # Store transition
        transition = {
            "obs": obs_tensor,
            "action": action_idx_dict,
            "reward": reward,
            "next_obs": next_obs_tensor,
            "done": done,
            "phase": phase,
        }
        rollout_data.append(transition)

        obs = next_obs
        step_count += 1
        if done:
            obs = env.reset()

    return rollout_data


def train(args: argparse.Namespace) -> None:
    device = args.device

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize DQN trainer with Catan-compatible dimensions
    trainer = DQNTrainer(
        board_dim=args.board_dim,
        self_dim=args.self_dim,
        opponent_dim=args.opponent_dim,
        hidden_dim=args.hidden_dim,
        resources=args.resources,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        target_update_freq=args.target_update_freq,
    )

    # Resume from checkpoint if specified
    start_update = 0
    if args.resume_from:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.resume_from)
        if os.path.exists(checkpoint_path):
            trainer.load(checkpoint_path)
            start_update = int(args.resume_from.split('_')[-1].split('.')[0])
            print(f"Resumed from update {start_update}")
        else:
            print(f"Checkpoint {checkpoint_path} not found, starting from scratch")

    replay_buffer = ReplayBuffer(args.buffer_size)
    epsilon_scheduler = EpsilonScheduler()

    loss_window = deque(maxlen=100)

    print(
        f"Starting DQN baseline training for {args.num_updates} updates "
        f"(batch_size={args.batch_size}, buffer_size={args.buffer_size}, hidden_dim={args.hidden_dim}, seed={args.seed})"
    )

    env = CatanEnv()

    update = start_update
    try:
        while update < args.num_updates:
            # Collect rollout
            epsilon = epsilon_scheduler.value(update)
            rollout = collect_rollout(env, trainer, epsilon, device)

            # Add to replay buffer
            for transition in rollout:
                replay_buffer.add(
                    obs=transition["obs"],
                    action=transition["action"],
                    reward=transition["reward"],
                    next_obs=transition["next_obs"],
                    done=transition["done"],
                    phase=transition["phase"],
                )

            # Update if buffer has enough data
            if len(replay_buffer) >= args.batch_size:
                batch = replay_buffer.sample(args.batch_size)

                # Convert batch to proper format for trainer
                formatted_batch = {
                    "obs": stack_items([item.obs for item in batch]),
                    "actions": {
                        phase: stack_items([item.action[f"{phase}_action"] for item in batch if item.phase == phase])
                        for phase in ["gameplay", "trade"] if any(item.phase == phase for item in batch)
                    },
                    "rewards": torch.tensor([item.reward for item in batch], dtype=torch.float32),
                    "next_obs": stack_items([item.next_obs for item in batch]),
                    "dones": torch.tensor([item.done for item in batch], dtype=torch.float32),
                    "phases": [item.phase for item in batch],
                }

                metrics = trainer.update(formatted_batch)
                loss_window.append(metrics["td_loss"])

                avg_loss = sum(loss_window) / len(loss_window)

                if update > 0 and update % 100 == 0:
                    print(f"Update {update}: td_loss={avg_loss:.4f}, q_mean={metrics['q_mean']:.4f}")

            update += 1

            if update > 0 and update % args.save_freq == 0:
                path = save_checkpoint(trainer, args.checkpoint_dir, update)
                print(f"saved checkpoint -> {path}")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Save final checkpoint
        final_checkpoint_path = os.path.join(args.checkpoint_dir, f"dqn_final_{update}.pt")
        trainer.save(final_checkpoint_path)
        print(f"Saved final checkpoint: {final_checkpoint_path}")

    print("DQN baseline training complete")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    train(args)
