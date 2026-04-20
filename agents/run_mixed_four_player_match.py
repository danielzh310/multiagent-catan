from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.random_no_trade_baseline.agent import RandomNoTradeBaselineAgent
from core.constants import PlayerId
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv
from learning.dqn.dqn_policy import DQNBaselinePolicy
from learning.ppo.ppo_policy import PPOPolicy
from learning.unified.unified_policy import UnifiedPolicy
from play_checkpoint import build_policy_obs, build_ppo_obs


PLAYER_ORDER = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]


@dataclass
class SeatStats:
    player_id: PlayerId
    agent_name: str
    wins: int = 0
    total_vp: int = 0
    decisions: int = 0
    gameplay_decisions: int = 0
    trade_decisions: int = 0
    fallback_decisions: int = 0


class BaseSeatAgent:
    def __init__(self, name: str):
        self.name = name
        self.fallback_count = 0

    def select_action(self, env: CatanEnv) -> Optional[dict[str, Any]]:
        raise NotImplementedError


class RandomSeatAgent(BaseSeatAgent):
    def __init__(self, seed: int, name: str = "random_no_trade"):
        super().__init__(name)
        self.policy = RandomNoTradeBaselineAgent(seed=seed)

    def select_action(self, env: CatanEnv) -> Optional[dict[str, Any]]:
        return self.policy.select_action(env.get_legal_actions())


class CandidatePolicySeatAgent(BaseSeatAgent):
    def __init__(self, name: str, policy: torch.nn.Module, device: str, deterministic: bool):
        super().__init__(name)
        self.policy = policy.to(device)
        self.policy.eval()
        self.device = device
        self.deterministic = deterministic

    def _fallback(self, env: CatanEnv) -> Optional[dict[str, Any]]:
        self.fallback_count += 1
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None
        return random.choice(legal_actions)

    def _action_dict_from_result(self, result: Any) -> Optional[dict[str, Any]]:
        if isinstance(result, dict):
            return result
        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, dict):
                    return item
        return None

    def _unwrap(self, action_dict: dict[str, Any]) -> dict[str, int]:
        return env_unwrap_action_dict(action_dict)

    def select_action(self, env: CatanEnv) -> Optional[dict[str, Any]]:
        phase = env.get_phase()
        if phase == TurnPhase.ROLL:
            return None

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        policy_phase = env._phase_name(phase)
        if policy_phase == "auto":
            return None

        raw_obs = env.get_observation()
        raw_obs["legal_actions"] = legal_actions
        obs = (
            build_ppo_obs(env, raw_obs, self.device)
            if isinstance(self.policy, PPOPolicy)
            else build_policy_obs(env, raw_obs, self.device)
        )
        obs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in obs.items()
        }

        try:
            if isinstance(self.policy, DQNBaselinePolicy):
                result = self.policy.act(obs, policy_phase, epsilon=0.0)
            else:
                result = self.policy.act(obs, policy_phase, deterministic=self.deterministic)
        except Exception:
            return self._fallback(env)

        action_dict = self._action_dict_from_result(result)
        if not isinstance(action_dict, dict):
            return self._fallback(env)

        unwrapped = self._unwrap(action_dict)
        if policy_phase == "gameplay":
            idx = unwrapped.get("gameplay_action", unwrapped.get("action", -1))
            return env._decode_gameplay(int(idx), legal_actions, phase)

        idx = unwrapped.get("trade_action", unwrapped.get("action", -1))
        return env._decode_trade(int(idx), legal_actions, phase)


class HybridGameplaySeatAgent(CandidatePolicySeatAgent):
    """
    Hybrid smoke adapter: uses the hybrid DQN gameplay checkpoint, and declines
    player trades. This keeps the 4-way smoke game runnable without reimplementing
    the hybrid trade action-head decoder here.
    """

    def __init__(self, checkpoint: str, device: str, deterministic: bool):
        policy = DQNBaselinePolicy(hidden_dim=256, device=device)
        if checkpoint:
            checkpoint_data = torch.load(checkpoint, map_location=device)
            policy.load_state_dict(checkpoint_data["dqn"]["q_network"], strict=False)
        super().__init__("hybrid_gameplay_no_trade", policy, device, deterministic)
        self.no_trade = RandomNoTradeBaselineAgent(seed=9001)

    def select_action(self, env: CatanEnv) -> Optional[dict[str, Any]]:
        if env._phase_name(env.get_phase()) == "trade":
            return self.no_trade.select_action(env.get_legal_actions())
        return super().select_action(env)


def env_unwrap_action_dict(action_dict: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, value in action_dict.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            if value.size == 0:
                continue
            result[key] = int(value.reshape(-1)[0])
        elif isinstance(value, list):
            result[key] = int(value[0]) if value else -1
        else:
            result[key] = int(value)
    return result


def load_state_dict(policy: torch.nn.Module, checkpoint_path: str, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    own_state = policy.state_dict()

    def _load_compatible(state_dict: dict[str, Any]) -> bool:
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in own_state and hasattr(value, "shape") and own_state[key].shape == value.shape
        }
        if not compatible:
            return False
        policy.load_state_dict(compatible, strict=False)
        skipped = len(state_dict) - len(compatible)
        if skipped:
            print(
                f"Loaded {len(compatible)} tensors from {checkpoint_path}; skipped {skipped} incompatible tensors.",
                file=sys.stderr,
            )
        return True

    if isinstance(checkpoint, dict):
        for key in [
            "policy",
            "policy_state_dict",
            "model_state_dict",
            "state_dict",
            "gameplay_state_dict",
        ]:
            if key in checkpoint:
                if _load_compatible(checkpoint[key]):
                    return
        try:
            if _load_compatible(checkpoint):
                return
        except Exception:
            pass
    raise ValueError(f"Could not load checkpoint into {policy.__class__.__name__}: {checkpoint_path}")


def build_agents(args: argparse.Namespace) -> dict[PlayerId, BaseSeatAgent]:
    device = args.device
    agents: dict[PlayerId, BaseSeatAgent] = {
        PlayerId.WHITE: RandomSeatAgent(seed=args.seed + 10),
    }

    unified = UnifiedPolicy()
    if args.unified_checkpoint:
        load_state_dict(unified, args.unified_checkpoint, device)
    agents[PlayerId.BLUE] = CandidatePolicySeatAgent("unified", unified, device, args.deterministic)

    dqn = DQNBaselinePolicy(device=device)
    if args.dqn_checkpoint:
        load_state_dict(dqn, args.dqn_checkpoint, device)
    agents[PlayerId.ORANGE] = CandidatePolicySeatAgent("dqn", dqn, device, args.deterministic)

    if args.use_hybrid and args.hybrid_checkpoint:
        agents[PlayerId.RED] = HybridGameplaySeatAgent(args.hybrid_checkpoint, device, args.deterministic)
    else:
        ppo = PPOPolicy()
        if args.ppo_checkpoint:
            try:
                load_state_dict(ppo, args.ppo_checkpoint, device)
            except ValueError as exc:
                print(f"Warning: {exc}. Using randomly initialized raw PPO seat.", file=sys.stderr)
        agents[PlayerId.RED] = CandidatePolicySeatAgent("ppo", ppo, device, args.deterministic)

    return agents


def run_one_game(
    game_idx: int,
    args: argparse.Namespace,
    agents: dict[PlayerId, BaseSeatAgent],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = CatanEnv(seed=args.seed + game_idx, enable_trading=args.enable_trading, max_steps=args.max_steps)
    obs = env.reset()
    done = False
    steps = 0
    truncated = False
    step_rows: list[dict[str, Any]] = []

    while not done and steps < args.max_steps:
        player_id = env.get_current_player_id()
        agent = agents[player_id]
        phase = env.get_phase()
        before_fallbacks = agent.fallback_count
        action = agent.select_action(env)
        action_type = action.get("type") if action else "auto"

        obs, reward, done, info = env.step(action)
        truncated = bool(info.get("truncated", False))

        step_rows.append(
            {
                "game": game_idx,
                "step": steps,
                "player": player_id.name,
                "agent": agent.name,
                "phase": phase.name,
                "action": action_type,
                "reward": float(reward),
                "fallback": int(agent.fallback_count > before_fallbacks),
            }
        )
        steps += 1

    player_summaries = {}
    for player_id in PLAYER_ORDER:
        state = env.engine.players[player_id]
        player_summaries[player_id.name] = {
            "agent": agents[player_id].name,
            "victory_points": int(state.update_victory_points()),
            "played_knights": int(state.played_knights),
            "dev_cards_held": int(len(state.dev_cards) + len(state.new_dev_cards)),
            "hidden_vp_cards": int(state.hidden_vp_cards),
            "settlements": int(state.n_settlements),
            "cities": int(state.n_cities),
            "roads": int(state.n_roads),
            "longest_road_length": int(state.longest_road_length),
            "largest_army": env.engine.largest_army_owner == player_id,
            "longest_road": env.engine.longest_road_owner == player_id,
        }

    summary = {
        "game": game_idx,
        "steps": steps,
        "turns": int(env.engine.turn_number),
        "winner": env.engine.winner.name if env.engine.winner is not None else "NONE",
        "truncated": truncated,
        "players": player_summaries,
    }
    return summary, step_rows


def format_game_log(summary: dict[str, Any], agents: dict[PlayerId, BaseSeatAgent]) -> str:
    lines = [
        f"Game {summary['game']}",
        (
            f"result  | winner={summary['winner']} steps={summary['steps']} "
            f"turns={summary['turns']} truncated={int(summary['truncated'])}"
        ),
    ]

    for player_id in PLAYER_ORDER:
        player = summary["players"][player_id.name]
        lines.append(
            f"seat    | player={player_id.name} agent={agents[player_id].name} "
            f"vp={player['victory_points']} settlements={player['settlements']} "
            f"cities={player['cities']} roads={player['roads']}"
        )
        lines.append(
            f"extras  | player={player_id.name} knights={player['played_knights']} "
            f"dev_cards={player['dev_cards_held']} hidden_vp={player['hidden_vp_cards']} "
            f"largest_army={int(player['largest_army'])} longest_road={int(player['longest_road'])} "
            f"longest_road_len={player['longest_road_length']}"
        )

    return "\n".join(lines)


def format_final_summary(summaries: list[dict[str, Any]], agents: dict[PlayerId, BaseSeatAgent]) -> str:
    win_counts = {agents[player_id].name: 0 for player_id in PLAYER_ORDER}
    truncations = 0
    total_turns = 0
    total_steps = 0
    vp_totals = {agents[player_id].name: 0 for player_id in PLAYER_ORDER}

    for summary in summaries:
        truncations += int(summary["truncated"])
        total_turns += int(summary["turns"])
        total_steps += int(summary["steps"])
        winner_name = summary["winner"]
        if winner_name != "NONE":
            player_id = PlayerId[winner_name]
            win_counts[agents[player_id].name] += 1
        for player_id in PLAYER_ORDER:
            vp_totals[agents[player_id].name] += int(summary["players"][player_id.name]["victory_points"])

    games = max(len(summaries), 1)
    lines = [
        "Summary",
        f"games   | count={len(summaries)} avg_steps={total_steps / games:.2f} avg_turns={total_turns / games:.2f} truncations={truncations}",
        "wins    | "
        + " ".join(f"{name}={count} ({count / games:.3f})" for name, count in win_counts.items()),
        "avg_vp  | "
        + " ".join(f"{name}={vp_totals[name] / games:.2f}" for name in vp_totals),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mixed 4-player Catan agent match.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--enable-trading", action="store_true", help="Enable player trading. Off by default for stable mixed-agent smoke runs.")
    parser.add_argument("--unified-checkpoint", type=str, default="checkpoints/unified_checkpoint_200.pt")
    parser.add_argument("--dqn-checkpoint", type=str, default="")
    parser.add_argument("--hybrid-checkpoint", type=str, default="checkpoints/hybrid_checkpoint_200.pt")
    parser.add_argument("--ppo-checkpoint", type=str, default="")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid in RED seat instead of raw PPO.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/mixed_four_player"))
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)
        args.device = "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    agents = build_agents(args)
    summaries: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.output_dir / "mixed_four_player_training_log.txt")

    header = "\n".join(
        [
            f"Starting mixed four-player evaluation for {args.games} games",
            f"Config: max_steps={args.max_steps} seed={args.seed} trading={int(args.enable_trading)} deterministic={int(args.deterministic)}",
            "Seat map:",
            *[f"  {player_id.name}: {agents[player_id].name}" for player_id in PLAYER_ORDER],
            "",
        ]
    )
    log_path.write_text(header, encoding="utf-8")
    print(header, end="")

    for game_idx in range(args.games):
        summary, rows = run_one_game(game_idx, args, agents)
        summaries.append(summary)
        game_log = format_game_log(summary, agents) + "\n\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(game_log)
        print(game_log, end="")

    final_summary = format_final_summary(summaries, agents)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(final_summary)
    print(final_summary, end="")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()
