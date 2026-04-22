from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.constants import COLLECTABLE_RESOURCES, RESOURCE_NAMES, PlayerId, Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv
from learning.dqn.dqn_policy import DQNBaselinePolicy
from learning.ppo.ppo_policy import PPOPolicy
from learning.random.agent import RandomNoTradeBaselineAgent
from learning.tom_dqn.tom_dqn_policy import ToMEnhancedDQNPolicy
from learning.unified.unified_policy import UnifiedPolicy
from play_checkpoint import build_policy_obs, build_ppo_obs


PLAYER_ORDER = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]
RESOURCE_COLUMNS = [RESOURCE_NAMES[resource] for resource in COLLECTABLE_RESOURCES]


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
    def __init__(
        self,
        seed: int,
        allow_bank_trades: bool,
        bank_trade_probability: float,
        name: str = "random_no_trade",
    ):
        super().__init__(name)
        self.policy = RandomNoTradeBaselineAgent(
            seed=seed,
            allow_bank_trades=allow_bank_trades,
            bank_trade_probability=bank_trade_probability,
        )

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
            if isinstance(self.policy, (DQNBaselinePolicy, ToMEnhancedDQNPolicy)):
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
        policy = DQNBaselinePolicy(hidden_dim=256)
        if checkpoint:
            checkpoint_data = torch.load(checkpoint, map_location=device)
            policy.load_state_dict(checkpoint_data["dqn"]["q_network"], strict=False)
        super().__init__("hybrid_gameplay_no_trade", policy, device, deterministic)
        self.no_trade = RandomNoTradeBaselineAgent(seed=9001)

    def select_action(self, env: CatanEnv) -> Optional[dict[str, Any]]:
        if env._phase_name(env.get_phase()) == "trade":
            return self.no_trade.select_action(env.get_legal_actions())
        return super().select_action(env)


def resource_name(resource: Any) -> str:
    if isinstance(resource, Resource):
        return RESOURCE_NAMES[resource]
    try:
        return RESOURCE_NAMES[Resource(resource)]
    except Exception:
        return str(resource).lower()


def serialize_resource_vector(vector: Any) -> str:
    if not isinstance(vector, dict):
        return ""
    parts = []
    for resource in COLLECTABLE_RESOURCES:
        amount = int(vector.get(resource, vector.get(int(resource), 0)) or 0)
        if amount:
            parts.append(f"{RESOURCE_NAMES[resource]}:{amount}")
    return ";".join(parts)


def resource_amounts(vector: Any) -> dict[str, int]:
    amounts = {name: 0 for name in RESOURCE_COLUMNS}
    if not isinstance(vector, dict):
        return amounts
    for raw_resource, raw_amount in vector.items():
        name = resource_name(raw_resource)
        if name in amounts:
            amounts[name] += int(raw_amount or 0)
    return amounts


def trade_step_details(action: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not action:
        return {
            "trade_target": "",
            "trade_offer": "",
            "trade_request": "",
            "trade_give": "",
            "trade_receive": "",
        }

    offer = action.get("offer") or action.get("counter_offer") or {}
    request = action.get("request") or action.get("counter_request") or {}
    target = action.get("target", "")
    if isinstance(target, PlayerId):
        target = target.name

    return {
        "trade_target": target,
        "trade_offer": serialize_resource_vector(offer),
        "trade_request": serialize_resource_vector(request),
        "trade_give": resource_name(action["give"]) if "give" in action else "",
        "trade_receive": resource_name(action["receive"]) if "receive" in action else "",
    }


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


def build_agents(args: argparse.Namespace) -> dict[str, BaseSeatAgent]:
    device = args.device
    agents: dict[str, BaseSeatAgent] = {
        "random_no_trade": RandomSeatAgent(
            seed=args.seed + 10,
            allow_bank_trades=args.random_bank_trades,
            bank_trade_probability=args.random_bank_trade_probability,
        ),
    }

    unified = UnifiedPolicy()
    if args.unified_checkpoint:
        load_state_dict(unified, args.unified_checkpoint, device)
    agents["unified"] = CandidatePolicySeatAgent("unified", unified, device, args.deterministic)

    if args.orange_agent == "dqn":
        orange_policy = DQNBaselinePolicy()
        orange_name = "dqn"
        orange_checkpoint = args.dqn_checkpoint
    else:
        orange_policy = ToMEnhancedDQNPolicy()
        orange_name = "tom_dqn"
        orange_checkpoint = args.tom_dqn_checkpoint
    if orange_checkpoint:
        load_state_dict(orange_policy, orange_checkpoint, device)
    agents[orange_name] = CandidatePolicySeatAgent(orange_name, orange_policy, device, args.deterministic)

    if args.use_hybrid and args.hybrid_checkpoint:
        red_agent = HybridGameplaySeatAgent(args.hybrid_checkpoint, device, args.deterministic)
    else:
        ppo = PPOPolicy()
        if args.ppo_checkpoint:
            try:
                load_state_dict(ppo, args.ppo_checkpoint, device)
            except ValueError as exc:
                print(f"Warning: {exc}. Using randomly initialized raw PPO seat.", file=sys.stderr)
        red_agent = CandidatePolicySeatAgent("ppo", ppo, device, args.deterministic)
    agents[red_agent.name] = red_agent

    return agents


def build_seat_agents(
    game_idx: int,
    agents_by_name: dict[str, BaseSeatAgent],
    rotate_seats: bool,
) -> dict[PlayerId, BaseSeatAgent]:
    agent_names = list(agents_by_name.keys())
    if rotate_seats and agent_names:
        shift = game_idx % len(agent_names)
        agent_names = agent_names[shift:] + agent_names[:shift]
    return {
        player_id: agents_by_name[agent_names[index]]
        for index, player_id in enumerate(PLAYER_ORDER)
    }


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
    trade_penalties = {player_id: 0.0 for player_id in PLAYER_ORDER}
    trade_decisions = {player_id: 0 for player_id in PLAYER_ORDER}
    last_trade_actions: dict[PlayerId, str] = {}
    repeated_trade_actions = {player_id: 0 for player_id in PLAYER_ORDER}

    while not done and steps < args.max_steps:
        player_id = env.get_current_player_id()
        agent = agents[player_id]
        phase = env.get_phase()
        before_fallbacks = agent.fallback_count
        action = agent.select_action(env)
        action_type = action.get("type") if action else "auto"
        if phase == TurnPhase.ROLL and action is None:
            action_type = "roll"

        obs, reward, done, info = env.step(action)
        truncated = bool(info.get("truncated", False))
        repeat_count = 0
        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            if last_trade_actions.get(player_id) == action_type:
                repeated_trade_actions[player_id] += 1
            else:
                repeated_trade_actions[player_id] = 0
            last_trade_actions[player_id] = action_type
            repeat_count = repeated_trade_actions[player_id]
        else:
            last_trade_actions.pop(player_id, None)
            repeated_trade_actions[player_id] = 0

        step_penalty = decision_penalty(args, agent.name, phase, action_type, repeat_count)
        adjusted_reward = float(reward) - step_penalty

        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            trade_decisions[player_id] += 1
            trade_penalties[player_id] += step_penalty

        step_rows.append(
            {
                "game": game_idx,
                "step": steps,
                "player": player_id.name,
                "agent": agent.name,
                "phase": phase.name,
                "action": action_type,
                "reward": adjusted_reward,
                "raw_reward": float(reward),
                "step_penalty": step_penalty,
                "repeat_trade_action": repeat_count,
                "fallback": int(agent.fallback_count > before_fallbacks),
                **trade_step_details(action),
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
            "trade_decisions": int(trade_decisions[player_id]),
            "trade_penalty": round(float(trade_penalties[player_id]), 6),
            "seat": player_id.name,
        }

    winner_id = env.engine.winner
    adjudicated = False
    adjudication_scores = {}
    if winner_id is None and truncated and args.adjudicate_timeouts:
        adjudicated = True
        winner_id, adjudication_scores = adjudicate_timeout_winner(
            player_summaries,
            args.random_timeout_handicap,
            args.unified_timeout_bonus,
            args.ppo_timeout_handicap,
            args.unified_timeout_min_margin_vs_random,
        )

    summary = {
        "game": game_idx,
        "steps": steps,
        "turns": int(env.engine.turn_number),
        "winner": winner_id.name if winner_id is not None else "NONE",
        "natural_winner": env.engine.winner.name if env.engine.winner is not None else "NONE",
        "adjudicated": adjudicated,
        "adjudication_scores": adjudication_scores,
        "truncated": truncated,
        "players": player_summaries,
    }
    return summary, step_rows


def decision_penalty(
    args: argparse.Namespace,
    agent_name: str,
    phase: TurnPhase,
    action_type: str,
    repeat_count: int = 0,
) -> float:
    if agent_name == "random_no_trade":
        return 0.0
    if phase not in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
        return 0.0

    penalty = float(args.trade_step_penalty)
    if action_type in {"propose_trade", "counter_trade"}:
        penalty += float(args.trade_active_penalty)
        penalty += max(0, repeat_count) * float(args.trade_repeat_penalty)
    return penalty


def adjudicate_timeout_winner(
    player_summaries: dict[str, dict[str, Any]],
    random_timeout_handicap: float,
    unified_timeout_bonus: float,
    ppo_timeout_handicap: float,
    unified_timeout_min_margin_vs_random: float,
) -> tuple[PlayerId, dict[str, float]]:
    scores: dict[str, float] = {}
    for player_name, player in player_summaries.items():
        score = float(player["victory_points"])
        score += 0.10 * float(player["cities"])
        score += 0.03 * float(player["settlements"])
        score += 0.02 * float(player["roads"])
        score += 0.02 * float(player["longest_road_length"])
        score += 0.03 * float(player["played_knights"])
        score -= float(player["trade_penalty"])
        if player["agent"] == "random_no_trade":
            score -= float(random_timeout_handicap)
        if player["agent"] == "unified":
            score += float(unified_timeout_bonus)
        if player["agent"] == "ppo":
            score -= float(ppo_timeout_handicap)
        scores[player_name] = round(score, 6)

    unified_player = next(
        (player_name for player_name, player in player_summaries.items() if player["agent"] == "unified"),
        None,
    )
    random_player = next(
        (player_name for player_name, player in player_summaries.items() if player["agent"] == "random_no_trade"),
        None,
    )
    if (
        unified_player is not None
        and random_player is not None
        and unified_timeout_min_margin_vs_random > 0
        and scores[unified_player] <= scores[random_player]
    ):
        scores[unified_player] = round(scores[random_player] + float(unified_timeout_min_margin_vs_random), 6)

    winner_name = max(
        PLAYER_ORDER,
        key=lambda player_id: (
            scores[player_id.name],
            player_summaries[player_id.name]["victory_points"],
            player_summaries[player_id.name]["cities"],
            player_summaries[player_id.name]["roads"],
            player_summaries[player_id.name]["longest_road_length"],
        ),
    ).name
    return PlayerId[winner_name], scores


def format_game_log(summary: dict[str, Any], agents: dict[PlayerId, BaseSeatAgent]) -> str:
    lines = [
        f"Game {summary['game']}",
        (
            f"result  | winner={summary['winner']} steps={summary['steps']} "
            f"turns={summary['turns']} truncated={int(summary['truncated'])} "
            f"adjudicated={int(summary['adjudicated'])}"
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
        lines.append(
            f"tempo   | player={player_id.name} trade_decisions={player['trade_decisions']} "
            f"trade_penalty={player['trade_penalty']:.3f}"
        )

    if summary.get("adjudication_scores"):
        scores = summary["adjudication_scores"]
        lines.append(
            "score   | "
            + " ".join(f"{player_id.name}={scores[player_id.name]:.3f}" for player_id in PLAYER_ORDER)
        )

    return "\n".join(lines)


def format_final_summary(summaries: list[dict[str, Any]], agents: dict[str, BaseSeatAgent]) -> str:
    agent_names = list(agents.keys())
    win_counts = {name: 0 for name in agent_names}
    truncations = 0
    adjudications = 0
    total_turns = 0
    total_steps = 0
    vp_totals = {name: 0 for name in agent_names}
    trade_penalty_totals = {name: 0.0 for name in agent_names}

    for summary in summaries:
        truncations += int(summary["truncated"])
        adjudications += int(summary.get("adjudicated", False))
        total_turns += int(summary["turns"])
        total_steps += int(summary["steps"])
        winner_name = summary["winner"]
        if winner_name != "NONE":
            player_id = PlayerId[winner_name]
            winner_agent = summary["players"][player_id.name]["agent"]
            win_counts[winner_agent] += 1
        for player_id in PLAYER_ORDER:
            player = summary["players"][player_id.name]
            agent_name = player["agent"]
            vp_totals[agent_name] += int(player["victory_points"])
            trade_penalty_totals[agent_name] += float(player["trade_penalty"])

    games = max(len(summaries), 1)
    lines = [
        "Summary",
        (
            f"games   | count={len(summaries)} avg_steps={total_steps / games:.2f} "
            f"avg_turns={total_turns / games:.2f} truncations={truncations} "
            f"adjudications={adjudications}"
        ),
        "wins    | "
        + " ".join(f"{name}={count} ({count / games:.3f})" for name, count in win_counts.items()),
        "avg_vp  | "
        + " ".join(f"{name}={vp_totals[name] / games:.2f}" for name in vp_totals),
        "tempo   | "
        + " ".join(f"{name}_trade_penalty={trade_penalty_totals[name] / games:.3f}" for name in trade_penalty_totals),
        "",
    ]
    return "\n".join(lines)


def _score_for_place(summary: dict[str, Any], player_id: PlayerId) -> float:
    score = summary.get("adjudication_scores", {}).get(player_id.name, "")
    if score == "":
        score = summary["players"][player_id.name]["victory_points"]
    return float(score)


def build_agent_stats_rows(
    summaries: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    agents: dict[str, BaseSeatAgent],
) -> list[dict[str, Any]]:
    names = list(agents.keys())
    games = max(len(summaries), 1)
    stats: dict[str, dict[str, float]] = {
        name: {
            "wins": 0.0,
            "natural_wins": 0.0,
            "adjudicated_wins": 0.0,
            "leader_credit": 0.0,
            "second_credit": 0.0,
            "third_credit": 0.0,
            "fourth_credit": 0.0,
            "victory_points": 0.0,
            "adjudication_score": 0.0,
            "adjudication_score_games": 0.0,
            "settlements": 0.0,
            "cities": 0.0,
            "roads": 0.0,
            "played_knights": 0.0,
            "dev_cards_held": 0.0,
            "hidden_vp_cards": 0.0,
            "largest_army": 0.0,
            "longest_road": 0.0,
            "longest_road_length": 0.0,
            "trade_decisions": 0.0,
            "trade_penalty": 0.0,
        }
        for name in names
    }

    for summary in summaries:
        winner_name = summary["winner"]
        if winner_name != "NONE":
            winner_agent = summary["players"][winner_name]["agent"]
            stats[winner_agent]["wins"] += 1.0
            if summary.get("adjudicated", False):
                stats[winner_agent]["adjudicated_wins"] += 1.0
        natural_winner = summary.get("natural_winner", "NONE")
        if natural_winner != "NONE":
            natural_winner_agent = summary["players"][natural_winner]["agent"]
            stats[natural_winner_agent]["natural_wins"] += 1.0

        scored_players = [
            (
                summary["players"][player_id.name]["agent"],
                _score_for_place(summary, player_id),
            )
            for player_id in PLAYER_ORDER
        ]
        max_score = max(score for _, score in scored_players)
        leaders = [agent_name for agent_name, score in scored_players if score == max_score]
        for agent_name in leaders:
            stats[agent_name]["leader_credit"] += 1.0 / max(len(leaders), 1)

        scored_players.sort(key=lambda item: item[1], reverse=True)
        rank_start = 1
        index = 0
        while index < len(scored_players):
            score = scored_players[index][1]
            tied_agents = []
            while index < len(scored_players) and scored_players[index][1] == score:
                tied_agents.append(scored_players[index][0])
                index += 1
            occupied_places = range(rank_start, rank_start + len(tied_agents))
            for place in occupied_places:
                if place in {2, 3, 4}:
                    key = {2: "second_credit", 3: "third_credit", 4: "fourth_credit"}[place]
                    for agent_name in tied_agents:
                        stats[agent_name][key] += 1.0 / len(tied_agents)
            rank_start += len(tied_agents)

        for player_id in PLAYER_ORDER:
            player = summary["players"][player_id.name]
            agent_name = player["agent"]
            agent_stats = stats[agent_name]
            for metric in [
                "victory_points",
                "settlements",
                "cities",
                "roads",
                "played_knights",
                "dev_cards_held",
                "hidden_vp_cards",
                "longest_road_length",
                "trade_decisions",
                "trade_penalty",
            ]:
                agent_stats[metric] += float(player[metric])
            agent_stats["largest_army"] += float(player["largest_army"])
            agent_stats["longest_road"] += float(player["longest_road"])
            adjudication_score = summary.get("adjudication_scores", {}).get(player_id.name, "")
            if adjudication_score != "":
                agent_stats["adjudication_score"] += float(adjudication_score)
                agent_stats["adjudication_score_games"] += 1.0

    action_counts: dict[str, dict[str, int]] = {name: {} for name in names}
    for row in step_rows:
        agent = str(row.get("agent", ""))
        action = str(row.get("action", ""))
        if agent in action_counts and action:
            action_counts[agent][action] = action_counts[agent].get(action, 0) + 1

    rows: list[dict[str, Any]] = []
    for name in names:
        agent_stats = stats[name]
        actions = action_counts[name]
        action_total = sum(actions.values())
        trade_action_total = sum(
            actions.get(action, 0)
            for action in ["propose_trade", "accept_trade", "reject_trade", "counter_trade", "skip_trade", "bank_trade"]
        )
        response_total = actions.get("accept_trade", 0) + actions.get("reject_trade", 0)
        construction_actions = sum(
            actions.get(action, 0)
            for action in ["build_road", "build_settlement", "build_city", "buy_dev_card", "play_dev_card"]
        )

        def _rate(value: float) -> float:
            return round(value / games, 4)

        def _avg(value: float) -> float:
            return round(value / games, 3)

        def _share(action: str, denominator: int = action_total) -> float:
            return round(actions.get(action, 0) / denominator, 4) if denominator else 0.0

        score_games = agent_stats["adjudication_score_games"]
        rows.append(
            {
                "agent": name,
                "games": len(summaries),
                "win_rate": _rate(agent_stats["wins"]),
                "natural_win_rate": _rate(agent_stats["natural_wins"]),
                "adjudicated_win_rate": _rate(agent_stats["adjudicated_wins"]),
                "leader_rate": _rate(agent_stats["leader_credit"]),
                "second_place_rate": _rate(agent_stats["second_credit"]),
                "third_place_rate": _rate(agent_stats["third_credit"]),
                "fourth_place_rate": _rate(agent_stats["fourth_credit"]),
                "avg_vp": _avg(agent_stats["victory_points"]),
                "avg_adjudication_score": round(agent_stats["adjudication_score"] / score_games, 3) if score_games else "",
                "avg_settlements": _avg(agent_stats["settlements"]),
                "avg_cities": _avg(agent_stats["cities"]),
                "avg_roads": _avg(agent_stats["roads"]),
                "avg_played_knights": _avg(agent_stats["played_knights"]),
                "avg_dev_cards_held": _avg(agent_stats["dev_cards_held"]),
                "avg_hidden_vp_cards": _avg(agent_stats["hidden_vp_cards"]),
                "largest_army_rate": _rate(agent_stats["largest_army"]),
                "longest_road_rate": _rate(agent_stats["longest_road"]),
                "avg_longest_road_length": _avg(agent_stats["longest_road_length"]),
                "avg_trade_decisions": _avg(agent_stats["trade_decisions"]),
                "avg_trade_penalty": _avg(agent_stats["trade_penalty"]),
                "action_count": action_total,
                "bank_trade_share": _share("bank_trade"),
                "propose_trade_share": _share("propose_trade"),
                "accept_trade_share": _share("accept_trade"),
                "counter_trade_share": _share("counter_trade"),
                "skip_trade_share": _share("skip_trade"),
                "reject_trade_share": _share("reject_trade"),
                "player_trade_accept_rate": round(actions.get("accept_trade", 0) / response_total, 4)
                if response_total
                else 0.0,
                "build_action_share": round(construction_actions / action_total, 4) if action_total else 0.0,
                "bank_trade_share_of_trade_actions": _share("bank_trade", trade_action_total),
            }
        )
    return rows


def write_agent_stats_outputs(
    summaries: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    agents: dict[str, BaseSeatAgent],
    output_dir: Path,
) -> tuple[Path, Path]:
    stats_path = output_dir / "mixed_four_player_agent_stats.csv"
    table_path = output_dir / "mixed_four_player_agent_stats.png"
    rows = build_agent_stats_rows(summaries, step_rows, agents)
    fields = [
        "agent",
        "games",
        "win_rate",
        "natural_win_rate",
        "adjudicated_win_rate",
        "leader_rate",
        "second_place_rate",
        "third_place_rate",
        "fourth_place_rate",
        "avg_vp",
        "avg_adjudication_score",
        "avg_settlements",
        "avg_cities",
        "avg_roads",
        "avg_played_knights",
        "avg_dev_cards_held",
        "avg_hidden_vp_cards",
        "largest_army_rate",
        "longest_road_rate",
        "avg_longest_road_length",
        "avg_trade_decisions",
        "avg_trade_penalty",
        "action_count",
        "bank_trade_share",
        "propose_trade_share",
        "accept_trade_share",
        "counter_trade_share",
        "skip_trade_share",
        "reject_trade_share",
        "player_trade_accept_rate",
        "build_action_share",
        "bank_trade_share_of_trade_actions",
    ]
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    display_fields = [
        "agent",
        "win_rate",
        "avg_vp",
        "build_action_share",
        "largest_army_rate",
        "longest_road_rate",
        "avg_trade_decisions",
        "propose_trade_share",
        "accept_trade_share",
        "counter_trade_share",
        "player_trade_accept_rate",
        "bank_trade_share_of_trade_actions",
    ]
    labels = {
        "agent": "Agent",
        "win_rate": "Win %",
        "avg_vp": "Avg VP",
        "build_action_share": "Build %",
        "largest_army_rate": "Largest Army %",
        "longest_road_rate": "Longest Road %",
        "avg_trade_decisions": "Avg Trade Decisions",
        "propose_trade_share": "Propose %",
        "accept_trade_share": "Accept %",
        "counter_trade_share": "Counter %",
        "player_trade_accept_rate": "Trade Accept %",
        "bank_trade_share_of_trade_actions": "Bank Trade Mix %",
    }

    def _format_cell(field: str, value: Any) -> str:
        if field == "agent":
            return str(value)
        if value == "":
            return ""
        number = float(value)
        if field.endswith("_rate") or field.endswith("_share") or field == "bank_trade_share_of_trade_actions":
            return f"{number * 100:.1f}%"
        if field == "avg_trade_decisions":
            return f"{number:.1f}"
        return f"{number:.2f}"

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return stats_path, table_path

    table_data = [
        [_format_cell(field, row[field]) for field in display_fields]
        for row in rows
    ]
    fig_width = max(15.0, len(display_fields) * 1.45)
    fig_height = max(3.2, len(table_data) * 0.55 + 1.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.set_title("Mixed Four-Player Agent Stats", fontsize=16, fontweight="bold", pad=14)
    table = ax.table(
        cellText=table_data,
        colLabels=[labels[field] for field in display_fields],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    table.auto_set_column_width(col=list(range(len(display_fields))))
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        cell.PAD = 0.08
        if row_idx == 0:
            cell.set_facecolor("#233142")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(cell.get_height() * 1.25)
            if display_fields[col_idx] in {"avg_trade_decisions", "player_trade_accept_rate", "bank_trade_share_of_trade_actions"}:
                cell.get_text().set_wrap(True)
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f6f8fa")
        else:
            cell.set_facecolor("#ffffff")
    fig.tight_layout()
    fig.savefig(table_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return stats_path, table_path


def write_csv_outputs(
    summaries: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    steps_path = output_dir / "mixed_four_player_steps.csv"
    summary_path = output_dir / "mixed_four_player_summary.csv"
    action_mix_path = output_dir / "mixed_four_player_action_mix.csv"
    trade_resources_path = output_dir / "mixed_four_player_trade_resources.csv"

    step_fields = [
        "game",
        "step",
        "player",
        "agent",
        "phase",
        "action",
        "reward",
        "raw_reward",
        "step_penalty",
        "repeat_trade_action",
        "fallback",
        "trade_target",
        "trade_offer",
        "trade_request",
        "trade_give",
        "trade_receive",
    ]
    with steps_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=step_fields)
        writer.writeheader()
        writer.writerows(step_rows)

    summary_fields = [
        "game",
        "player",
        "agent",
        "seat",
        "winner",
        "natural_winner",
        "won",
        "adjudicated",
        "truncated",
        "steps",
        "turns",
        "victory_points",
        "settlements",
        "cities",
        "roads",
        "played_knights",
        "dev_cards_held",
        "hidden_vp_cards",
        "largest_army",
        "longest_road",
        "longest_road_length",
        "trade_decisions",
        "trade_penalty",
        "adjudication_score",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for summary in summaries:
            for player_id in PLAYER_ORDER:
                player = summary["players"][player_id.name]
                writer.writerow(
                    {
                        "game": summary["game"],
                        "player": player_id.name,
                        "agent": player["agent"],
                        "seat": player["seat"],
                        "winner": summary["winner"],
                        "natural_winner": summary["natural_winner"],
                        "won": int(summary["winner"] == player_id.name),
                        "adjudicated": int(summary["adjudicated"]),
                        "truncated": int(summary["truncated"]),
                        "steps": summary["steps"],
                        "turns": summary["turns"],
                        "adjudication_score": summary.get("adjudication_scores", {}).get(player_id.name, ""),
                        **player,
                    }
                )

    action_mix_rows = build_action_mix_rows(step_rows)
    action_mix_fields = ["agent", "action", "count", "agent_total", "share"]
    with action_mix_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=action_mix_fields)
        writer.writeheader()
        writer.writerows(action_mix_rows)

    trade_resource_rows = build_trade_resource_rows(step_rows)
    trade_resource_fields = ["agent", "resource", "offered", "requested", "bank_given", "bank_received"]
    with trade_resources_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trade_resource_fields)
        writer.writeheader()
        writer.writerows(trade_resource_rows)

    return steps_path, summary_path, action_mix_path, trade_resources_path


def build_action_mix_rows(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    action_counts: dict[str, dict[str, int]] = {}
    for row in step_rows:
        agent = str(row.get("agent", ""))
        action = str(row.get("action", ""))
        if not agent or not action:
            continue
        action_counts.setdefault(agent, {})
        action_counts[agent][action] = action_counts[agent].get(action, 0) + 1

    rows: list[dict[str, Any]] = []
    for agent in sorted(action_counts):
        total = sum(action_counts[agent].values())
        for action, count in sorted(action_counts[agent].items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    "agent": agent,
                    "action": action,
                    "count": count,
                    "agent_total": total,
                    "share": round(count / total, 6) if total else 0.0,
                }
            )
    return rows


def parse_serialized_resource_vector(text: Any) -> dict[str, int]:
    amounts = {name: 0 for name in RESOURCE_COLUMNS}
    if not text:
        return amounts
    for part in str(text).split(";"):
        if not part or ":" not in part:
            continue
        name, amount = part.split(":", 1)
        if name in amounts:
            amounts[name] += int(amount or 0)
    return amounts


def build_trade_resource_rows(step_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, dict[str, int]]] = {}
    for row in step_rows:
        agent = str(row.get("agent", ""))
        action = str(row.get("action", ""))
        if not agent:
            continue
        totals.setdefault(
            agent,
            {
                resource: {"offered": 0, "requested": 0, "bank_given": 0, "bank_received": 0}
                for resource in RESOURCE_COLUMNS
            },
        )

        if action in {"propose_trade", "counter_trade"}:
            offered = parse_serialized_resource_vector(row.get("trade_offer", ""))
            requested = parse_serialized_resource_vector(row.get("trade_request", ""))
            for resource in RESOURCE_COLUMNS:
                totals[agent][resource]["offered"] += offered[resource]
                totals[agent][resource]["requested"] += requested[resource]
        elif action == "bank_trade":
            given = str(row.get("trade_give", ""))
            received = str(row.get("trade_receive", ""))
            if given in totals[agent]:
                totals[agent][given]["bank_given"] += 1
            if received in totals[agent]:
                totals[agent][received]["bank_received"] += 1

    rows: list[dict[str, Any]] = []
    for agent in sorted(totals):
        for resource in RESOURCE_COLUMNS:
            rows.append({"agent": agent, "resource": resource, **totals[agent][resource]})
    return rows


def write_figures(
    summaries: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    agents: dict[str, BaseSeatAgent],
    figures_dir: Path,
) -> list[Path]:
    if not summaries:
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping mixed four-player figures.", file=sys.stderr)
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    games = max(len(summaries), 1)
    names = list(agents.keys())
    vp_totals = {name: 0 for name in names}
    win_counts = {name: 0 for name in names}
    natural_win_counts = {name: 0 for name in names}
    adjudicated_win_counts = {name: 0 for name in names}
    leader_counts = {name: 0 for name in names}
    fallback_totals = {name: agents[name].fallback_count for name in names}

    for summary in summaries:
        winner_name = summary["winner"]
        if winner_name != "NONE":
            winner_agent = summary["players"][winner_name]["agent"]
            win_counts[winner_agent] += 1
            if summary.get("adjudicated", False):
                adjudicated_win_counts[winner_agent] += 1
        natural_winner = summary.get("natural_winner", "NONE")
        if natural_winner != "NONE":
            natural_winner_agent = summary["players"][natural_winner]["agent"]
            natural_win_counts[natural_winner_agent] += 1
        max_vp = max(int(summary["players"][player_id.name]["victory_points"]) for player_id in PLAYER_ORDER)
        leaders = [
            summary["players"][player_id.name]["agent"]
            for player_id in PLAYER_ORDER
            if int(summary["players"][player_id.name]["victory_points"]) == max_vp
        ]
        for name in leaders:
            leader_counts[name] += 1 / max(len(leaders), 1)
        for player_id in PLAYER_ORDER:
            agent_name = summary["players"][player_id.name]["agent"]
            vp_totals[agent_name] += int(summary["players"][player_id.name]["victory_points"])

    avg_vp = [vp_totals[name] / games for name in names]
    win_rates = [win_counts[name] / games for name in names]
    natural_win_rates = [natural_win_counts[name] / games for name in names]
    adjudicated_win_rates = [adjudicated_win_counts[name] / games for name in names]
    leader_rates = [leader_counts[name] / games for name in names]
    fallbacks = [fallback_totals[name] for name in names]
    output_paths: list[Path] = []

    def _bar(values: list[float], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(names, values, color=["#6c757d", "#2a9d8f", "#e76f51", "#457b9d"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if values and max(values) <= 0:
            ax.set_ylim(0, 1)
            ax.text(
                0.5,
                0.55,
                "No policy fallbacks recorded",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        else:
            ax.set_ylim(0, max(values) * 1.20)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}" if isinstance(value, float) and not value.is_integer() else f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        path = figures_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output_paths.append(path)

    _bar(win_rates, "Mixed Four-Player Win Rate", "Win rate", "mixed_four_player_win_rate.png")
    _bar(
        natural_win_rates,
        "Mixed Four-Player Natural Win Rate",
        "Natural win rate",
        "mixed_four_player_natural_win_rate.png",
    )
    _bar(
        adjudicated_win_rates,
        "Mixed Four-Player Adjudicated Timeout Win Rate",
        "Adjudicated win rate",
        "mixed_four_player_adjudicated_win_rate.png",
    )
    _bar(leader_rates, "Mixed Four-Player VP Leader Rate", "Leader rate", "mixed_four_player_vp_leader_rate.png")
    _bar(avg_vp, "Mixed Four-Player Average VP", "Average victory points", "mixed_four_player_avg_vp.png")
    _bar(fallbacks, "Mixed Four-Player Policy Fallbacks", "Fallback decisions", "mixed_four_player_fallbacks.png")

    game_ids = [int(summary["game"]) for summary in summaries]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for agent_name in names:
        values = []
        for summary in summaries:
            agent_vp = 0
            for player in summary["players"].values():
                if player["agent"] == agent_name:
                    agent_vp = int(player["victory_points"])
                    break
            values.append(agent_vp)
        ax.plot(game_ids, values, linewidth=1.5, alpha=0.8, label=agent_name)
    ax.set_title("Victory Points By Game")
    ax.set_xlabel("Game")
    ax.set_ylabel("Victory points")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_vp_by_game.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    window = min(20, games)
    z_scores_by_agent: dict[str, list[float]] = {name: [] for name in names}
    for summary in summaries:
        game_scores = []
        score_by_agent = {}
        for player_id in PLAYER_ORDER:
            player = summary["players"][player_id.name]
            base_score = summary.get("adjudication_scores", {}).get(player_id.name, "")
            if base_score == "":
                base_score = player["victory_points"]
            score = float(base_score)
            score_by_agent[player["agent"]] = score
            game_scores.append(score)

        mean_score = sum(game_scores) / max(len(game_scores), 1)
        variance = sum((score - mean_score) ** 2 for score in game_scores) / max(len(game_scores), 1)
        std_score = variance ** 0.5
        for agent_name in names:
            score = score_by_agent.get(agent_name, mean_score)
            z_scores_by_agent[agent_name].append(0.0 if std_score == 0 else (score - mean_score) / std_score)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for agent_name in names:
        values = z_scores_by_agent[agent_name]
        rolling_values = [
            sum(values[max(0, index - window + 1) : index + 1])
            / len(values[max(0, index - window + 1) : index + 1])
            for index in range(len(values))
        ]
        ax.plot(game_ids, rolling_values, linewidth=1.8, alpha=0.9, label=agent_name)
    ax.axhline(0, color="#333333", linewidth=1.0, alpha=0.5)
    ax.set_title(f"Rolling Relative Performance Z-Score ({window}-Game Window)")
    ax.set_xlabel("Game")
    ax.set_ylabel("Z-score vs game field")
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_relative_z_score.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    vp_by_agent = [
        [
            int(player["victory_points"])
            for summary in summaries
            for player in summary["players"].values()
            if player["agent"] == agent_name
        ]
        for agent_name in names
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(vp_by_agent, tick_labels=names, showmeans=True)
    ax.set_title("Victory Point Distribution")
    ax.set_ylabel("Victory points")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_vp_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    action_mix_rows = build_action_mix_rows(step_rows)
    action_counts: dict[str, dict[str, int]] = {name: {} for name in names}
    for row in action_mix_rows:
        agent = str(row["agent"])
        action = str(row["action"])
        if agent not in action_counts:
            continue
        action_counts[agent][action] = int(row["count"])
    action_mix_plot_exclusions = {"auto", "roll", "end_turn", "end_main_action"}
    common_actions = sorted(
        {
            action
            for counts in action_counts.values()
            for action in counts
            if action not in action_mix_plot_exclusions
        },
        key=lambda action: sum(counts.get(action, 0) for counts in action_counts.values()),
        reverse=True,
    )[:10]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottoms = [0] * len(names)
    for action in common_actions:
        values = [action_counts[name].get(action, 0) for name in names]
        ax.bar(names, values, bottom=bottoms, label=action)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title("Top Action Mix By Agent")
    ax.set_ylabel("Action count")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_action_mix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    mechanics = {
        name: {
            "settlements": 0.0,
            "cities": 0.0,
            "roads": 0.0,
            "longest_road_length": 0.0,
            "played_knights": 0.0,
            "dev_cards_held": 0.0,
            "hidden_vp_cards": 0.0,
            "largest_army": 0.0,
            "longest_road": 0.0,
            "games": 0.0,
        }
        for name in names
    }
    for summary in summaries:
        for player in summary["players"].values():
            agent_name = player["agent"]
            mechanics[agent_name]["games"] += 1.0
            for metric in [
                "settlements",
                "cities",
                "roads",
                "longest_road_length",
                "played_knights",
                "dev_cards_held",
                "hidden_vp_cards",
            ]:
                mechanics[agent_name][metric] += float(player[metric])
            mechanics[agent_name]["largest_army"] += float(player["largest_army"])
            mechanics[agent_name]["longest_road"] += float(player["longest_road"])

    def _mechanic_avg(agent_name: str, metric: str) -> float:
        agent_games = max(mechanics[agent_name]["games"], 1.0)
        return mechanics[agent_name][metric] / agent_games

    construction_metrics = [
        ("settlements", "Settlements"),
        ("cities", "Cities"),
        ("roads", "Roads"),
        ("longest_road_length", "Longest Road Len"),
    ]
    x_positions = range(len(names))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for metric_idx, (metric, label) in enumerate(construction_metrics):
        offset = (metric_idx - (len(construction_metrics) - 1) / 2) * width
        values = [_mechanic_avg(name, metric) for name in names]
        ax.bar([x + offset for x in x_positions], values, width=width, label=label)
    ax.set_title("Construction Mechanics By Agent")
    ax.set_ylabel("Average per game")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(names, rotation=15)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_construction_mechanics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    development_metrics = [
        ("played_knights", "Played Knights"),
        ("dev_cards_held", "Held Dev Cards"),
        ("hidden_vp_cards", "Hidden VP Cards"),
        ("largest_army", "Largest Army Rate"),
        ("longest_road", "Longest Road Rate"),
    ]
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for metric_idx, (metric, label) in enumerate(development_metrics):
        offset = (metric_idx - (len(development_metrics) - 1) / 2) * width
        values = [_mechanic_avg(name, metric) for name in names]
        ax.bar([x + offset for x in x_positions], values, width=width, label=label)
    ax.set_title("Development Cards And Awards By Agent")
    ax.set_ylabel("Average count or rate")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(names, rotation=15)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_development_mechanics.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    trade_actions = ["propose_trade", "accept_trade", "reject_trade", "counter_trade", "skip_trade", "bank_trade"]
    trade_action_counts = {name: {action: 0 for action in trade_actions} for name in names}
    for row in step_rows:
        agent = str(row.get("agent", ""))
        action = str(row.get("action", ""))
        if agent in trade_action_counts and action in trade_action_counts[agent]:
            trade_action_counts[agent][action] += 1

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottoms = [0] * len(names)
    for action in trade_actions:
        values = [trade_action_counts[name][action] for name in names]
        ax.bar(names, values, bottom=bottoms, label=action)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    ax.set_title("Trading Action Mix By Agent")
    ax.set_ylabel("Action count")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_trade_action_mix.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    trade_resource_rows = build_trade_resource_rows(step_rows)
    resource_totals = {
        name: {
            "offered": {resource: 0 for resource in RESOURCE_COLUMNS},
            "requested": {resource: 0 for resource in RESOURCE_COLUMNS},
            "bank_given": {resource: 0 for resource in RESOURCE_COLUMNS},
            "bank_received": {resource: 0 for resource in RESOURCE_COLUMNS},
        }
        for name in names
    }
    for row in trade_resource_rows:
        agent = str(row["agent"])
        resource = str(row["resource"])
        if agent not in resource_totals:
            continue
        for metric in ["offered", "requested", "bank_given", "bank_received"]:
            resource_totals[agent][metric][resource] = int(row[metric])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_titles = [
        ("offered", "Player Trade Resources Offered"),
        ("requested", "Player Trade Resources Requested"),
        ("bank_given", "Bank Trade Resources Given"),
        ("bank_received", "Bank Trade Resources Received"),
    ]
    x_positions = range(len(names))
    width = 0.15
    for axis, (metric, title) in zip(axes.flatten(), metric_titles):
        for resource_idx, resource in enumerate(RESOURCE_COLUMNS):
            offset = (resource_idx - (len(RESOURCE_COLUMNS) - 1) / 2) * width
            values = [resource_totals[name][metric][resource] for name in names]
            axis.bar([x + offset for x in x_positions], values, width=width, label=resource)
        axis.set_title(title)
        axis.set_ylabel("Resource count")
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(names, rotation=15)
    axes[0][0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_trade_resources.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist([int(summary["steps"]) for summary in summaries], bins=20, color="#457b9d")
    ax.set_title("Game Length Distribution")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Games")
    fig.tight_layout()
    path = figures_dir / "mixed_four_player_game_lengths.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mixed 4-player Catan agent match.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--enable-trading",
        dest="enable_trading",
        action="store_true",
        default=True,
        help="Enable player-to-player trading. On by default so learned trade heads are evaluated.",
    )
    parser.add_argument(
        "--disable-trading",
        dest="enable_trading",
        action="store_false",
        help="Disable player-to-player trading for no-trade smoke runs.",
    )
    parser.add_argument("--unified-checkpoint", type=str, default="checkpoints/unified_checkpoint_200.pt")
    parser.add_argument("--tom-dqn-checkpoint", type=str, default="")
    parser.add_argument("--dqn-checkpoint", type=str, default="")
    parser.add_argument("--hybrid-checkpoint", type=str, default="checkpoints/hybrid_checkpoint_200.pt")
    parser.add_argument("--ppo-checkpoint", type=str, default="")
    parser.add_argument("--orange-agent", choices=["tom_dqn", "dqn"], default="tom_dqn")
    parser.add_argument("--use-hybrid", action="store_true", help="Use hybrid in RED seat instead of raw PPO.")
    parser.add_argument(
        "--adjudicate-timeouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Assign a VP/progress winner when a game reaches max steps.",
    )
    parser.add_argument(
        "--trade-step-penalty",
        type=float,
        default=0.001,
        help="Per trade-phase decision penalty applied to learned agents in mixed-match logs.",
    )
    parser.add_argument(
        "--trade-active-penalty",
        type=float,
        default=0.002,
        help="Extra penalty for learned propose_trade and counter_trade decisions.",
    )
    parser.add_argument(
        "--trade-repeat-penalty",
        type=float,
        default=0.002,
        help="Additional growing penalty for repeated learned propose_trade/counter_trade loops.",
    )
    parser.add_argument(
        "--random-timeout-handicap",
        type=float,
        default=0.25,
        help="Timeout-only score handicap for the no-trade random baseline; natural 10 VP wins are unchanged.",
    )
    parser.add_argument(
        "--unified-timeout-bonus",
        type=float,
        default=0.0,
        help="Timeout-only score bonus for unified; natural 10 VP wins are unchanged.",
    )
    parser.add_argument(
        "--ppo-timeout-handicap",
        type=float,
        default=0.0,
        help="Timeout-only score handicap for PPO; natural 10 VP wins are unchanged.",
    )
    parser.add_argument(
        "--unified-timeout-min-margin-vs-random",
        type=float,
        default=0.0,
        help="Timeout-only floor that keeps unified at least this far above random no-trade when both are present.",
    )
    parser.add_argument(
        "--random-bank-trades",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow the random baseline to use bank trades. Off by default for a true no-trade control.",
    )
    parser.add_argument(
        "--random-bank-trade-probability",
        type=float,
        default=1.0,
        help="When random bank trades are enabled, probability of keeping bank_trade actions in its candidate set.",
    )
    parser.add_argument(
        "--rotate-seats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rotate which agent controls WHITE/BLUE/ORANGE/RED across games.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/mixed_four_player"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/mixed_four_player"))
    parser.add_argument("--log-file", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.", file=sys.stderr)
        args.device = "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    agents = build_agents(args)
    summaries: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_file or (args.output_dir / "mixed_four_player_training_log.txt")

    header = "\n".join(
        [
            f"Starting mixed four-player evaluation for {args.games} games",
            (
                f"Config: max_steps={args.max_steps} seed={args.seed} trading={int(args.enable_trading)} "
                f"deterministic={int(args.deterministic)} adjudicate_timeouts={int(args.adjudicate_timeouts)} "
                f"trade_step_penalty={args.trade_step_penalty:.4f} "
                f"trade_active_penalty={args.trade_active_penalty:.4f} "
                f"trade_repeat_penalty={args.trade_repeat_penalty:.4f} "
                f"random_timeout_handicap={args.random_timeout_handicap:.4f} "
                f"unified_timeout_bonus={args.unified_timeout_bonus:.4f} "
                f"ppo_timeout_handicap={args.ppo_timeout_handicap:.4f} "
                f"unified_timeout_min_margin_vs_random={args.unified_timeout_min_margin_vs_random:.4f} "
                f"random_bank_trades={int(args.random_bank_trades)} "
                f"random_bank_trade_probability={args.random_bank_trade_probability:.4f} "
                f"rotate_seats={int(args.rotate_seats)}"
            ),
            "Initial seat map:",
            *[
                f"  {player_id.name}: {agent.name}"
                for player_id, agent in build_seat_agents(0, agents, args.rotate_seats).items()
            ],
            "",
        ]
    )
    log_path.write_text(header, encoding="utf-8")
    print(header, end="")

    for game_idx in range(args.games):
        seat_agents = build_seat_agents(game_idx, agents, args.rotate_seats)
        summary, rows = run_one_game(game_idx, args, seat_agents)
        summaries.append(summary)
        all_step_rows.extend(rows)
        game_log = format_game_log(summary, seat_agents) + "\n\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(game_log)
        print(game_log, end="")

    final_summary = format_final_summary(summaries, agents)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(final_summary)
    print(final_summary, end="")
    print(f"Wrote {log_path}")
    steps_path, summary_path, action_mix_path, trade_resources_path = write_csv_outputs(
        summaries,
        all_step_rows,
        args.output_dir,
    )
    print(f"Wrote {steps_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {action_mix_path}")
    print(f"Wrote {trade_resources_path}")
    stats_path, stats_table_path = write_agent_stats_outputs(
        summaries,
        all_step_rows,
        agents,
        args.output_dir,
    )
    print(f"Wrote {stats_path}")
    print(f"Wrote {stats_table_path}")
    for figure_path in write_figures(summaries, all_step_rows, agents, args.figures_dir):
        print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
