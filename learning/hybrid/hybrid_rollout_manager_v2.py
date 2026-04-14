from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from core.constants import Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv
from learning.hybrid.hybrid_gameplay_obs_v2 import build_gameplay_state_vector
from learning.hybrid.hybrid_trade_obs_v2 import ACTION_TYPE_INDEX, build_trade_action_masks, build_trade_obs


class HybridRolloutManagerV2:
    MAX_GAMEPLAY_ACTIONS = 256

    def __init__(self, num_envs: int, device: str = "cpu", enable_trading: bool = True, seed: int = 0):
        self.num_envs = num_envs
        self.device = device
        self.enable_trading = enable_trading
        self.envs = [CatanEnv(enable_trading=enable_trading, seed=seed + i) for i in range(num_envs)]
        self.obs = [env.reset() for env in self.envs]

    def _route_phase(self, env: CatanEnv) -> str:
        phase = env.get_phase()
        if phase in (TurnPhase.SETUP, TurnPhase.MAIN_ACTION, TurnPhase.END_TURN):
            return "gameplay"
        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            return "trade"
        return "auto"

    def _build_gameplay_obs(self, env: CatanEnv) -> torch.Tensor:
        vec = build_gameplay_state_vector(env)
        return torch.tensor([vec], dtype=torch.float32, device=self.device)

    def _resolve_discard_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if "resources" in action:
            return action

        required = int(action.get("required", 0))
        available = action.get("available", {})
        resources_to_discard = {
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        }
        remaining = required
        ordered = sorted(available.items(), key=lambda item: (-int(item[1]), item[0]))
        for resource_name, count in ordered:
            if remaining <= 0:
                break
            take = min(int(count), remaining)
            if take <= 0:
                continue
            try:
                resource = Resource[resource_name]
            except KeyError:
                continue
            resources_to_discard[resource] = take
            remaining -= take

        resolved = dict(action)
        if remaining == 0:
            resolved["resources"] = resources_to_discard
        return resolved

    def _build_gameplay_action_mask(self, env: CatanEnv) -> torch.Tensor:
        legal_actions = env.get_legal_actions()
        mask = torch.zeros(self.MAX_GAMEPLAY_ACTIONS, dtype=torch.float32, device=self.device)
        for idx, _ in enumerate(legal_actions[: self.MAX_GAMEPLAY_ACTIONS]):
            mask[idx] = 1.0
        if mask.sum() == 0:
            mask[0] = 1.0
        return mask

    def _decode_gameplay_action(self, action_idx: int, env: CatanEnv) -> dict:
        if env.get_phase() == TurnPhase.END_TURN:
            return {"type": "end_turn"}

        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return {"type": "end_main_action"}

        mapped_idx = min(max(int(action_idx), 0), len(legal_actions) - 1)
        chosen = legal_actions[mapped_idx]
        if chosen.get("type") == "discard_cards":
            return self._resolve_discard_action(chosen)
        return chosen

    def _argmax_resource(self, vec: torch.Tensor) -> Resource:
        idx = int(torch.argmax(vec, dim=-1).item())
        return [Resource.WOOD, Resource.BRICK, Resource.SHEEP, Resource.WHEAT, Resource.ORE][idx]

    def _pick_matching_trade_action(self, legal_actions: List[dict], action_dict: Dict[str, torch.Tensor], phase: TurnPhase) -> dict:
        desired_type = int(action_dict["action_type"].item())
        desired_target = int(action_dict["target"].item())
        desired_offer = self._argmax_resource(action_dict["offer"])
        desired_request = self._argmax_resource(action_dict["request"])

        if phase == TurnPhase.TRADE_PROPOSE:
            if desired_type == ACTION_TYPE_INDEX["skip_trade"]:
                for action in legal_actions:
                    if action.get("type") == "skip_trade":
                        return action

            proposals = [action for action in legal_actions if action.get("type") == "propose_trade"]
            if not proposals:
                return legal_actions[0]

            by_target = [action for action in proposals if int(action.get("target")) == desired_target]
            candidates = by_target or proposals
            exact = [
                action for action in candidates
                if action.get("offer", {}).get(desired_offer, 0) > 0
                and action.get("request", {}).get(desired_request, 0) > 0
            ]
            return exact[0] if exact else candidates[0]

        if phase == TurnPhase.TRADE_RESPOND:
            if desired_type == ACTION_TYPE_INDEX["accept_trade"]:
                for action in legal_actions:
                    if action.get("type") == "accept_trade":
                        return action

            if desired_type == ACTION_TYPE_INDEX["reject_trade"]:
                for action in legal_actions:
                    if action.get("type") == "reject_trade":
                        return action

            counters = [action for action in legal_actions if action.get("type") == "counter_trade"]
            if counters:
                exact = [
                    action for action in counters
                    if action.get("counter_offer", {}).get(desired_offer, 0) > 0
                    and action.get("counter_request", {}).get(desired_request, 0) > 0
                ]
                return exact[0] if exact else counters[0]

            for action in legal_actions:
                if action.get("type") == "reject_trade":
                    return action

        return legal_actions[0]

    def _decode_trade_action(self, action_dict: Dict[str, torch.Tensor], env: CatanEnv) -> dict:
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            if env.get_phase() == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}
        return self._pick_matching_trade_action(legal_actions, action_dict, env.get_phase())

    def collect(
        self,
        dqn_trainer,
        epsilon: float,
        trade_policy,
        steps: int = 64,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:
        gameplay_transitions: List[Dict[str, Any]] = []
        trade_rollouts: List[Dict[str, Any]] = []
        stats = {
            "gameplay_rollouts": 0,
            "trade_rollouts": 0,
            "gameplay_reward_mean": 0.0,
            "trade_reward_mean": 0.0,
            "trade_propose_count": 0,
            "trade_accept_count": 0,
            "trade_reject_count": 0,
            "trade_counter_count": 0,
            "trade_skip_count": 0,
        }

        for _ in range(steps):
            for i, env in enumerate(self.envs):
                route = self._route_phase(env)

                if route == "gameplay":
                    obs_tensor = self._build_gameplay_obs(env)
                    action_mask = self._build_gameplay_action_mask(env)
                    action_idx = dqn_trainer.act(obs_tensor, action_mask.unsqueeze(0), epsilon)
                    env_action = self._decode_gameplay_action(action_idx, env)
                    _, reward, done, _ = env.step(env_action)

                    next_obs_tensor = self._build_gameplay_obs(env)
                    next_route = self._route_phase(env)
                    next_action_mask = self._build_gameplay_action_mask(env) if next_route == "gameplay" else torch.zeros_like(action_mask)
                    if next_action_mask.sum() == 0:
                        next_action_mask[0] = 1.0

                    gameplay_transitions.append(
                        {
                            "obs": obs_tensor.detach().cpu(),
                            "action": int(action_idx),
                            "reward": float(reward),
                            "next_obs": next_obs_tensor.detach().cpu(),
                            "done": bool(done or next_route != "gameplay"),
                            "action_mask": action_mask.detach().cpu(),
                            "next_action_mask": next_action_mask.detach().cpu(),
                            "env_action": env_action,
                        }
                    )
                    stats["gameplay_rollouts"] += 1
                    stats["gameplay_reward_mean"] += float(reward)
                    self.obs[i] = env.reset() if done else env.get_observation()
                    continue

                if route == "trade":
                    obs_dict = build_trade_obs(env, self.device)
                    action_masks = build_trade_action_masks(env, self.device)
                    with torch.no_grad():
                        value, action_dict, log_prob, _, _ = trade_policy.act(
                            obs_dict,
                            action_masks=action_masks,
                            deterministic=False,
                        )

                    env_action = self._decode_trade_action(action_dict, env)
                    _, reward, done, _ = env.step(env_action)
                    trade_rollouts.append(
                        {
                            "obs": {
                                "board": obs_dict["board"].detach().cpu().clone(),
                                "self": obs_dict["self"].detach().cpu().clone(),
                                "opponent": obs_dict["opponent"].detach().cpu().clone(),
                                "trade_history": {k: v.detach().cpu().clone() for k, v in obs_dict["trade_history"].items()},
                            },
                            "action": {k: v.detach().cpu().clone() for k, v in action_dict.items()},
                            "log_prob": {k: v.detach().cpu().clone() for k, v in log_prob.items()},
                            "value": value.detach().cpu().clone(),
                            "reward": float(reward),
                            "done": bool(done),
                            "action_masks": {k: v.detach().cpu().clone() for k, v in action_masks.items()},
                            "env_action": env_action,
                        }
                    )
                    stats["trade_rollouts"] += 1
                    stats["trade_reward_mean"] += float(reward)

                    action_type = env_action.get("type", "")
                    if action_type == "propose_trade":
                        stats["trade_propose_count"] += 1
                    elif action_type == "accept_trade":
                        stats["trade_accept_count"] += 1
                    elif action_type == "reject_trade":
                        stats["trade_reject_count"] += 1
                    elif action_type == "counter_trade":
                        stats["trade_counter_count"] += 1
                    elif action_type == "skip_trade":
                        stats["trade_skip_count"] += 1

                    self.obs[i] = env.reset() if done else env.get_observation()
                    continue

                _, _, done, _ = env.step(None)
                self.obs[i] = env.reset() if done else env.get_observation()

        if stats["gameplay_rollouts"] > 0:
            stats["gameplay_reward_mean"] /= stats["gameplay_rollouts"]
        if stats["trade_rollouts"] > 0:
            stats["trade_reward_mean"] /= stats["trade_rollouts"]

        return gameplay_transitions, trade_rollouts, stats
