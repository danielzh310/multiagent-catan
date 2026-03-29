from __future__ import annotations

from typing import Any, Dict, List

import torch

from core.constants import PlayerId, Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv


class UnifiedRolloutManager:
    def __init__(self, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.envs = [CatanEnv() for _ in range(num_envs)]
        self.obs = [env.reset() for env in self.envs]

    def _phase_name(self, env: CatanEnv) -> str:
        phase = env.get_phase()
        if phase in (TurnPhase.MAIN_ACTION, TurnPhase.END_TURN):
            return "gameplay"
        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            return "trade"
        return "auto"

    def _build_obs(self, env: CatanEnv) -> Dict[str, torch.Tensor]:
        z = lambda shape: torch.zeros(shape, dtype=torch.float32, device=self.device)
        return {
            "board": z((1, 64)),
            "self": z((1, 64)),
            "opponent": z((1, 64)),
        }

    def _one_hot_trade_vector(self, idx: int) -> Dict[Resource, int]:
        resources = [
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ]
        out = {r: 0 for r in resources}
        if 0 <= idx < len(resources):
            out[resources[idx]] = 1
        return out

    def _decode_gameplay(self, action_idx: int, env: CatanEnv) -> dict:
        if env.get_phase() == TurnPhase.END_TURN:
            return {"type": "end_turn"}

        if action_idx == 0:
            return {"type": "build_road"}
        if action_idx == 1:
            return {"type": "build_settlement"}
        if action_idx == 2:
            return {"type": "build_city"}

        return {"type": "end_main_action"}

    def _decode_trade(self, action_dict: Dict[str, torch.Tensor], env: CatanEnv) -> dict:
        phase = env.get_phase()

        engage = int(action_dict["engage_trade"].item())
        response = int(action_dict["trade_response"].item())

        if engage == 0:
            return {"type": "skip_trade"}

        if phase == TurnPhase.TRADE_PROPOSE:
            players = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]
            current_player = env.get_current_player_id()
            legal_targets = [p for p in players if p != current_player]
            if not legal_targets:
                return {"type": "skip_trade"}

            target_idx = int(action_dict["target"].item()) % len(legal_targets)
            target = legal_targets[target_idx]

            offer_idx = int(action_dict["offer"].item())
            request_idx = int(action_dict["request"].item())

            return {
                "type": "propose_trade",
                "target": target,
                "offer": self._one_hot_trade_vector(offer_idx),
                "request": self._one_hot_trade_vector(request_idx),
            }

        if phase == TurnPhase.TRADE_RESPOND:
            if response == 0:
                return {"type": "accept_trade", "response_type": "accept"}

            if response == 1:
                return {"type": "reject_trade", "response_type": "reject"}

            if response == 2:
                offer_idx = int(action_dict["offer"].item())
                request_idx = int(action_dict["request"].item())
                return {
                    "type": "counter_trade",
                    "response_type": "counter",
                    "counter_offer": self._one_hot_trade_vector(offer_idx),
                    "counter_request": self._one_hot_trade_vector(request_idx),
                }

            return {"type": "reject_trade", "response_type": "reject"}

        return {"type": "skip_trade"}

    def collect(self, policy, steps: int = 128) -> List[Dict[str, Any]]:
        storage: List[Dict[str, Any]] = []

        for _ in range(steps):
            for i, env in enumerate(self.envs):
                phase_name = self._phase_name(env)

                if phase_name == "auto":
                    _, reward, done, info = env.step(None)
                    if done:
                        self.obs[i] = env.reset()
                    else:
                        self.obs[i] = env.get_observation()
                    continue

                obs = self._build_obs(env)
                value, action_dict, log_prob_dict, tom_outputs = policy.act(
                    obs=obs,
                    phase=phase_name,
                    deterministic=False,
                )

                if phase_name == "gameplay":
                    env_action = self._decode_gameplay(int(action_dict["gameplay_action"].item()), env)
                else:
                    env_action = self._decode_trade(action_dict, env)

                _, reward, done, info = env.step(env_action)

                storage.append(
                    {
                        "obs": {k: v.detach().cpu().clone() for k, v in obs.items()},
                        "phase": phase_name,
                        "action": {k: v.detach().cpu().clone() for k, v in action_dict.items()},
                        "log_prob": {k: v.detach().cpu().clone() for k, v in log_prob_dict.items()},
                        "value": value.detach().cpu().clone(),
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                        "env_action": env_action,
                    }
                )

                if done:
                    self.obs[i] = env.reset()
                else:
                    self.obs[i] = env.get_observation()

        return storage