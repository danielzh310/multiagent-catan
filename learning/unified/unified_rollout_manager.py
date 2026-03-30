from __future__ import annotations

from typing import Any, Dict, List

import torch

from core.constants import PlayerId, Resource
from core.phase_router import TurnPhase
from environment.catan_env import CatanEnv


class UnifiedRolloutManager:
    def __init__(self, num_envs: int, device: str = "cpu", enable_trading: bool = True):
        self.num_envs = num_envs
        self.device = device
        self.enable_trading = enable_trading
        self.envs = [CatanEnv(enable_trading=enable_trading) for _ in range(num_envs)]
        self.obs = [env.reset() for env in self.envs]

    def _phase_name(self, env: CatanEnv) -> str:
        phase = env.get_phase()
        if phase in (TurnPhase.MAIN_ACTION, TurnPhase.END_TURN):
            return "gameplay"
        if phase in (TurnPhase.TRADE_PROPOSE, TurnPhase.TRADE_RESPOND):
            return "trade"
        return "auto"

    def _build_obs(self, env: CatanEnv) -> Dict[str, torch.Tensor]:
        raw = env.get_observation()
        player = raw["player"]
        other_players = [v for k, v in raw["players"].items() if k != raw["game"]["current_player"]]

        def to_vec(state: dict) -> List[float]:
            resources = state.get("resources", {})
            roads_val = state.get("roads", 0)
            if isinstance(roads_val, list):
                roads_val = len(roads_val)

            dev_cards_val = state.get("dev_cards", 0)
            if isinstance(dev_cards_val, list):
                dev_cards_val = len(dev_cards_val)

            vec = [
                float(resources.get("WOOD", 0)),
                float(resources.get("BRICK", 0)),
                float(resources.get("SHEEP", 0)),
                float(resources.get("WHEAT", 0)),
                float(resources.get("ORE", 0)),
                float(state.get("victory_points", 0)),
                float(state.get("num_settlements", 0)),
                float(state.get("num_cities", 0)),
                float(roads_val),
                float(state.get("bonus_vp", 0)),
                float(state.get("dev_victory_points", 0)),
                float(dev_cards_val),
            ]
            vec += [0.0] * (64 - len(vec))
            return vec[:64]

        self_vec = torch.tensor([to_vec(player)], dtype=torch.float32, device=self.device)

        op_vec = [0.0] * 64
        if other_players:
            nums = len(other_players)
            sum_vec = [0.0] * 64
            for opp in other_players:
                opp_v = to_vec(opp)
                for i in range(64):
                    sum_vec[i] += opp_v[i]
            op_vec = [x / nums for x in sum_vec]

        board_vec = torch.zeros((1, 64), dtype=torch.float32, device=self.device)
        board_vec[0, 0] = float(raw["game"].get("turn_number", 0))
        board_vec[0, 1] = float(int(env.get_current_player_id()))
        board_vec[0, 2] = float(env.get_phase().value)
        board_vec[0, 3] = 1.0 if raw["game"].get("enable_trading", True) else 0.0

        pending_trade = raw.get("trade")
        if pending_trade is not None:
            board_vec[0, 4] = 1.0
            board_vec[0, 5] = float(pending_trade.counter_count)

        return {
            "board": board_vec,
            "self": self_vec,
            "opponent": torch.tensor([op_vec], dtype=torch.float32, device=self.device),
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
        legal_actions = env.get_legal_actions()

        if env.get_phase() == TurnPhase.END_TURN:
            return {"type": "end_turn"}

        gameplay_actions = [
            a for a in legal_actions
            if a["type"] in {"build_road", "build_settlement", "build_city", "end_main_action"}
        ]

        if not gameplay_actions:
            return {"type": "end_main_action"}

        mapped_idx = int(action_idx) % len(gameplay_actions)
        return gameplay_actions[mapped_idx]

    def _decode_trade(self, action_dict: Dict[str, torch.Tensor], env: CatanEnv) -> dict:
        if not self.enable_trading:
            phase = env.get_phase()
            if phase == TurnPhase.TRADE_RESPOND:
                return {"type": "reject_trade", "response_type": "reject"}
            return {"type": "skip_trade"}

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