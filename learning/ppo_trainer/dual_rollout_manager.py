from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from environment.catan_env import CatanEnv
from core.phase_router import TurnPhase
from core.constants import PlayerId, Resource


class DualRolloutManager:
    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        gameplay_builder,
        trade_builder,
        device: str = "cpu",
        seed: int = 0,
        debug: bool = False,
    ):
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = device
        self.debug = debug

        self.envs = [CatanEnv(seed=seed + i) for i in range(num_envs)]

        self.gameplay_policies = [gameplay_builder(device=device) for _ in range(num_envs)]
        self.trade_policies = [trade_builder(device=device) for _ in range(num_envs)]

        self.reset()

    def reset(self):
        self.obs = [env.reset() for env in self.envs]
        self.gameplay_storage = []
        self.trade_storage = []

    def _clone_tensor_dict(self, obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().clone()

        if isinstance(obj, dict):
            return {k: self._clone_tensor_dict(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._clone_tensor_dict(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self._clone_tensor_dict(v) for v in obj)

        return obj

    def _build_gameplay_obs(self, env: CatanEnv) -> Dict[str, torch.Tensor]:
        raw = env.build_gameplay_observation()

        flat = torch.tensor(
            [float(raw["turn_number"]) % 100] * 64,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        return {"flat": flat}

    def _build_trade_obs(self, env: CatanEnv) -> Dict[str, Any]:
        def zeros(shape: Tuple[int, ...]) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.float32, device=self.device)

        return {
            "board": zeros((1, 64)),
            "self": zeros((1, 64)),
            "opponent": zeros((1, 64)),
            "trade_history": {
                "proposer_ids": torch.zeros((1, 4), dtype=torch.long, device=self.device),
                "target_ids": torch.zeros((1, 4), dtype=torch.long, device=self.device),
                "response_types": torch.zeros((1, 4), dtype=torch.long, device=self.device),
                "offers": torch.zeros((1, 4, 5), dtype=torch.float32, device=self.device),
                "requests": torch.zeros((1, 4, 5), dtype=torch.float32, device=self.device),
                "accepted_flags": torch.zeros((1, 4), dtype=torch.float32, device=self.device),
                "turn_numbers": torch.zeros((1, 4), dtype=torch.float32, device=self.device),
            },
        }

    def _default_empty_trade_vector(self) -> Dict[Resource, int]:
        return {
            Resource.WOOD: 0,
            Resource.BRICK: 0,
            Resource.SHEEP: 0,
            Resource.WHEAT: 0,
            Resource.ORE: 0,
        }

    def _one_hot_trade_vector(self, idx: int) -> Dict[Resource, int]:
        vector = self._default_empty_trade_vector()
        resources = [
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ]
        if 0 <= idx < len(resources):
            vector[resources[idx]] = 1
        return vector

    def _decode_gameplay_action(self, action_dict: Dict[str, torch.Tensor], env: CatanEnv) -> dict:
        phase = env.get_phase()
        if phase == TurnPhase.END_TURN:
            return {"type": "end_turn"}

        idx = int(action_dict["action_type"].detach().cpu().item())

        if idx == 0:
            return {"type": "build_road"}
        if idx == 1:
            return {"type": "build_settlement"}
        if idx == 2:
            return {"type": "build_city"}

        return {"type": "end_main_action"}

    def _decode_trade_action(self, action_dict: Dict[str, torch.Tensor], env: CatanEnv) -> dict:
        phase = env.get_phase()

        if phase == TurnPhase.TRADE_PROPOSE:
            action_idx = int(action_dict["action_type"].detach().cpu().item())
            target_idx = int(action_dict["target"].detach().cpu().item())

            players = [PlayerId.WHITE, PlayerId.BLUE, PlayerId.ORANGE, PlayerId.RED]
            current_player = env.get_current_player_id()
            legal_targets = [p for p in players if p != current_player]

            if len(legal_targets) == 0:
                return {"type": "skip_trade"}

            target = legal_targets[target_idx % len(legal_targets)]

            offer_idx = int(torch.argmax(action_dict["offer"], dim=-1).detach().cpu().item())
            request_idx = int(torch.argmax(action_dict["request"], dim=-1).detach().cpu().item())

            if action_idx == 0:
                return {
                    "type": "propose_trade",
                    "target": target,
                    "offer": self._one_hot_trade_vector(offer_idx),
                    "request": self._one_hot_trade_vector(request_idx),
                }

            return {"type": "skip_trade"}

        if phase == TurnPhase.TRADE_RESPOND:
            action_idx = int(action_dict["action_type"].detach().cpu().item())
            offer_idx = int(torch.argmax(action_dict["offer"], dim=-1).detach().cpu().item())
            request_idx = int(torch.argmax(action_dict["request"], dim=-1).detach().cpu().item())

            if action_idx == 1:
                return {"type": "accept_trade", "response_type": "accept"}

            if action_idx == 2:
                return {"type": "reject_trade", "response_type": "reject"}

            if action_idx == 3:
                return {
                    "type": "counter_trade",
                    "response_type": "counter",
                    "counter_offer": self._one_hot_trade_vector(offer_idx),
                    "counter_request": self._one_hot_trade_vector(request_idx),
                }

            return {"type": "reject_trade", "response_type": "reject"}

        return {"type": "skip_trade"}

    def _maybe_reset_env(self, env: CatanEnv, done: bool) -> dict:
        if done:
            return env.reset()
        return env.get_observation()

    def _build_rollout_stats(self, gameplay_storage: list, trade_storage: list) -> Dict[str, Any]:
        stats = {
            "gameplay_rollouts": len(gameplay_storage),
            "trade_rollouts": len(trade_storage),
            "gameplay_reward_mean": 0.0,
            "trade_reward_mean": 0.0,
            "trade_propose_count": 0,
            "trade_accept_count": 0,
            "trade_reject_count": 0,
            "trade_counter_count": 0,
            "trade_skip_count": 0,
            "gameplay_action_counts": {
                "build_road": 0,
                "build_settlement": 0,
                "build_city": 0,
                "end_main_action": 0,
                "end_turn": 0,
            },
        }

        if len(gameplay_storage) > 0:
            stats["gameplay_reward_mean"] = sum(item["reward"] for item in gameplay_storage) / len(gameplay_storage)

            for item in gameplay_storage:
                env_action = item.get("env_action", {})
                action_type = env_action.get("type")
                if action_type in stats["gameplay_action_counts"]:
                    stats["gameplay_action_counts"][action_type] += 1

        if len(trade_storage) > 0:
            stats["trade_reward_mean"] = sum(item["reward"] for item in trade_storage) / len(trade_storage)

            for item in trade_storage:
                env_action = item.get("env_action", {})
                action_type = env_action.get("type")

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

        return stats

    def collect(self) -> Tuple[list, list, Dict[str, Any]]:
        self.gameplay_storage = []
        self.trade_storage = []

        for _ in range(self.num_steps):
            for i, env in enumerate(self.envs):
                controller = env.get_active_model_name()

                if self.debug:
                    print(f"[DEBUG] Controller: {controller}")
                    print(f"[DEBUG] Phase: {env.get_phase()}")

                if controller == "gameplay":
                    obs = self._build_gameplay_obs(env)

                    value, action_dict, log_prob, _, _ = self.gameplay_policies[i].act(
                        obs,
                        deterministic=False,
                    )

                    env_action = self._decode_gameplay_action(action_dict, env)
                    _, reward, done, _ = env.step(env_action)

                    self.gameplay_storage.append({
                        "obs": self._clone_tensor_dict(obs),
                        "action": self._clone_tensor_dict(action_dict),
                        "env_action": env_action,
                        "reward": float(reward),
                        "value": self._clone_tensor_dict(value),
                        "log_prob": self._clone_tensor_dict(log_prob),
                        "done": bool(done),
                    })

                    self.obs[i] = self._maybe_reset_env(env, done)

                elif controller == "trade":
                    obs = self._build_trade_obs(env)

                    value, action_dict, log_prob, _, tom = self.trade_policies[i].act(
                        obs,
                        deterministic=False,
                    )

                    env_action = self._decode_trade_action(action_dict, env)
                    _, reward, done, _ = env.step(env_action)

                    self.trade_storage.append({
                        "obs": self._clone_tensor_dict(obs),
                        "action": self._clone_tensor_dict(action_dict),
                        "env_action": env_action,
                        "reward": float(reward),
                        "value": self._clone_tensor_dict(value),
                        "log_prob": self._clone_tensor_dict(log_prob),
                        "done": bool(done),
                        "tom": self._clone_tensor_dict(tom),
                    })

                    self.obs[i] = self._maybe_reset_env(env, done)

                else:
                    _, reward, done, _ = env.step(None)
                    self.obs[i] = self._maybe_reset_env(env, done)

        gameplay_out = self._clone_tensor_dict(self.gameplay_storage)
        trade_out = self._clone_tensor_dict(self.trade_storage)
        rollout_stats = self._build_rollout_stats(gameplay_out, trade_out)

        return gameplay_out, trade_out, rollout_stats
