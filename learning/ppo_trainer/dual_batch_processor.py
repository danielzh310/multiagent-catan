from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


class DualBatchProcessor:
    """
    Processes separate rollout buffers for:
    - gameplay policy
    - trade policy

    Improvements in this version:
    - normalized returns for more stable critics
    - clipped advantages to reduce spikes
    """

    def __init__(
        self,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        normalize_returns: bool = True,
        advantage_clip: float = 5.0,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_returns = normalize_returns
        self.advantage_clip = advantage_clip

    def process_gameplay_rollouts(self, rollouts: List[dict]) -> Dict[str, Any]:
        if len(rollouts) == 0:
            return {}

        obs = torch.cat([item["obs"]["flat"] for item in rollouts], dim=0).to(self.device)
        actions = torch.stack([item["action"]["action_type"] for item in rollouts]).to(self.device)
        rewards = torch.tensor([item["reward"] for item in rollouts], dtype=torch.float32, device=self.device)
        values = torch.cat([item["value"] for item in rollouts], dim=0).squeeze(-1).to(self.device)
        log_probs = torch.cat([item["log_prob"] for item in rollouts], dim=0).squeeze(-1).to(self.device)
        dones = torch.tensor([item["done"] for item in rollouts], dtype=torch.float32, device=self.device)

        returns, advantages = self._compute_returns_advantages(rewards, values, dones)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "dones": dones,
        }

    def process_trade_rollouts(self, rollouts: List[dict]) -> Dict[str, Any]:
        if len(rollouts) == 0:
            return {}

        board = torch.cat([item["obs"]["board"] for item in rollouts], dim=0).to(self.device)
        self_state = torch.cat([item["obs"]["self"] for item in rollouts], dim=0).to(self.device)
        opponent = torch.cat([item["obs"]["opponent"] for item in rollouts], dim=0).to(self.device)

        trade_history = {
            "proposer_ids": torch.cat([item["obs"]["trade_history"]["proposer_ids"] for item in rollouts], dim=0).to(self.device),
            "target_ids": torch.cat([item["obs"]["trade_history"]["target_ids"] for item in rollouts], dim=0).to(self.device),
            "response_types": torch.cat([item["obs"]["trade_history"]["response_types"] for item in rollouts], dim=0).to(self.device),
            "offers": torch.cat([item["obs"]["trade_history"]["offers"] for item in rollouts], dim=0).to(self.device),
            "requests": torch.cat([item["obs"]["trade_history"]["requests"] for item in rollouts], dim=0).to(self.device),
            "accepted_flags": torch.cat([item["obs"]["trade_history"]["accepted_flags"] for item in rollouts], dim=0).to(self.device),
            "turn_numbers": torch.cat([item["obs"]["trade_history"]["turn_numbers"] for item in rollouts], dim=0).to(self.device),
        }

        action_type = torch.stack([item["action"]["action_type"] for item in rollouts]).to(self.device)
        target = torch.stack([item["action"]["target"] for item in rollouts]).to(self.device)
        offer = torch.stack([item["action"]["offer"] for item in rollouts]).to(self.device)
        request = torch.stack([item["action"]["request"] for item in rollouts]).to(self.device)

        rewards = torch.tensor([item["reward"] for item in rollouts], dtype=torch.float32, device=self.device)
        values = torch.cat([item["value"] for item in rollouts], dim=0).squeeze(-1).to(self.device)

        action_log_probs = []
        for item in rollouts:
            lp = 0.0
            for value in item["log_prob"].values():
                lp = lp + value.mean()
            action_log_probs.append(lp)
        log_probs = torch.stack(action_log_probs).to(self.device)

        dones = torch.tensor([item["done"] for item in rollouts], dtype=torch.float32, device=self.device)

        returns, advantages = self._compute_returns_advantages(rewards, values, dones)

        return {
            "obs": {
                "board": board,
                "self": self_state,
                "opponent": opponent,
                "trade_history": trade_history,
            },
            "actions": {
                "action_type": action_type,
                "target": target,
                "offer": offer,
                "request": request,
            },
            "rewards": rewards,
            "values": values,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "dones": dones,
        }

    def _compute_returns_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        next_value = 0.0
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]

        advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.normalize_returns:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns, advantages
