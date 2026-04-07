from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from learning.dqn.dqn_policy import DQNBaselinePolicy
from learning.ppo_trainer import PPOTrainer  # Assuming this exists


class HybridPolicy:
    """
    Hybrid policy that uses DQN for gameplay decisions and PPO for trading decisions.
    This allows independent optimization of each component while maintaining coordination.
    """

    def __init__(
        self,
        dqn_policy: DQNBaselinePolicy,
        trade_policy: Any,  # PPO policy for trade
        device: str = "cpu",
    ):
        self.dqn_policy = dqn_policy
        self.trade_policy = trade_policy
        self.device = device

    def get_gameplay_action(
        self,
        obs: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Use DQN for gameplay decisions
        return self.dqn_policy.get_gameplay_action(obs, epsilon)

    def get_trade_action(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # Use PPO for trade decisions
        # This should interface with the PPO policy's action selection
        # For now, return a placeholder - will be implemented when PPO is integrated
        return {"trade_action": torch.tensor([0])}, torch.tensor(0.0)

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        phase: str,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if phase == "gameplay":
            # Use DQN evaluation for gameplay
            return self.dqn_policy.evaluate_actions(obs, actions, phase)
        else:
            # Use PPO evaluation for trade
            # Placeholder - will be implemented when PPO is integrated
            log_prob = torch.tensor(0.0)
            entropy = torch.tensor(0.0)
            value = torch.tensor(0.0)
            tom_outputs = {}
            return {}, entropy, value, tom_outputs

    def get_value(self, obs: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        if phase == "gameplay":
            return self.dqn_policy.get_value(obs, phase)
        else:
            # Placeholder for PPO value
            return torch.tensor(0.0)