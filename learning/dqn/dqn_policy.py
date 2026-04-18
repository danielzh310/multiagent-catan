from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from ..networks.opponent_need_predictor import OpponentNeedPredictor


class DQNGameplayHead(nn.Module):
    def __init__(self, hidden_dim: int, gameplay_feature_dim: int = 40):
        super().__init__()
        self.gameplay_action_encoder = nn.Sequential(
            nn.Linear(gameplay_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gameplay_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trunk: torch.Tensor, gameplay_candidates: torch.Tensor, gameplay_mask: torch.Tensor) -> torch.Tensor:
        candidate_emb = self.gameplay_action_encoder(gameplay_candidates)
        repeated_trunk = trunk.unsqueeze(1).expand(-1, candidate_emb.shape[1], -1)
        q_values = self.gameplay_scorer(torch.cat([repeated_trunk, candidate_emb], dim=-1)).squeeze(-1)
        q_values = q_values.masked_fill(~gameplay_mask.bool(), -1e9)
        return q_values


class DQNTradeHead(nn.Module):
    def __init__(self, hidden_dim: int, trade_feature_dim: int = 32):
        super().__init__()
        self.trade_action_encoder = nn.Sequential(
            nn.Linear(trade_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.trade_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trunk: torch.Tensor, trade_candidates: torch.Tensor, trade_mask: torch.Tensor) -> torch.Tensor:
        candidate_emb = self.trade_action_encoder(trade_candidates)
        repeated_trunk = trunk.unsqueeze(1).expand(-1, candidate_emb.shape[1], -1)
        q_values = self.trade_scorer(torch.cat([repeated_trunk, candidate_emb], dim=-1)).squeeze(-1)
        q_values = q_values.masked_fill(~trade_mask.bool(), -1e9)
        return q_values


class DQNBaselinePolicy(nn.Module):
    """
    DQN baseline policy that can handle both gameplay and trading phases.
    This is a simplified version that uses Q-learning for all decisions.
    """

    def __init__(
        self,
        board_dim: int = 64,
        self_dim: int = 64,
        opponent_dim: int = 64,
        hidden_dim: int = 192,
        resources: int = 5,
        device: str = "cpu",
    ):
        super().__init__()

        self.device = device

        # Simple encoders (simplified compared to unified model)
        self.board_encoder = nn.Sequential(
            nn.Linear(board_dim, hidden_dim),
            nn.ReLU(),
        )
        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
        )
        self.opponent_encoder = nn.Sequential(
            nn.Linear(opponent_dim, hidden_dim),
            nn.ReLU(),
        )

        # Simple fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate value heads for gameplay and trade
        self.gameplay_value_head = nn.Linear(hidden_dim, 1)
        self.trade_value_head = nn.Linear(hidden_dim, 1)

        # Action heads
        self.gameplay_head = DQNGameplayHead(hidden_dim=hidden_dim)
        self.trade_head = DQNTradeHead(hidden_dim=hidden_dim)

        # Simplified need predictor for ToM (optional for baseline)
        self.need_predictor = OpponentNeedPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_resources=resources,
        )

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        obs = {k: v.to(self.device) for k, v in obs.items() if isinstance(v, torch.Tensor)}
        board_emb = self.board_encoder(obs["board"])
        self_emb = self.self_encoder(obs["self"])
        opp_emb = self.opponent_encoder(obs["opponent"])

        combined = torch.cat([board_emb, self_emb, opp_emb], dim=-1)
        trunk = self.fusion(combined)
        return trunk

    def act(self, obs: Dict[str, torch.Tensor], phase: str, epsilon: float = 0.0) -> Dict[str, torch.Tensor]:
        """Selects an action based on the phase, using epsilon-greedy exploration."""
        if phase == "gameplay":
            action_dict, _ = self.get_gameplay_action(obs, epsilon)
        else:
            action_dict, _ = self.get_trade_action(obs, epsilon)
        return action_dict

    def get_gameplay_action(
        self,
        obs: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        trunk = self.encode(obs)
        q_values = self.gameplay_head(
            trunk=trunk,
            gameplay_candidates=obs["gameplay_candidates"],
            gameplay_mask=obs["gameplay_mask"]
        )

        if torch.rand(1).item() < epsilon:
            valid_actions = torch.nonzero(obs["gameplay_mask"].squeeze(0) > 0, as_tuple=False).squeeze(-1)
            action_idx = valid_actions[torch.randint(0, len(valid_actions), (1,))].item()
        else:
            action_idx = q_values.argmax().item()

        return {"gameplay_action": torch.tensor([action_idx])}, q_values

    def get_trade_action(
        self,
        obs: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        trunk = self.encode(obs)
        q_values = self.trade_head(
            trunk=trunk,
            trade_candidates=obs["trade_candidates"],
            trade_mask=obs["trade_mask"],
        )

        if torch.rand(1).item() < epsilon:
            valid_actions = torch.nonzero(obs["trade_mask"].squeeze(0) > 0, as_tuple=False).squeeze(-1)
            action_idx = valid_actions[torch.randint(0, len(valid_actions), (1,))].item()
        else:
            action_idx = q_values.argmax().item()

        return {"trade_action": torch.tensor([action_idx])}, q_values

    def get_value(self, obs: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        trunk = self.encode(obs)
        if phase == "gameplay":
            return self.gameplay_value_head(trunk)
        else:
            return self.trade_value_head(trunk)

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        phase: str,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        trunk = self.encode(obs)

        if phase == "gameplay":
            q_values = self.gameplay_head(
                trunk=trunk,
                gameplay_candidates=obs["gameplay_candidates"],
                gameplay_mask=obs["gameplay_mask"]
            )
            action_idx = actions["gameplay_action"]
            action_q = q_values.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
            log_prob = torch.log(torch.softmax(q_values, dim=-1).gather(1, action_idx.unsqueeze(-1)).squeeze(-1) + 1e-8)
        else:
            q_values = self.trade_head(trunk)
            action_idx = actions["trade_action"]
            action_q = q_values.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
            log_prob = torch.log(torch.softmax(q_values, dim=-1).gather(1, action_idx.unsqueeze(-1)).squeeze(-1) + 1e-8)

        value = self.get_value(obs, phase)
        tom_outputs = self.need_predictor(trunk) if hasattr(self, 'need_predictor') else {}

        return {"action": log_prob}, torch.tensor(0.0), value, tom_outputs