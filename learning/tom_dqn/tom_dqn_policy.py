from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from ..networks.opponent_need_predictor import OpponentNeedPredictor


class ToMEnhancedDQNPolicy(nn.Module):
    """
    DQN policy enhanced with Theory of Mind (ToM) capabilities.
    Includes opponent need prediction and global state awareness for better trading decisions.
    """

    def __init__(
        self,
        board_dim: int = 64,
        self_dim: int = 64,
        opponent_dim: int = 64,
        global_dim: int = 265,
        hidden_dim: int = 192,
        resources: int = 5,
    ):
        super().__init__()

        # Enhanced encoders with global state
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
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Enhanced fusion with global context
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),  # board + self + opponent + global
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ToM-enhanced need predictor
        self.need_predictor = OpponentNeedPredictor(
            input_dim=hidden_dim * 2,  # opponent + global embeddings
            hidden_dim=hidden_dim,
            num_resources=resources,
        )

        # Separate value heads for gameplay and trade
        self.gameplay_value_head = nn.Linear(hidden_dim, 1)
        self.trade_value_head = nn.Linear(hidden_dim, 1)

        # Action heads with ToM integration
        self.gameplay_head = ToMGameplayHead(hidden_dim=hidden_dim)
        self.trade_head = ToMTradeHead(hidden_dim=hidden_dim, num_resources=resources)

    def encode(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        obs = {k: v.to(device) for k, v in obs.items() if isinstance(v, torch.Tensor)}
        board_emb = self.board_encoder(obs["board"])
        self_emb = self.self_encoder(obs["self"])
        opp_emb = self.opponent_encoder(obs["opponent"])

        global_emb = torch.zeros_like(opp_emb)
        if "global_state" in obs:
            global_emb = self.global_encoder(obs["global_state"])

        combined = torch.cat([board_emb, self_emb, opp_emb, global_emb], dim=-1)
        trunk = self.fusion(combined)

        # Predict opponent needs using opponent and global context
        need_pred = self.need_predictor(torch.cat([opp_emb, global_emb], dim=-1))

        return trunk, global_emb, need_pred

    def act(self, obs: Dict[str, torch.Tensor], phase: str, epsilon: float = 0.0) -> Dict[str, torch.Tensor]:
        """Selects an action based on the phase, using epsilon-greedy exploration with ToM."""
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
        trunk, global_emb, need_pred = self.encode(obs)
        q_values = self.gameplay_head(trunk, obs["gameplay_candidates"], obs["gameplay_mask"])
        batch_size = q_values.shape[0]
        device = q_values.device

        # Greedy actions
        greedy_actions = q_values.argmax(dim=-1)

        # Random actions for exploration
        mask = obs["gameplay_mask"].squeeze(1) if obs["gameplay_mask"].dim() == 3 else obs["gameplay_mask"]
        probs = mask.float()
        # Handle rows with no valid actions to avoid division by zero
        probs_sum = probs.sum(dim=-1, keepdim=True)
        safe_probs_sum = probs_sum.clamp(min=1.0)
        probs = probs / safe_probs_sum

        random_actions = torch.multinomial(probs, 1).squeeze(-1)

        # Choose between greedy and random based on epsilon
        is_random = torch.rand(batch_size, device=device) < epsilon
        action_indices = torch.where(is_random, random_actions, greedy_actions)

        return {"gameplay_action": action_indices}, q_values

    def get_trade_action(
        self,
        obs: Dict[str, torch.Tensor],
        epsilon: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        trunk, global_emb, need_pred = self.encode(obs)
        q_values = self.trade_head(trunk, obs["trade_candidates"], obs["trade_mask"], need_pred)
        batch_size = q_values.shape[0]
        device = q_values.device

        # Greedy actions
        greedy_actions = q_values.argmax(dim=-1)

        # Random actions for exploration
        mask = obs["trade_mask"].squeeze(1) if obs["trade_mask"].dim() == 3 else obs["trade_mask"]
        probs = mask.float()
        # Handle rows with no valid actions to avoid division by zero
        probs_sum = probs.sum(dim=-1, keepdim=True)
        safe_probs_sum = probs_sum.clamp(min=1.0)
        probs = probs / safe_probs_sum

        random_actions = torch.multinomial(probs, 1).squeeze(-1)

        # Choose between greedy and random based on epsilon
        is_random = torch.rand(batch_size, device=device) < epsilon
        action_indices = torch.where(is_random, random_actions, greedy_actions)

        return {"trade_action": action_indices}, q_values

    def get_value(self, obs: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        trunk, global_emb, need_pred = self.encode(obs)
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
        trunk, global_emb, need_pred = self.encode(obs)

        if phase == "gameplay":
            q_values = self.gameplay_head(trunk, obs["gameplay_candidates"], obs["gameplay_mask"])
            action_idx = actions["gameplay_action"]
            action_q = q_values.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
            log_prob = torch.log(torch.softmax(q_values, dim=-1).gather(1, action_idx.unsqueeze(-1)).squeeze(-1) + 1e-8)
        else:
            q_values = self.trade_head(trunk, obs["trade_candidates"], obs["trade_mask"], need_pred)
            action_idx = actions["trade_action"]
            action_q = q_values.gather(1, action_idx.unsqueeze(-1)).squeeze(-1)
            log_prob = torch.log(torch.softmax(q_values, dim=-1).gather(1, action_idx.unsqueeze(-1)).squeeze(-1) + 1e-8)

        value = self.get_value(obs, phase)
        tom_outputs = {"need_pred": need_pred}

        return {"gameplay_action": log_prob} if phase == "gameplay" else {"trade_action": log_prob}, action_q, value, tom_outputs


class ToMGameplayHead(nn.Module):
    """Gameplay action head - similar to original but can be extended with ToM features."""
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


class ToMTradeHead(nn.Module):
    """Trade action head enhanced with opponent need predictions for better trading decisions."""
    def __init__(self, hidden_dim: int, trade_feature_dim: int = 32, num_resources: int = 5):
        super().__init__()
        self.num_resources = num_resources

        self.trade_action_encoder = nn.Sequential(
            nn.Linear(trade_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Enhanced trade scorer that incorporates opponent needs
        self.trade_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_resources, hidden_dim),  # trunk + candidate + opponent_needs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trunk: torch.Tensor, trade_candidates: torch.Tensor, trade_mask: torch.Tensor, opponent_needs: torch.Tensor) -> torch.Tensor:
        candidate_emb = self.trade_action_encoder(trade_candidates)
        repeated_trunk = trunk.unsqueeze(1).expand(-1, candidate_emb.shape[1], -1)

        # Expand opponent needs to match trade candidates
        expanded_opponent_needs = opponent_needs.unsqueeze(1).expand(-1, candidate_emb.shape[1], -1)

        q_input = torch.cat([repeated_trunk, candidate_emb, expanded_opponent_needs], dim=-1)
        q_values = self.trade_scorer(q_input).squeeze(-1)
        q_values = q_values.masked_fill(~trade_mask.bool(), -1e9)
        return q_values