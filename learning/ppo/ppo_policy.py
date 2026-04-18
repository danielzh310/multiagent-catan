from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class PPOPolicy(nn.Module):
    """
    Basic PPO policy for Catan with separate actor-critic heads for gameplay and trade phases.
    This is a raw PPO implementation without unified architecture or advanced ToM.
    """

    def __init__(
        self,
        board_dim: int = 64,
        self_dim: int = 64,
        opponent_dim: int = 64,
        hidden_dim: int = 192,
        resources: int = 5,
        gameplay_feature_dim: int = 40,
        trade_feature_dim: int = 32,
    ):
        super().__init__()

        # Encoders
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

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Separate value heads
        self.gameplay_value_head = nn.Linear(hidden_dim, 1)
        self.trade_value_head = nn.Linear(hidden_dim, 1)

        # Actor heads (separate for gameplay and trade)
        self.gameplay_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, gameplay_feature_dim),
            nn.ReLU(),
        )
        self.trade_actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, trade_feature_dim),
            nn.ReLU(),
        )

    def encode(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        obs = {k: v.to(device) for k, v in obs.items() if isinstance(v, torch.Tensor)}
        board_emb = self.board_encoder(obs["board"])
        self_emb = self.self_encoder(obs["self"])
        opp_emb = self.opponent_encoder(obs["opponent"])

        combined = torch.cat([board_emb, self_emb, opp_emb], dim=-1)
        trunk = self.fusion(combined)
        return trunk

    def get_gameplay_logits(self, trunk: torch.Tensor, gameplay_candidates: torch.Tensor, gameplay_mask: torch.Tensor) -> torch.Tensor:
        """Get logits for gameplay actions."""
        actor_emb = self.gameplay_actor(trunk)
        # Simple dot-product attention-like scoring
        scores = torch.matmul(gameplay_candidates, actor_emb.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~gameplay_mask.bool(), -1e9)
        return scores

    def get_trade_logits(self, trunk: torch.Tensor, trade_candidates: torch.Tensor, trade_mask: torch.Tensor) -> torch.Tensor:
        """Get logits for trade actions."""
        actor_emb = self.trade_actor(trunk)
        # Simple dot-product attention-like scoring
        scores = torch.matmul(trade_candidates, actor_emb.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(~trade_mask.bool(), -1e9)
        return scores

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        trunk = self.encode(obs)

        gameplay_logits = self.get_gameplay_logits(
            trunk, obs["gameplay_candidates"], obs["gameplay_mask"]
        )
        trade_logits = self.get_trade_logits(
            trunk, obs["trade_candidates"], obs["trade_mask"]
        )

        logits = {
            "gameplay": gameplay_logits,
            "trade": trade_logits,
        }

        gameplay_value = self.gameplay_value_head(trunk)
        trade_value = self.trade_value_head(trunk)

        return logits, gameplay_value, trade_value

    def get_value(self, obs: Dict[str, torch.Tensor], phase: str) -> torch.Tensor:
        trunk = self.encode(obs)
        if phase == "gameplay":
            return self.gameplay_value_head(trunk)
        else:
            return self.trade_value_head(trunk)

    def act(self, obs: Dict[str, torch.Tensor], phase: str, deterministic: bool = False):
        logits, gameplay_value, trade_value = self.forward(obs)

        device = gameplay_value.device
        batch_size = gameplay_value.shape[0]
        dummy_action = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
        dummy_log_prob = torch.zeros((batch_size, 1), device=device)

        if phase == "gameplay":
            dist = torch.distributions.Categorical(logits=logits["gameplay"])
            action = torch.argmax(logits["gameplay"], dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

            actions = {"gameplay_action": action.unsqueeze(-1), "trade_action": dummy_action}
            log_probs = {"gameplay_action": log_prob.unsqueeze(-1), "trade_action": dummy_log_prob}
            return gameplay_value, actions, log_probs

        # phase == "trade"
        dist = torch.distributions.Categorical(logits=logits["trade"])
        action = torch.argmax(logits["trade"], dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)

        actions = {"gameplay_action": dummy_action, "trade_action": action.unsqueeze(-1)}
        log_probs = {"gameplay_action": dummy_log_prob, "trade_action": log_prob.unsqueeze(-1)}
        return trade_value, actions, log_probs

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor], phase: str):
        logits, gameplay_value, trade_value = self.forward(obs)
        device = gameplay_value.device
        batch_size = gameplay_value.shape[0]
        dummy_log_prob = torch.zeros((batch_size, 1), device=device)

        if phase == "gameplay":
            dist = torch.distributions.Categorical(logits=logits["gameplay"])
            log_prob = dist.log_prob(actions["gameplay_action"].squeeze(-1))
            entropy = dist.entropy().mean()
            log_probs = {
                "gameplay_action": log_prob.unsqueeze(-1),
                "trade_action": dummy_log_prob,
            }
            return log_probs, entropy, gameplay_value

        # phase == "trade"
        dist = torch.distributions.Categorical(logits=logits["trade"])
        log_prob = dist.log_prob(actions["trade_action"].squeeze(-1))
        entropy = dist.entropy().mean()
        log_probs = {
            "gameplay_action": dummy_log_prob,
            "trade_action": log_prob.unsqueeze(-1),
        }
        return log_probs, entropy, trade_value