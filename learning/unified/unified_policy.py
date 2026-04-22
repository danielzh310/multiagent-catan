from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from learning.networks.opponent_need_predictor import OpponentNeedPredictor


class UnifiedActionHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        gameplay_feature_dim: int = 40,
        trade_feature_dim: int = 32,
        num_resources: int = 5,
    ):
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

        self.trade_action_encoder = nn.Sequential(
            nn.Linear(trade_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Bilateral trade scorer incorporates opponent needs for joint optimization
        self.trade_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_resources, hidden_dim),  # trunk + candidate + opponent_needs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        trunk: torch.Tensor,
        gameplay_candidates: torch.Tensor,
        gameplay_mask: torch.Tensor,
        trade_candidates: torch.Tensor,
        trade_mask: torch.Tensor,
        opponent_needs: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        candidate_emb = self.gameplay_action_encoder(gameplay_candidates)
        repeated_trunk = trunk.unsqueeze(1).expand(-1, candidate_emb.shape[1], -1)
        gameplay_logits = self.gameplay_scorer(torch.cat([repeated_trunk, candidate_emb], dim=-1)).squeeze(-1)
        gameplay_logits = gameplay_logits.masked_fill(~gameplay_mask.bool(), -1e9)

        trade_candidate_emb = self.trade_action_encoder(trade_candidates)
        repeated_trade_trunk = trunk.unsqueeze(1).expand(-1, trade_candidate_emb.shape[1], -1)

        # Bilateral trade optimization: condition on opponent needs for joint surplus
        if opponent_needs is not None:
            # Expand opponent needs to match trade candidates
            expanded_opponent_needs = opponent_needs.unsqueeze(1).expand(-1, trade_candidate_emb.shape[1], -1)
            trade_input = torch.cat([repeated_trade_trunk, trade_candidate_emb, expanded_opponent_needs], dim=-1)
        else:
            # Fallback to basic scoring if no opponent needs available
            num_res = self.trade_scorer[0].in_features - (trunk.shape[-1] * 2)
            device = trade_candidate_emb.device
            dummy_needs = torch.zeros(trade_candidate_emb.shape[0], trade_candidate_emb.shape[1], num_res, device=device)
            trade_input = torch.cat([repeated_trade_trunk, trade_candidate_emb, dummy_needs], dim=-1)

        trade_logits = self.trade_scorer(trade_input).squeeze(-1)
        trade_logits = trade_logits.masked_fill(~trade_mask.bool(), -1e9)

        return {
            "gameplay": gameplay_logits,
            "trade": trade_logits,
        }


class UnifiedPolicy(nn.Module):
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

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.need_predictor = OpponentNeedPredictor(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            num_resources=resources,
        )

        self.gameplay_value_head = nn.Linear(hidden_dim, 1)
        self.trade_value_head = nn.Linear(hidden_dim, 1)
        self.central_value_head = nn.Linear(hidden_dim * 2, 1)
        self.action_heads = UnifiedActionHeads(
            hidden_dim=hidden_dim,
            num_resources=resources,
        )

    def encode(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        obs = {k: v.to(device) for k, v in obs.items() if isinstance(v, torch.Tensor)}
        board_emb = self.board_encoder(obs["board"])
        self_emb = self.self_encoder(obs["self"])
        opp_emb = self.opponent_encoder(obs["opponent"])

        global_emb = torch.zeros_like(opp_emb)
        if "global_state" in obs:
            global_emb = self.global_encoder(obs["global_state"])

        trunk = self.fusion(torch.cat([board_emb, self_emb, opp_emb], dim=-1))
        
        # Feed the original (non-detached) features to need_predictor so it learns well.
        # Gradient stopping is handled at the loss level (delayed activation in trainer).
        need_pred = self.need_predictor(torch.cat([opp_emb, global_emb], dim=-1))

        return trunk, global_emb, need_pred

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        trunk, global_emb, need_pred = self.encode(obs)
        logits = self.action_heads(
            trunk,
            obs["gameplay_candidates"],
            obs["gameplay_mask"],
            obs["trade_candidates"],
            obs["trade_mask"],
            opponent_needs=need_pred,  # Pass opponent needs for bilateral trade optimization
        )
        tom_outputs = {"need_pred": need_pred}
        return logits, trunk, tom_outputs, global_emb

    def _phase_value(self, trunk: torch.Tensor, global_emb: torch.Tensor, phase: str) -> torch.Tensor:
        if global_emb is not None:
            combined = torch.cat([trunk, global_emb], dim=-1)
            return self.central_value_head(combined)

        if phase == "gameplay":
            return self.gameplay_value_head(trunk)
        return self.trade_value_head(trunk)

    def act(self, obs: Dict[str, torch.Tensor], phase: str, deterministic: bool = False):
        logits, trunk, tom_outputs, global_emb = self.forward(obs)
        value = self._phase_value(trunk, global_emb, phase)

        if phase == "gameplay":
            dist = torch.distributions.Categorical(logits=logits["gameplay"])
            action = torch.argmax(logits["gameplay"], dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

            return value, {"gameplay_action": action}, {"gameplay_action": log_prob}, tom_outputs

        dist = torch.distributions.Categorical(logits=logits["trade"])
        action = torch.argmax(logits["trade"], dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return value, {"trade_action": action}, {"trade_action": log_prob}, tom_outputs

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor], phase: str):
        logits, trunk, tom_outputs, global_emb = self.forward(obs)
        value = self._phase_value(trunk, global_emb, phase)

        if phase == "gameplay":
            dist = torch.distributions.Categorical(logits=logits["gameplay"])
            log_prob = dist.log_prob(actions["gameplay_action"])
            entropy = dist.entropy().mean()
            return {"gameplay_action": log_prob}, entropy, value, tom_outputs

        dist = torch.distributions.Categorical(logits=logits["trade"])
        log_prob = dist.log_prob(actions["trade_action"])
        entropy = dist.entropy().mean()
        return {"trade_action": log_prob}, entropy, value, tom_outputs
