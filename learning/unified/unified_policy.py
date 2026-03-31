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
        trade_targets: int = 3,
        resources: int = 5,
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

        self.trade_engage_head = nn.Linear(hidden_dim, 2)
        self.trade_response_head = nn.Linear(hidden_dim, 4)
        self.trade_target_head = nn.Linear(hidden_dim, trade_targets)
        self.trade_offer_head = nn.Linear(hidden_dim, resources)
        self.trade_request_head = nn.Linear(hidden_dim, resources)

    def forward(self, trunk: torch.Tensor, gameplay_candidates: torch.Tensor, gameplay_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        candidate_emb = self.gameplay_action_encoder(gameplay_candidates)
        repeated_trunk = trunk.unsqueeze(1).expand(-1, candidate_emb.shape[1], -1)
        gameplay_logits = self.gameplay_scorer(torch.cat([repeated_trunk, candidate_emb], dim=-1)).squeeze(-1)
        gameplay_logits = gameplay_logits.masked_fill(~gameplay_mask.bool(), -1e9)

        return {
            "gameplay": gameplay_logits,
            "trade_engage": self.trade_engage_head(trunk),
            "trade_response": self.trade_response_head(trunk),
            "trade_target": self.trade_target_head(trunk),
            "trade_offer": self.trade_offer_head(trunk),
            "trade_request": self.trade_request_head(trunk),
        }


class UnifiedPolicy(nn.Module):
    def __init__(
        self,
        board_dim: int = 64,
        self_dim: int = 64,
        opponent_dim: int = 64,
        hidden_dim: int = 128,
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

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.need_predictor = OpponentNeedPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_resources=resources,
        )

        self.value_head = nn.Linear(hidden_dim, 1)
        self.action_heads = UnifiedActionHeads(
            hidden_dim=hidden_dim,
            resources=resources,
        )

    def encode(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        board_emb = self.board_encoder(obs["board"])
        self_emb = self.self_encoder(obs["self"])
        opp_emb = self.opponent_encoder(obs["opponent"])

        trunk = self.fusion(torch.cat([board_emb, self_emb, opp_emb], dim=-1))
        need_pred = self.need_predictor(opp_emb)

        return trunk, need_pred

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        trunk, need_pred = self.encode(obs)
        logits = self.action_heads(
            trunk,
            obs["gameplay_candidates"],
            obs["gameplay_mask"],
        )
        value = self.value_head(trunk)
        tom_outputs = {"need_pred": need_pred}
        return logits, value, tom_outputs

    def act(self, obs: Dict[str, torch.Tensor], phase: str, deterministic: bool = False):
        logits, value, tom_outputs = self.forward(obs)

        if phase == "gameplay":
            dist = torch.distributions.Categorical(logits=logits["gameplay"])
            action = torch.argmax(logits["gameplay"], dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

            return value, {"gameplay_action": action}, {"gameplay_action": log_prob}, tom_outputs

        engage_dist = torch.distributions.Categorical(logits=logits["trade_engage"])
        response_dist = torch.distributions.Categorical(logits=logits["trade_response"])
        target_dist = torch.distributions.Categorical(logits=logits["trade_target"])
        offer_dist = torch.distributions.Categorical(logits=logits["trade_offer"])
        request_dist = torch.distributions.Categorical(logits=logits["trade_request"])

        if deterministic:
            engage = torch.argmax(logits["trade_engage"], dim=-1)
            response = torch.argmax(logits["trade_response"], dim=-1)
            target = torch.argmax(logits["trade_target"], dim=-1)
            offer = torch.argmax(logits["trade_offer"], dim=-1)
            request = torch.argmax(logits["trade_request"], dim=-1)
        else:
            engage = engage_dist.sample()
            response = response_dist.sample()
            target = target_dist.sample()
            offer = offer_dist.sample()
            request = request_dist.sample()

        action_dict = {
            "engage_trade": engage,
            "trade_response": response,
            "target": target,
            "offer": offer,
            "request": request,
        }

        log_prob_dict = {
            "engage_trade": engage_dist.log_prob(engage),
            "trade_response": response_dist.log_prob(response),
            "target": target_dist.log_prob(target),
            "offer": offer_dist.log_prob(offer),
            "request": request_dist.log_prob(request),
        }

        return value, action_dict, log_prob_dict, tom_outputs

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor], phase: str):
        logits, value, tom_outputs = self.forward(obs)

        if phase == "gameplay":
            dist = torch.distributions.Categorical(logits=logits["gameplay"])
            log_prob = dist.log_prob(actions["gameplay_action"])
            entropy = dist.entropy().mean()
            return {"gameplay_action": log_prob}, entropy, value, tom_outputs

        d_engage = torch.distributions.Categorical(logits=logits["trade_engage"])
        d_response = torch.distributions.Categorical(logits=logits["trade_response"])
        d_target = torch.distributions.Categorical(logits=logits["trade_target"])
        d_offer = torch.distributions.Categorical(logits=logits["trade_offer"])
        d_request = torch.distributions.Categorical(logits=logits["trade_request"])

        log_prob_dict = {
            "engage_trade": d_engage.log_prob(actions["engage_trade"]),
            "trade_response": d_response.log_prob(actions["trade_response"]),
            "target": d_target.log_prob(actions["target"]),
            "offer": d_offer.log_prob(actions["offer"]),
            "request": d_request.log_prob(actions["request"]),
        }

        entropy = (
            d_engage.entropy()
            + d_response.entropy()
            + d_target.entropy()
            + d_offer.entropy()
            + d_request.entropy()
        ).mean() / 5.0

        return log_prob_dict, entropy, value, tom_outputs
