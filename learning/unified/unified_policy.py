from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from learning.networks.opponent_need_predictor import OpponentNeedPredictor


class UnifiedActionHeads(nn.Module):
    def __init__(self, hidden_dim: int, gameplay_actions: int = 128, trade_targets: int = 3, resources: int = 5):
        super().__init__()
        self.gameplay_head = nn.Linear(hidden_dim, gameplay_actions)
        self.trade_action_head = nn.Linear(hidden_dim, 4)
        self.trade_target_head = nn.Linear(hidden_dim, trade_targets)
        self.trade_offer_head = nn.Linear(hidden_dim, resources)
        self.trade_request_head = nn.Linear(hidden_dim, resources)

    def forward(self, trunk: torch.Tensor, trade_context: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "gameplay": self.gameplay_head(trunk),
            "trade_action": self.trade_action_head(trunk),
            "trade_target": self.trade_target_head(trunk),
            "trade_offer": self.trade_offer_head(trade_context),
            "trade_request": self.trade_request_head(trade_context),
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
        self.action_heads = UnifiedActionHeads(hidden_dim=hidden_dim, resources=resources)

    def encode(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        board_emb = self.board_encoder(obs["board"])
        self_emb = self.self_encoder(obs["self"])
        opp_emb = self.opponent_encoder(obs["opponent"])

        trunk = self.fusion(torch.cat([board_emb, self_emb, opp_emb], dim=-1))
        need_pred = self.need_predictor(opp_emb)

        trade_context = torch.cat([trunk, need_pred], dim=-1)
        trade_context = trade_context[:, : trunk.shape[-1]]

        return trunk, need_pred

    def forward(self, obs: Dict[str, torch.Tensor]) -> tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor]]:
        trunk, need_pred = self.encode(obs)
        logits = self.action_heads(trunk, trunk)
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

        trade_action_dist = torch.distributions.Categorical(logits=logits["trade_action"])
        trade_target_dist = torch.distributions.Categorical(logits=logits["trade_target"])
        trade_offer_dist = torch.distributions.Categorical(logits=logits["trade_offer"])
        trade_request_dist = torch.distributions.Categorical(logits=logits["trade_request"])

        if deterministic:
            trade_action = torch.argmax(logits["trade_action"], dim=-1)
            trade_target = torch.argmax(logits["trade_target"], dim=-1)
            trade_offer = torch.argmax(logits["trade_offer"], dim=-1)
            trade_request = torch.argmax(logits["trade_request"], dim=-1)
        else:
            trade_action = trade_action_dist.sample()
            trade_target = trade_target_dist.sample()
            trade_offer = trade_offer_dist.sample()
            trade_request = trade_request_dist.sample()

        action_dict = {
            "action_type": trade_action,
            "target": trade_target,
            "offer": trade_offer,
            "request": trade_request,
        }
        log_prob_dict = {
            "action_type": trade_action_dist.log_prob(trade_action),
            "target": trade_target_dist.log_prob(trade_target),
            "offer": trade_offer_dist.log_prob(trade_offer),
            "request": trade_request_dist.log_prob(trade_request),
        }

        return value, action_dict, log_prob_dict, tom_outputs

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: Dict[str, torch.Tensor], phase: str):
        logits, value, tom_outputs = self.forward(obs)

        if phase == "gameplay":
            dist = torch.distributions.Categorical(logits=logits["gameplay"])
            log_prob = dist.log_prob(actions["gameplay_action"])
            entropy = dist.entropy().mean()
            return {"gameplay_action": log_prob}, entropy, value, tom_outputs

        d1 = torch.distributions.Categorical(logits=logits["trade_action"])
        d2 = torch.distributions.Categorical(logits=logits["trade_target"])
        d3 = torch.distributions.Categorical(logits=logits["trade_offer"])
        d4 = torch.distributions.Categorical(logits=logits["trade_request"])

        log_prob_dict = {
            "action_type": d1.log_prob(actions["action_type"]),
            "target": d2.log_prob(actions["target"]),
            "offer": d3.log_prob(actions["offer"]),
            "request": d4.log_prob(actions["request"]),
        }
        entropy = (d1.entropy() + d2.entropy() + d3.entropy() + d4.entropy()).mean() / 4.0
        return log_prob_dict, entropy, value, tom_outputs