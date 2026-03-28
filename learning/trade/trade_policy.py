from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from learning.trade.trade_history_encoder import TradeHistoryEncoder
from learning.trade.cross_attention_fusion import CrossAttentionFusion
from learning.trade.trade_heads import TradeActionHeads
from learning.trade.tom_head import ToMHead


class TradePolicy(nn.Module):
    """
    Trade policy with ToM + PPO.

    Components:
    - simple encoders for board/self/opponent
    - trade history encoder
    - cross-attention fusion
    - shared trunk
    - trade action heads
    - ToM auxiliary head
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.device = device
        self.use_lstm = False

        embed_dim = 128

        # simple encoders
        self.board_encoder = nn.Linear(64, embed_dim)
        self.self_encoder = nn.Linear(64, embed_dim)
        self.opponent_encoder = nn.Linear(64, embed_dim)

        self.trade_history_encoder = TradeHistoryEncoder(
            embed_dim=64,
            hidden_dim=embed_dim,
        )

        self.fusion = CrossAttentionFusion(embed_dim=embed_dim)

        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        self.action_heads = TradeActionHeads(hidden_dim=embed_dim)
        self.value_head = nn.Linear(embed_dim, 1)

        self.tom_head = ToMHead(hidden_dim=embed_dim)

        self.to(self.device)

    def forward(self, obs: Dict[str, torch.Tensor]):
        board = obs["board"]
        self_state = obs["self"]
        opponent = obs["opponent"]
        trade_seq = obs["trade_history"]

        board_emb = self.board_encoder(board)
        self_emb = self.self_encoder(self_state)
        opp_emb = self.opponent_encoder(opponent)

        trade_emb = self.trade_history_encoder(trade_seq)

        fused = self.fusion(board_emb, self_emb, opp_emb, trade_emb)

        features = self.trunk(fused)

        logits = self.action_heads(features)
        value = self.value_head(features)

        tom_outputs = self.tom_head(features)

        return logits, value, tom_outputs

    def act(
        self,
        obs: Dict[str, torch.Tensor],
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
        hidden_state=None,
        done_mask=None,
        deterministic: bool = False,
    ):
        logits, value, tom_outputs = self.forward(obs)

        if action_masks is not None:
            logits = self.action_heads.apply_action_mask(logits, action_masks)

        action_dict, log_probs = self.action_heads.sample_action(
            logits,
            deterministic=deterministic,
        )

        return value, action_dict, log_probs, hidden_state, tom_outputs

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        action_dict: Dict[str, torch.Tensor],
        action_masks: Optional[Dict[str, torch.Tensor]] = None,
        tom_targets: Optional[Dict[str, torch.Tensor]] = None,
    ):
        logits, value, tom_outputs = self.forward(obs)

        if action_masks is not None:
            logits = self.action_heads.apply_action_mask(logits, action_masks)

        log_probs, entropy = self.action_heads.evaluate_actions(
            logits,
            action_dict,
        )

        tom_loss = 0.0
        if tom_targets is not None:
            tom_loss = self.tom_head.compute_loss(tom_outputs, tom_targets)

        return value, log_probs, entropy, tom_loss