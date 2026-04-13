from __future__ import annotations

from typing import Dict, Any

from ..dqn.dqn_trainer import DQNTrainer


def build_gameplay_q_model(config: Dict[str, Any]) -> DQNTrainer:
    hidden_dim = config.get("hidden_dim", 256)
    board_dim = config.get("board_dim", 64)
    self_dim = config.get("self_dim", 64)
    opponent_dim = config.get("opponent_dim", 64)
    resources = config.get("resources", 5)
    lr = config.get("lr", 1e-3)
    gamma = config.get("gamma", 0.99)
    target_update_freq = config.get("target_update_freq", 1000)
    device = config.get("device", "cpu")

    trainer = DQNTrainer(
        board_dim=board_dim,
        self_dim=self_dim,
        opponent_dim=opponent_dim,
        device=device,
        lr=lr,
        gamma=gamma,
        target_update_freq=target_update_freq,
        hidden_dim=hidden_dim,
        resources=resources,
    )

    return trainer
