from __future__ import annotations

from typing import Dict, Any

from ..dqn.dqn_trainer import DQNTrainer


def build_gameplay_q_model(config: Dict[str, Any]) -> DQNTrainer:
    state_dim = config["state_dim"]
    num_actions = config["num_actions"]

    hidden_dim = config.get("hidden_dim", 256)
    lr = config.get("lr", 1e-3)
    gamma = config.get("gamma", 0.99)
    target_update_freq = config.get("target_update_freq", 1000)
    device = config.get("device", "cpu")

    trainer = DQNTrainer(
        state_dim=state_dim,
        num_actions=num_actions,
        device=device,
        lr=lr,
        gamma=gamma,
        target_update_freq=target_update_freq,
        hidden_dim=hidden_dim,
    )

    return trainer