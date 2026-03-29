from __future__ import annotations

import os
from typing import Any, Dict

import torch


class HybridCheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _build_path(self, step: int) -> str:
        return os.path.join(self.checkpoint_dir, f"hybrid_checkpoint_{step}.pt")

    def save(
        self,
        step: int,
        dqn_trainer: Any,
        trade_trainer: Any,
        extra: Dict[str, Any] | None = None,
    ) -> str:
        path = self._build_path(step)

        checkpoint = {
            "step": step,
            "dqn": {
                "q_network": dqn_trainer.q_network.state_dict(),
                "target_q_network": dqn_trainer.target_q_network.state_dict(),
                "optimizer": dqn_trainer.optimizer.state_dict(),
                "train_step_count": getattr(dqn_trainer, "train_step_count", 0),
            },
            "trade": {
                "policy": getattr(trade_trainer, "policy").state_dict(),
                "optimizer": getattr(trade_trainer, "optimizer").state_dict(),
            },
        }

        if extra is not None:
            checkpoint["extra"] = extra

        torch.save(checkpoint, path)
        return path

    def load(
        self,
        path: str,
        dqn_trainer: Any,
        trade_trainer: Any,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        checkpoint = torch.load(path, map_location=map_location)

        dqn_trainer.q_network.load_state_dict(checkpoint["dqn"]["q_network"])
        dqn_trainer.target_q_network.load_state_dict(checkpoint["dqn"]["target_q_network"])
        dqn_trainer.optimizer.load_state_dict(checkpoint["dqn"]["optimizer"])
        if "train_step_count" in checkpoint["dqn"]:
            dqn_trainer.train_step_count = checkpoint["dqn"]["train_step_count"]

        trade_trainer.policy.load_state_dict(checkpoint["trade"]["policy"])
        trade_trainer.optimizer.load_state_dict(checkpoint["trade"]["optimizer"])

        return checkpoint