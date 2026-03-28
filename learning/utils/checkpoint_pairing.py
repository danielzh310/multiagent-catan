from __future__ import annotations

import os
import torch
from typing import Tuple


def save_checkpoint_pair(
    gameplay_model,
    trade_model,
    save_dir: str,
    step: int,
):
    """
    Saves both models together so they stay synchronized.
    """

    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        "step": step,
        "gameplay_state_dict": gameplay_model.state_dict(),
        "trade_state_dict": trade_model.state_dict(),
    }

    path = os.path.join(save_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, path)

    return path


def load_checkpoint_pair(
    gameplay_model,
    trade_model,
    checkpoint_path: str,
    device: str = "cpu",
) -> Tuple[int, object, object]:
    """
    Loads both models from a paired checkpoint.
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)

    gameplay_model.load_state_dict(checkpoint["gameplay_state_dict"])
    trade_model.load_state_dict(checkpoint["trade_state_dict"])

    step = checkpoint.get("step", 0)

    return step, gameplay_model, trade_model


def list_checkpoints(save_dir: str):
    if not os.path.exists(save_dir):
        return []

    files = [f for f in os.listdir(save_dir) if f.endswith(".pt")]

    def extract_step(name: str):
        try:
            return int(name.split("_")[-1].split(".")[0])
        except Exception:
            return -1

    files = sorted(files, key=extract_step)

    return [os.path.join(save_dir, f) for f in files]