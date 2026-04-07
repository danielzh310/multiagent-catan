from __future__ import annotations

from .gameplay_policy import GameplayPolicy


def build_gameplay_model(device: str = "cpu") -> GameplayPolicy:
    """
    Factory function for the gameplay model.

    This keeps model construction centralized so training,
    evaluation, and rollout workers all use the same setup.
    """
    model = GameplayPolicy(
        device=device,
    )

    return model