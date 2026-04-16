from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple


def _infer_policy_type_from_path(path: Path) -> str:
    """Infers policy type from the checkpoint filename."""
    name = path.stem.lower()
    if "unified" in name:
        return "unified"
    if "tom_dqn" in name:
        return "tom_dqn"
    if "ppo" in name:
        return "ppo"
    if "dqn" in name:
        return "dqn"
    # Fallback for unknown or generic names
    return "unknown"

class LeagueManager:
    """
    Manages a "league" of saved agent checkpoints for self-play.

    This allows the agent to train against a diverse pool of its own past
    versions, which is a key technique for achieving robust performance in
    multi-agent settings.
    """

    def __init__(self, checkpoint_dir: str, frozen_ratio: float = 0.2, load_existing: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.frozen_ratio = max(0.0, min(1.0, frozen_ratio))
        self._checkpoints: List[Tuple[str, int, str]] = []  # List of (path, step, type)
        if load_existing:
            self.load_checkpoints()

    def __len__(self) -> int:
        """Returns the number of checkpoints in the league."""
        return len(self._checkpoints)

    def load_checkpoints(self):
        """Scans the checkpoint directory and loads existing checkpoints."""
        self._checkpoints = []
        if not self.checkpoint_dir.exists():
            return

        for f in self.checkpoint_dir.glob("*.pt"):
            try:
                # A bit of a hack to get the step number from the filename
                step_str = f.stem.split('_')[-1]
                step = int(step_str)
                policy_type = _infer_policy_type_from_path(f)
                if policy_type != "unknown":
                    self._checkpoints.append((str(f), step, policy_type))
            except (ValueError, IndexError):
                # Ignore files that don't match the naming convention
                continue

        # Sort by step number
        self._checkpoints.sort(key=lambda x: x[1])
        print(f"LeagueManager loaded {len(self)} checkpoints from {self.checkpoint_dir}")

    def maybe_add_checkpoint(self, path: str, step: int):
        """Adds a new checkpoint to the league."""
        policy_type = _infer_policy_type_from_path(Path(path))
        if policy_type == "unknown":
            return
        if any(existing_path == path for existing_path, _, _ in self._checkpoints):
            return
        self._checkpoints.append((path, step, policy_type))
        self._checkpoints.sort(key=lambda x: x[1])

    def sample_opponents(self, k: int, policy_type: str | None = None) -> List[str]:
        """Samples k opponent checkpoint paths from the league."""
        available = [entry for entry in self._checkpoints if policy_type is None or entry[2] == policy_type]
        if not available:
            return []

        if len(available) <= k:
            return [path for path, _, _ in available]

        frozen_count = int(round(k * self.frozen_ratio))
        frozen_count = min(frozen_count, max(0, len(available) - 1))
        recent_count = k - frozen_count

        split_index = max(1, len(available) // 2)
        frozen_pool = available[:split_index]
        recent_pool = available[split_index:]

        sampled: List[Tuple[str, int, str]] = []
        if frozen_count > 0 and frozen_pool:
            sampled.extend(random.sample(frozen_pool, min(frozen_count, len(frozen_pool))))
        if recent_count > 0 and recent_pool:
            sampled.extend(random.sample(recent_pool, min(recent_count, len(recent_pool))))

        if len(sampled) < k:
            remaining = [entry for entry in available if entry not in sampled]
            sampled.extend(random.sample(remaining, min(k - len(sampled), len(remaining))))

        random.shuffle(sampled)
        return [path for path, _, _ in sampled]