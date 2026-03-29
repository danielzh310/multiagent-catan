from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FrozenCheckpoint:
    path: str
    step: int


class LeagueManager:
    def __init__(
        self,
        checkpoint_dir: str,
        frozen_ratio: float = 0.2,
        max_frozen: int = 10,
        seed: int = 0,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.frozen_ratio = float(frozen_ratio)
        self.max_frozen = int(max_frozen)
        self.rng = random.Random(seed)
        self.pool: List[FrozenCheckpoint] = []

    def maybe_add_checkpoint(self, path: str, step: int) -> None:
        self.pool.append(FrozenCheckpoint(path=path, step=step))
        self.pool = sorted(self.pool, key=lambda x: x.step)[-self.max_frozen :]

    def sample_opponent_checkpoint(self) -> Optional[str]:
        if len(self.pool) == 0:
            return None

        if self.rng.random() > self.frozen_ratio:
            return None

        return self.rng.choice(self.pool).path