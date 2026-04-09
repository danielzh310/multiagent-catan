from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Dict, Any

import torch


@dataclass
class CatanTransition:
    obs: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_obs: Dict[str, Any]
    done: bool
    phase: str


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.device = device
        self.buffer: Deque[CatanTransition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_obs: Dict[str, Any],
        done: bool,
        phase: str,
    ) -> None:
        transition = CatanTransition(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            done=bool(done),
            phase=phase,
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[CatanTransition]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return self.rng.sample(list(self.buffer), batch_size)

    def clear(self) -> None:
        self.buffer.clear()