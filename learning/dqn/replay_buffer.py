from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Dict, Any

import torch


@dataclass
class CatanTransition:
    obs: Dict[str, torch.Tensor]
    action: Dict[str, torch.Tensor]
    reward: float
    next_obs: Dict[str, torch.Tensor]
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
        # Convert to tensors if needed
        obs_tensors = {k: torch.tensor(v, dtype=torch.float32).detach().cpu().clone()
                      for k, v in obs.items()}
        action_tensors = {k: torch.tensor(v, dtype=torch.long).detach().cpu().clone()
                         for k, v in action.items()}
        next_obs_tensors = {k: torch.tensor(v, dtype=torch.float32).detach().cpu().clone()
                           for k, v in next_obs.items()}

        transition = CatanTransition(
            obs=obs_tensors,
            action=action_tensors,
            reward=float(reward),
            next_obs=next_obs_tensors,
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
            next_obs=next_obs.detach().cpu().clone(),
            done=bool(done),
            action_mask=action_mask.detach().cpu().clone(),
            next_action_mask=next_action_mask.detach().cpu().clone(),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[CatanTransition]:
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return self.rng.sample(list(self.buffer), batch_size)

    def clear(self) -> None:
        self.buffer.clear()