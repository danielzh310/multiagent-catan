from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import torch


@dataclass
class GameplayTransition:
    obs: torch.Tensor
    action: int
    reward: float
    next_obs: torch.Tensor
    done: bool
    action_mask: torch.Tensor
    next_action_mask: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 100_000,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        self.capacity = capacity
        self.device = device
        self.buffer: Deque[GameplayTransition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(
        self,
        obs: torch.Tensor,
        action: int,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
        action_mask: torch.Tensor,
        next_action_mask: torch.Tensor,
    ) -> None:
        transition = GameplayTransition(
            obs=obs.detach().cpu().clone(),
            action=int(action),
            reward=float(reward),
            next_obs=next_obs.detach().cpu().clone(),
            done=bool(done),
            action_mask=action_mask.detach().cpu().clone(),
            next_action_mask=next_action_mask.detach().cpu().clone(),
        )
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict:
        if len(self.buffer) < batch_size:
            raise ValueError(
                f"ReplayBuffer has {len(self.buffer)} transitions, "
                f"but batch_size={batch_size} was requested."
            )

        batch: List[GameplayTransition] = self.rng.sample(list(self.buffer), batch_size)

        obs = torch.cat([item.obs for item in batch], dim=0).to(self.device)
        actions = torch.tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_obs = torch.cat([item.next_obs for item in batch], dim=0).to(self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        action_masks = torch.stack([item.action_mask for item in batch]).to(self.device)
        next_action_masks = torch.stack([item.next_action_mask for item in batch]).to(self.device)

        return {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "action_masks": action_masks,
            "next_action_masks": next_action_masks,
        }

    def clear(self) -> None:
        self.buffer.clear()