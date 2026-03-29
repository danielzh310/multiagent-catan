from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.constants import Resource


@dataclass
class RewardWeights:
    win_reward: float = 1.0
    lose_reward: float = -1.0
    vp_delta_weight: float = 0.20
    diversity_weight: float = 0.01
    build_progress_weight: float = 0.02
    trade_weight: float = 1.0
    tom_weight: float = 0.05


def resource_diversity(resources: Dict[Resource, int]) -> int:
    return sum(1 for v in resources.values() if int(v) > 0)


def estimate_build_progress(resources: Dict[Resource, int]) -> float:
    wood = float(resources.get(Resource.WOOD, 0))
    brick = float(resources.get(Resource.BRICK, 0))
    sheep = float(resources.get(Resource.SHEEP, 0))
    wheat = float(resources.get(Resource.WHEAT, 0))
    ore = float(resources.get(Resource.ORE, 0))

    settlement_progress = min(wood, 1.0) + min(brick, 1.0) + min(sheep, 1.0) + min(wheat, 1.0)
    city_progress = min(wheat / 2.0, 1.0) + min(ore / 3.0, 1.0)
    dev_progress = min(sheep, 1.0) + min(wheat, 1.0) + min(ore, 1.0)

    return (settlement_progress / 4.0 + city_progress / 2.0 + dev_progress / 3.0) / 3.0


class RewardShaper:
    def __init__(self, weights: RewardWeights | None = None):
        self.weights = weights or RewardWeights()

    def step_reward(
        self,
        prev_vp: float,
        curr_vp: float,
        prev_resources: Dict[Resource, int],
        curr_resources: Dict[Resource, int],
        trade_reward: float = 0.0,
        tom_loss: float = 0.0,
        won: bool = False,
        lost: bool = False,
    ) -> float:
        reward = 0.0

        if won:
            reward += self.weights.win_reward
        if lost:
            reward += self.weights.lose_reward

        reward += self.weights.vp_delta_weight * max(curr_vp - prev_vp, 0.0)

        prev_div = resource_diversity(prev_resources)
        curr_div = resource_diversity(curr_resources)
        reward += self.weights.diversity_weight * max(curr_div - prev_div, 0.0)

        prev_prog = estimate_build_progress(prev_resources)
        curr_prog = estimate_build_progress(curr_resources)
        reward += self.weights.build_progress_weight * (curr_prog - prev_prog)

        reward += self.weights.trade_weight * trade_reward
        reward -= self.weights.tom_weight * float(tom_loss)

        return float(reward)