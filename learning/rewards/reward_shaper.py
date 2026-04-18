from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from core.constants import Resource


@dataclass
class RewardWeights:
    win_reward: float = 1.0
    lose_reward: float = -1.0

    vp_delta_weight: float = 0.16
    diversity_weight: float = 0.008
    build_progress_weight: float = 0.015

    trade_accept_bonus: float = 0.0 #0.002  # Reduced to prevent over-accepting
    trade_propose_bonus: float = 0.001  # Reduced to prevent over-proposing
    trade_counter_bonus: float = -0.0002  # Retain penalty for countering spam
    trade_reject_penalty: float = 0.0 #-0.001 # Reduced penalty for rejecting to allow strategic rejections without harsh punishment
    trade_skip_penalty: float = 0.0 # Neutral for single skip, to remove disincentive
    repeated_skip_penalty: float = 0 # reduced penalty for consecutive skips

    trade_long_term_scale: float = 1.0 # Retain 1.0 for direct r_trade scaling as per description
    tom_weight: float = 0.025

    reward_clip_min: float = -0.04
    reward_clip_max: float = 0.04


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

    def _clip(self, reward: float) -> float:
        return float(max(self.weights.reward_clip_min, min(self.weights.reward_clip_max, reward)))

    def gameplay_step_reward(
        self,
        prev_vp: float,
        curr_vp: float,
        prev_resources: Dict[Resource, int],
        curr_resources: Dict[Resource, int],
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

        return self._clip(float(reward))

    def trade_step_reward(
        self,
        action_type: str,
        reward_signal: float,
        consecutive_skips: int = 0,
        tom_loss: float = 0.0,
        curriculum_scale: float = 1.0,
    ) -> float:
        reward = self.weights.trade_long_term_scale * float(reward_signal)

        if action_type == "propose_trade":
            reward += self.weights.trade_propose_bonus
        elif action_type == "accept_trade":
            reward += self.weights.trade_accept_bonus
        elif action_type == "counter_trade":
            reward += self.weights.trade_counter_bonus
        elif action_type == "reject_trade":
            reward += self.weights.trade_reject_penalty
        elif action_type == "skip_trade":
            reward += self.weights.trade_skip_penalty
            if consecutive_skips >= 2:
                reward += self.weights.repeated_skip_penalty * float(consecutive_skips - 1)

        reward -= self.weights.tom_weight * float(tom_loss)

        reward *= float(curriculum_scale)
        return self._clip(float(reward))