"""
Forward search policy wrapper.

Uses a learned policy as a prior and performs rollout-based lookahead
to choose better actions.
"""

import torch
import numpy as np

from learning.search_policy.sample_actions import sample_candidate_actions
from learning.search_policy.worker import simulate_action_sequence


class ForwardSearchPolicy:
    def __init__(
        self,
        base_policy,
        num_simulations=16,
        rollout_depth=5,
        device="cpu",
    ):
        self.base_policy = base_policy
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.device = device

    def act(self, env, obs, hidden_state, action_mask):
        """
        Run forward search to select best action.
        """

        with torch.no_grad():
            _, action_outputs, _ = self.base_policy.forward(
                obs=obs,
                action_masks=action_mask,
                hidden_state=hidden_state,
                done_mask=torch.ones(1, 1, device=self.device),
            )

        candidates = sample_candidate_actions(
            action_outputs,
            num_samples=self.num_simulations,
        )

        best_score = -np.inf
        best_action = None

        for action in candidates:
            score = simulate_action_sequence(
                env=env,
                action=action,
                policy=self.base_policy,
                depth=self.rollout_depth,
                device=self.device,
            )

            if score > best_score:
                best_score = score
                best_action = action

        return best_action