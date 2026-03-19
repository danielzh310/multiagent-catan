"""
Candidate action sampler for forward search.

Samples diverse actions from policy distributions.
"""

import torch


def sample_candidate_actions(action_outputs, num_samples=16):
    """
    Sample a set of structured actions from the policy distributions.
    """
    candidates = []

    for _ in range(num_samples):
        action = {}

        action["action_type"] = action_outputs["action_type"].sample()
        action["settlement"] = action_outputs["settlement"].sample()
        action["road"] = action_outputs["road"].sample()
        action["city"] = action_outputs["city"].sample()
        action["robber"] = action_outputs["robber"].sample()
        action["trade"] = action_outputs["trade"].sample()

        candidates.append(action)

    return candidates