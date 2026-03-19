"""
Opponent scheduling for self-play.

This file decides how past policies are sampled and assigned
to opponent slots during training.
"""

import numpy as np


def get_sampling_distribution(num_policies, recent_window=800, recent_mass=0.5):
    """
    Build a probability distribution over stored policies.

    Older policies still get sampled, but newer ones are favored.
    """
    if num_policies <= 0:
        raise ValueError("num_policies must be positive")

    probs = ((1.0 - recent_mass) / num_policies) * np.ones(num_policies, dtype=np.float64)

    window = min(recent_window, num_policies)
    if window > 0:
        height = (2.0 * recent_mass) / (window + 1)
        step = height / window

        recent_bonus = np.zeros(window, dtype=np.float64)
        for i in range(window):
            recent_bonus[i] = i * step

        probs[-window:] += recent_bonus

    probs = probs / probs.sum()
    return probs


def sample_opponent_policies(policy_pool, num_opponents=3, recent_window=800, recent_mass=0.5):
    """
    Sample a set of opponent policies from the stored pool.
    """
    num_policies = len(policy_pool)
    if num_policies == 0:
        raise ValueError("policy_pool cannot be empty")

    probs = get_sampling_distribution(
        num_policies=num_policies,
        recent_window=recent_window,
        recent_mass=recent_mass,
    )

    sampled = np.random.choice(policy_pool, size=num_opponents, replace=True, p=probs)
    return list(sampled)


def refresh_opponents(policy_pool, distributed_rollout_manager, num_opponents=3, recent_window=800, recent_mass=0.5):
    """
    Update opponent policies on every rollout worker.

    Policy slot 0 is the main learning policy.
    Policy slots 1..N are sampled opponents.
    """
    for process_id in range(len(distributed_rollout_manager.processes)):
        sampled = sample_opponent_policies(
            policy_pool=policy_pool,
            num_opponents=num_opponents,
            recent_window=recent_window,
            recent_mass=recent_mass,
        )

        for i, state_dict in enumerate(sampled, start=1):
            distributed_rollout_manager.update_policy(
                state_dict=state_dict,
                process_id=process_id,
                policy_id=i,
            )