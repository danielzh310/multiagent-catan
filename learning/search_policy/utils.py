"""
Utility functions for forward search.

Includes:
- scoring helpers
- rollout heuristics
"""

import numpy as np


def normalize_scores(scores):
    """
    Normalize list of scores to [0,1]
    """
    scores = np.array(scores)
    if scores.max() == scores.min():
        return np.zeros_like(scores)

    return (scores - scores.min()) / (scores.max() - scores.min())


def softmax(x):
    """
    Stable softmax
    """
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def discounted_return(rewards, gamma=0.99):
    """
    Compute discounted return from reward list.
    """
    total = 0.0
    for t, r in enumerate(rewards):
        total += (gamma ** t) * r
    return total