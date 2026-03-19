"""
Evaluation protocol for training checkpoints.

This file runs the current policy against:
- random / baseline opponent policies
- earlier stored checkpoints

and returns both a machine-readable log and a printable summary.
"""

import copy
import numpy as np


def run_eval_protocol(
    distributed_eval_manager,
    current_policy,
    stored_policy_pool,
    random_policy_state,
    config,
    update_idx,
):
    """
    Evaluate the current policy against a small opponent set.

    Returns:
        log_dict, summary_string
    """
    opponent_states = [random_policy_state]
    opponent_labels = ["random"]

    if len(stored_policy_pool) >= 25:
        opponent_states.append(stored_policy_pool[-25])
        opponent_labels.append("25_updates_ago")

    if len(stored_policy_pool) >= 100:
        opponent_states.append(stored_policy_pool[-100])
        opponent_labels.append("100_updates_ago")

    log = {"update": update_idx}

    summary = "\n"
    summary += f"---------------------- EVALUATION (update {update_idx}) ----------------------\n"

    for label, opponent_state in zip(opponent_labels, opponent_states):
        state_dicts = [
            copy.deepcopy(current_policy.state_dict()),
            copy.deepcopy(opponent_state),
            copy.deepcopy(opponent_state),
            copy.deepcopy(opponent_state),
        ]

        distributed_eval_manager.update_policies(state_dicts)

        results = distributed_eval_manager.run_eval_games(config.num_eval_episodes)
        results = list(zip(*results))

        winners = np.concatenate(results[0])
        total_steps = np.concatenate(results[1])
        victory_points = np.concatenate(results[2])
        policy_decisions = np.concatenate(results[3])

        win_fraction = float(np.mean(winners == 0))
        avg_steps = float(np.mean(total_steps))
        avg_vp = float(np.mean(victory_points))
        avg_decisions = float(np.mean(policy_decisions))

        log[label] = {
            "win_fraction": win_fraction,
            "avg_steps": avg_steps,
            "avg_victory_points": avg_vp,
            "avg_policy_decisions": avg_decisions,
        }

        summary += (
            f"{config.num_eval_episodes} games vs {label}: "
            f"win_fraction={win_fraction:.3f}, "
            f"avg_steps={avg_steps:.2f}, "
            f"avg_vp={avg_vp:.2f}, "
            f"avg_policy_decisions={avg_decisions:.2f}\n"
        )

    summary += "\n"
    return log, summary