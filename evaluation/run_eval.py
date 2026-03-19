"""
Run evaluation for the learned policy.

This script:
- builds the model
- optionally loads a checkpoint
- runs evaluation games
- prints summary metrics

Usage examples:
    python evaluation/run_eval.py
    python evaluation/run_eval.py --checkpoint checkpoints/model_100.pt --games 64
"""

import argparse
import copy
import os
import sys
import torch
import numpy as np

# allow running from project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from learning.networks.build_model import build_model
from learning.ppo_trainer.evaluator import Evaluator
from learning.ppo_trainer.distributed_eval import DistributedEvalManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint")
    parser.add_argument("--games", type=int, default=32, help="Total number of evaluation games")
    parser.add_argument("--processes", type=int, default=4, help="Number of evaluation worker processes")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy actions instead of sampling")
    return parser.parse_args()


def make_evaluator(device="cpu", seed=0):
    def _fn():
        return Evaluator(
            model_builder=build_model,
            seed=seed,
            device=device,
        )
    return _fn


def load_checkpoint_into_model(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        # standard plain state_dict
        try:
            model.load_state_dict(checkpoint)
            return
        except Exception:
            pass

        # wrapped checkpoint formats
        for key in ["model_state_dict", "state_dict", "policy_state_dict", "actor_critic"]:
            if key in checkpoint:
                model.load_state_dict(checkpoint[key])
                return

    if isinstance(checkpoint, (list, tuple)) and len(checkpoint) > 0:
        # support old tuple-style save formats
        try:
            model.load_state_dict(checkpoint[0])
            return
        except Exception:
            pass

    raise ValueError(f"Could not load checkpoint format from: {checkpoint_path}")


def main():
    args = parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    model = build_model()
    model.to(device)
    model.eval()

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        load_checkpoint_into_model(model, args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("No checkpoint provided. Using randomly initialized model.")

    evaluator_fns = [
        make_evaluator(device=str(device), seed=1000 * i)
        for i in range(args.processes)
    ]
    eval_manager = DistributedEvalManager(evaluator_fns)

    state_dicts = [
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(model.state_dict()),
        copy.deepcopy(model.state_dict()),
    ]
    eval_manager.update_policies(state_dicts)

    results = eval_manager.run_eval_games(args.games)
    eval_manager.close()

    results = list(zip(*results))

    winners = np.concatenate(results[0]) if len(results[0]) else np.array([])
    total_steps = np.concatenate(results[1]) if len(results[1]) else np.array([])
    victory_points = np.concatenate(results[2]) if len(results[2]) else np.array([])
    policy_decisions = np.concatenate(results[3]) if len(results[3]) else np.array([])

    if len(winners) == 0:
        print("No evaluation results returned.")
        return

    win_fraction = float(np.mean(winners == 0))
    avg_steps = float(np.mean(total_steps))
    avg_vp = float(np.mean(victory_points))
    avg_policy_decisions = float(np.mean(policy_decisions))

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Games: {len(winners)}")
    print(f"Controlled policy win fraction: {win_fraction:.4f}")
    print(f"Average total steps: {avg_steps:.2f}")
    print(f"Average victory points: {avg_vp:.2f}")
    print(f"Average policy decisions: {avg_policy_decisions:.2f}")
    print("Winner histogram:")
    unique, counts = np.unique(winners, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  policy slot {int(u)}: {int(c)} wins")
    print("========================================\n")


if __name__ == "__main__":
    main()