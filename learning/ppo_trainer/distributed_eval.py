"""
Distributed evaluation manager.

Runs many Evaluator workers in subprocesses and aggregates results.
"""

import multiprocessing as mp
import torch

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def _worker(remote, parent_remote, evaluator_fn_wrapper):
    parent_remote.close()

    torch.set_num_threads(1)
    evaluator = evaluator_fn_wrapper.var()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == "run_eval_games":
                num_games = data
                winners = []
                total_steps = []
                victory_points = []
                policy_decisions = []

                for _ in range(num_games):
                    winner, vp, steps, decisions = evaluator.run_evaluation_game()
                    winners.append(winner)
                    victory_points.append(vp)
                    total_steps.append(steps)
                    policy_decisions.append(decisions)

                remote.send((winners, total_steps, victory_points, policy_decisions))

            elif cmd == "update_policies":
                state_dicts = data.var
                evaluator.update_policies(state_dicts)
                remote.send(True)

            elif cmd == "close":
                remote.close()
                break

            else:
                raise NotImplementedError(f"Unknown evaluator command: {cmd}")

        except EOFError:
            break


class DistributedEvalManager:
    """
    Subprocess wrapper around evaluator instances.
    """

    def __init__(self, evaluator_fns, start_method=None):
        self.waiting = False
        self.closed = False

        n_processes = len(evaluator_fns)

        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_processes)])
        self.processes = []

        for work_remote, remote, evaluator_fn in zip(self.work_remotes, self.remotes, evaluator_fns):
            args = (work_remote, remote, CloudpickleWrapper(evaluator_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def run_eval_async(self, total_games):
        games_per_process = total_games // len(self.processes)

        for remote in self.remotes:
            remote.send(("run_eval_games", games_per_process))

        self.waiting = True

    def run_eval_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return results

    def run_eval_games(self, total_games):
        self.run_eval_async(total_games)
        return self.run_eval_wait()

    def update_policies(self, state_dicts):
        for remote in self.remotes:
            remote.send(("update_policies", CloudpickleWrapper(state_dicts)))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(("close", None))

        for process in self.processes:
            process.join()

        self.closed = True