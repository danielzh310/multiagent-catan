"""
Distributed rollout collection.

This file runs multiple RolloutManager workers in subprocesses and
collects trajectory batches from each one.
"""

import multiprocessing as mp
import numpy as np
import random
import torch

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper


def _worker(remote, parent_remote, manager_fn_wrapper):
    parent_remote.close()

    torch.set_num_threads(1)
    rollout_manager = manager_fn_wrapper.var()

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == "gather_rollouts":
                rollout_data = rollout_manager.gather_rollouts()
                remote.send(CloudpickleWrapper(rollout_data))

            elif cmd == "reset":
                rollout_manager.reset()
                remote.send(True)

            elif cmd == "close":
                remote.close()
                break

            elif cmd == "update_policy":
                state_dict = data[0].var
                policy_id = data[1]
                rollout_manager.update_policy(state_dict, policy_id=policy_id)
                remote.send(True)

            elif cmd == "seed":
                np.random.seed(data)
                random.seed(data)
                torch.manual_seed(data)
                remote.send(True)

            else:
                raise NotImplementedError(f"Unknown worker command: {cmd}")

        except EOFError:
            break


class DistributedRolloutManager:
    """
    Subprocess wrapper around many rollout managers.
    """

    def __init__(self, rollout_manager_fns, start_method=None):
        self.waiting = False
        self.closed = False

        n_processes = len(rollout_manager_fns)

        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

        ctx = mp.get_context(start_method)
        process_factory = getattr(ctx, "Process")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_processes)])
        self.processes = []

        for work_remote, remote, manager_fn in zip(self.work_remotes, self.remotes, rollout_manager_fns):
            args = (work_remote, remote, CloudpickleWrapper(manager_fn))
            process = process_factory(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

    def gather_async(self):
        for remote in self.remotes:
            remote.send(("gather_rollouts", None))
        self.waiting = True

    def gather_wait(self):
        results = [remote.recv().var for remote in self.remotes]
        self.waiting = False
        return results

    def gather_rollouts(self):
        self.gather_async()
        return self.gather_wait()

    def update_policy(self, state_dict, process_id=None, policy_id=0):
        """
        Update policy weights either on all workers or one worker.
        """
        if process_id is None:
            for remote in self.remotes:
                remote.send(("update_policy", (CloudpickleWrapper(state_dict), policy_id)))
            return [remote.recv() for remote in self.remotes]

        self.remotes[process_id].send(("update_policy", (CloudpickleWrapper(state_dict), policy_id)))
        return self.remotes[process_id].recv()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return [remote.recv() for remote in self.remotes]

    def seed(self, seeds):
        for i, remote in enumerate(self.remotes):
            remote.send(("seed", seeds[i]))
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
