"""
Batch processing for PPO rollouts.

This file:
- stores rollout data
- converts trajectories into training tensors
- computes values / returns / advantages
- yields minibatches for PPO updates
"""

import itertools
import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from learning.ppo_trainer.schedule_utils import flatten_time_batch, reshape_flatten_time_batch


OBS_KEYS = [
    "tile_features",
    "current_player_main",
    "current_player_hidden_dev",
    "current_player_played_dev",
    "next_player_main",
    "next_player_played_dev",
    "next_next_player_main",
    "next_next_player_played_dev",
    "next_next_next_player_main",
    "next_next_next_player_played_dev",
]


class BatchProcessor:
    def __init__(self, config, lstm_dim, device="cpu", storage_device="cpu"):
        self.config = config
        self.num_steps = config.num_steps
        self.num_parallel = config.num_processes * config.num_envs_per_process
        self.lstm_dim = lstm_dim

        self.device = device
        self.storage_device = storage_device

        self.games_completed = 0

    def process_rollouts(self, rollouts):
        """
        Expected rollout format:
        (
            observations,
            hidden_states,
            rewards,
            actions,
            action_masks,
            action_log_probs,
            done_masks,
        )
        """

        rollouts = list(zip(*rollouts))

        observations = [inner for outer in rollouts[0] for inner in outer]
        self.obs = {}

        for key in OBS_KEYS:
            stacked = []
            for t in range(self.num_steps + 1):
                stacked.append(
                    torch.cat(
                        [observations[k][t][key] for k in range(self.num_parallel)],
                        dim=0,
                    )
                )
            self.obs[key] = torch.stack(stacked).to(self.storage_device)

        hidden_states = [inner for outer in rollouts[1] for inner in outer]
        self.hidden_states = (
            torch.stack(
                [
                    torch.cat([hidden_states[k][t][0] for k in range(self.num_parallel)], dim=0)
                    for t in range(self.num_steps + 1)
                ]
            ).to(self.storage_device),
            torch.stack(
                [
                    torch.cat([hidden_states[k][t][1] for k in range(self.num_parallel)], dim=0)
                    for t in range(self.num_steps + 1)
                ]
            ).to(self.storage_device),
        )

        rewards = [inner for outer in rollouts[2] for inner in outer]
        self.rewards = torch.stack(
            [
                torch.cat(
                    [
                        torch.tensor(rewards[k][t], dtype=torch.float32, device=self.storage_device).view(1, 1)
                        for k in range(self.num_parallel)
                    ],
                    dim=0,
                )
                for t in range(self.num_steps)
            ]
        ).to(self.storage_device)

        actions = [inner for outer in rollouts[3] for inner in outer]
        self.actions = {}
        action_keys = list(actions[0][0].keys())

        for key in action_keys:
            stacked = []
            for t in range(self.num_steps):
                stacked.append(
                    torch.cat(
                        [
                            torch.tensor(actions[k][t][key], dtype=torch.long, device=self.storage_device).view(1, -1)
                            for k in range(self.num_parallel)
                        ],
                        dim=0,
                    )
                )
            self.actions[key] = torch.stack(stacked).to(self.storage_device)

        action_masks = [inner for outer in rollouts[4] for inner in outer]
        self.action_masks = {}
        mask_keys = list(action_masks[0][0].keys())

        for key in mask_keys:
            stacked = []
            for t in range(self.num_steps):
                stacked.append(
                    torch.cat(
                        [action_masks[k][t][key] for k in range(self.num_parallel)],
                        dim=0,
                    )
                )
            self.action_masks[key] = torch.stack(stacked).to(self.storage_device)

        action_log_probs = [inner for outer in rollouts[5] for inner in outer]
        self.action_log_probs = torch.stack(
            [
                torch.cat([action_log_probs[k][t] for k in range(self.num_parallel)], dim=0)
                for t in range(self.num_steps)
            ]
        ).to(self.storage_device)

        done_masks = [inner for outer in rollouts[6] for inner in outer]
        self.done_masks = torch.stack(
            [
                torch.cat(
                    [
                        torch.tensor(done_masks[k][t], dtype=torch.float32).view(1, 1)
                        for k in range(self.num_parallel)
                    ],
                    dim=0,
                )
                for t in range(self.num_steps + 1)
            ]
        ).to(self.storage_device)

        self.games_completed += int(torch.sum(1.0 - self.done_masks).item())

    def compute_advantages(self, model, max_processes_at_once=8):
        self.values = torch.zeros(self.num_steps + 1, self.num_parallel, 1).to(self.storage_device)
        self.returns = torch.zeros_like(self.rewards).to(self.storage_device)

        start_indices = np.arange(0, self.num_parallel, max_processes_at_once)
        end_indices = np.minimum(start_indices + max_processes_at_once, self.num_parallel)

        for i in range(len(start_indices)):
            start = start_indices[i]
            end = end_indices[i]
            num_proc = end - start

            if model.use_lstm:
                hidden_in = (
                    reshape_flatten_time_batch(
                        self.num_steps + 1,
                        num_proc,
                        self.hidden_states[0][:, start:end, ...],
                    ).to(self.device),
                    reshape_flatten_time_batch(
                        self.num_steps + 1,
                        num_proc,
                        self.hidden_states[1][:, start:end, ...],
                    ).to(self.device),
                )
            else:
                hidden_in = None

            obs_in = {}
            for key in OBS_KEYS:
                obs_in[key] = reshape_flatten_time_batch(
                    self.num_steps + 1,
                    num_proc,
                    self.obs[key][:, start:end, ...],
                ).to(self.device)

            done_mask_in = reshape_flatten_time_batch(
                self.num_steps + 1,
                num_proc,
                self.done_masks[:, start:end, ...],
            ).to(self.device)

            values = model.get_value(
                obs=obs_in,
                hidden_state=hidden_in,
                done_mask=done_mask_in,
            )

            self.values[:, start:end, :] = values.reshape(
                self.num_steps + 1,
                num_proc,
                -1,
            ).to(self.storage_device)

        if model.normalize_values:
            self.values = model.value_normalizer.denormalize(self.values)

        gae = 0.0
        for step in reversed(range(self.num_steps)):
            delta = (
                self.rewards[step]
                + self.config.gamma * self.values[step + 1] * self.done_masks[step + 1]
                - self.values[step]
            )
            gae = delta + self.config.gamma * self.config.gae_lambda * self.done_masks[step + 1] * gae
            self.returns[step] = gae + self.values[step]

        advantages = self.returns - self.values[:-1]
        self.advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    def generate_minibatches(self, num_minibatches):
        batch_size = self.num_steps * self.num_parallel
        minibatch_size = batch_size // num_minibatches

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            minibatch_size,
            drop_last=True,
        )

        for indices in sampler:
            obs_batch = {}
            for key in OBS_KEYS:
                obs_batch[key] = self.obs[key][:-1].view(-1, *self.obs[key].size()[2:])[indices].to(self.device)

            hidden_batch = None

            actions_batch = {}
            for key in self.actions:
                actions_batch[key] = self.actions[key].view(-1, *self.actions[key].size()[2:])[indices].to(self.device)

            action_masks_batch = {}
            for key in self.action_masks:
                action_masks_batch[key] = self.action_masks[key].view(
                    -1,
                    *self.action_masks[key].size()[2:]
                )[indices].to(self.device)

            value_preds_batch = self.values[:-1].view(-1, 1)[indices].to(self.device)
            returns_batch = self.returns.view(-1, 1)[indices].to(self.device)
            done_masks_batch = self.done_masks[:-1].view(-1, 1)[indices].to(self.device)
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices].to(self.device)
            advantage_batch = self.advantages.view(-1, 1)[indices].to(self.device)

            yield (
                obs_batch,
                hidden_batch,
                actions_batch,
                action_masks_batch,
                value_preds_batch,
                returns_batch,
                done_masks_batch,
                old_action_log_probs_batch,
                advantage_batch,
            )

    def generate_lstm_minibatches(self, num_minibatches, total_batch_size, truncated_seq_len):
        T = self.num_steps
        num_parallel = self.num_parallel

        if T % truncated_seq_len != 0:
            raise ValueError("num_steps must be divisible by truncated_seq_len")

        sequences_per_minibatch = total_batch_size // num_minibatches // truncated_seq_len
        N = sequences_per_minibatch

        time_indices = []
        process_indices = []

        for process_id in range(num_parallel):
            for t_start in range(0, T, truncated_seq_len):
                process_indices.append([process_id] * truncated_seq_len)
                time_indices.append(np.arange(t_start, t_start + truncated_seq_len))

        permutation = np.random.permutation(len(time_indices))

        for start_idx in range(0, len(permutation), sequences_per_minibatch):
            obs_batch = {key: [] for key in OBS_KEYS}
            hidden_batch = [[], []]
            actions_batch = {key: [] for key in self.actions}
            action_masks_batch = {key: [] for key in self.action_masks}

            value_preds_batch = []
            returns_batch = []
            done_masks_batch = []
            old_action_log_probs_batch = []
            advantage_batch = []

            for offset in range(sequences_per_minibatch):
                t_inds = time_indices[permutation[start_idx + offset]]
                p_inds = process_indices[permutation[start_idx + offset]]

                for key in OBS_KEYS:
                    obs_batch[key].append(self.obs[key][t_inds, p_inds, ...])

                hidden_batch[0].append(self.hidden_states[0][t_inds[0]:t_inds[0] + 1, p_inds[0], ...])
                hidden_batch[1].append(self.hidden_states[1][t_inds[0]:t_inds[0] + 1, p_inds[0], ...])

                for key in self.actions:
                    actions_batch[key].append(self.actions[key][t_inds, p_inds, ...])

                for key in self.action_masks:
                    action_masks_batch[key].append(self.action_masks[key][t_inds, p_inds, ...])

                value_preds_batch.append(self.values[t_inds, p_inds])
                returns_batch.append(self.returns[t_inds, p_inds])
                done_masks_batch.append(self.done_masks[t_inds, p_inds])
                old_action_log_probs_batch.append(self.action_log_probs[t_inds, p_inds])
                advantage_batch.append(self.advantages[t_inds, p_inds])

            for key in OBS_KEYS:
                obs_batch[key] = torch.stack(obs_batch[key], 1)

            hidden_batch[0] = torch.stack(hidden_batch[0], 1).view(N, -1).to(self.device)
            hidden_batch[1] = torch.stack(hidden_batch[1], 1).view(N, -1).to(self.device)

            for key in self.actions:
                actions_batch[key] = torch.stack(actions_batch[key], 1)

            for key in self.action_masks:
                action_masks_batch[key] = torch.stack(action_masks_batch[key], 1)

            value_preds_batch = torch.stack(value_preds_batch, 1)
            returns_batch = torch.stack(returns_batch, 1)
            done_masks_batch = torch.stack(done_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            advantage_batch = torch.stack(advantage_batch, 1)

            for key in OBS_KEYS:
                obs_batch[key] = flatten_time_batch(truncated_seq_len, N, obs_batch[key]).to(self.device)

            for key in self.actions:
                actions_batch[key] = flatten_time_batch(truncated_seq_len, N, actions_batch[key]).to(self.device)

            for key in self.action_masks:
                action_masks_batch[key] = flatten_time_batch(
                    truncated_seq_len,
                    N,
                    action_masks_batch[key],
                ).to(self.device)

            value_preds_batch = flatten_time_batch(truncated_seq_len, N, value_preds_batch).to(self.device)
            returns_batch = flatten_time_batch(truncated_seq_len, N, returns_batch).to(self.device)
            done_masks_batch = flatten_time_batch(truncated_seq_len, N, done_masks_batch).to(self.device)
            old_action_log_probs_batch = flatten_time_batch(truncated_seq_len, N, old_action_log_probs_batch).to(self.device)
            advantage_batch = flatten_time_batch(truncated_seq_len, N, advantage_batch).to(self.device)

            yield (
                obs_batch,
                (hidden_batch[0], hidden_batch[1]),
                actions_batch,
                action_masks_batch,
                value_preds_batch,
                returns_batch,
                done_masks_batch,
                old_action_log_probs_batch,
                advantage_batch,
            )