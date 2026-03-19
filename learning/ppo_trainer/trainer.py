"""
PPO trainer.

This file handles:
- PPO loss computation
- policy update steps
- value loss
- entropy regularization
- gradient clipping
"""

import torch
import torch.nn as nn


class PPOTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        self.clip_param = config.clip_param
        self.ppo_epochs = config.ppo_epochs
        self.num_minibatches = config.num_minibatches

        self.value_loss_coef = config.value_loss_coef
        self.entropy_coef = config.entropy_coef_start
        self.max_grad_norm = config.max_grad_norm

        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.recompute_returns = config.recompute_returns

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps,
        )

    def update(self, batch_processor):
        """
        Run one PPO update over the rollout batch.
        """
        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        entropy_loss_epoch = 0.0

        for _ in range(self.ppo_epochs):
            with torch.no_grad():
                batch_processor.compute_advantages(self.model)

            if self.model.use_lstm:
                total_batch_size = batch_processor.num_parallel * batch_processor.num_steps
                data_generator = batch_processor.generate_lstm_minibatches(
                    num_minibatches=self.num_minibatches,
                    total_batch_size=total_batch_size,
                    truncated_seq_len=self.config.truncated_seq_len,
                )
            else:
                data_generator = batch_processor.generate_minibatches(
                    num_minibatches=self.num_minibatches,
                )

            for sample in data_generator:
                (
                    obs_batch,
                    hidden_batch,
                    actions_batch,
                    action_masks_batch,
                    value_preds_batch,
                    returns_batch,
                    done_masks_batch,
                    old_action_log_probs_batch,
                    advantage_batch,
                ) = sample

                if self.model.normalize_values:
                    value_preds_batch = self.model.value_normalizer.normalize(value_preds_batch)
                    returns_batch = self.model.value_normalizer.normalize(returns_batch)

                values, action_log_probs, entropy, _ = self.model.evaluate_actions(
                    obs=obs_batch,
                    actions=actions_batch,
                    action_masks=action_masks_batch,
                    hidden_state=hidden_batch,
                    done_mask=done_masks_batch,
                )

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_param,
                    1.0 + self.clip_param,
                ) * advantage_batch

                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + (
                    values - value_preds_batch
                ).clamp(-self.clip_param, self.clip_param)

                value_losses = (values - returns_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                self.optimizer.zero_grad()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - entropy * self.entropy_coef
                )

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item() * self.value_loss_coef
                action_loss_epoch += action_loss.item()
                entropy_loss_epoch += entropy.item() * self.entropy_coef

        num_updates = self.ppo_epochs * self.num_minibatches

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        entropy_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, entropy_loss_epoch

    def set_entropy_coef(self, value):
        """Update entropy regularization weight."""
        self.entropy_coef = float(value)