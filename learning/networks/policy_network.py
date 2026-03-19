"""
Main policy/value network for Catan.

This module combines:
- structured state encoding
- optional recurrent memory
- multi-head action outputs
- value prediction

It is the main actor-critic style model used by training and evaluation.
"""

import torch
import torch.nn as nn

from learning.networks.network_utils import ValueNormalizer, init_linear


class CatanPolicyNetwork(nn.Module):
    def __init__(
        self,
        state_encoder,
        action_heads,
        latent_dim=512,
        use_lstm=False,
        lstm_dim=256,
        value_hidden_dims=(256, 128),
        normalize_values=True,
    ):
        super().__init__()

        self.state_encoder = state_encoder
        self.action_heads = action_heads

        self.use_lstm = use_lstm
        self.lstm_dim = lstm_dim
        self.latent_dim = latent_dim

        self.normalize_values = normalize_values
        if self.normalize_values:
            self.value_normalizer = ValueNormalizer(mean=150.0, std=150.0)

        policy_input_dim = latent_dim

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=latent_dim,
                hidden_size=lstm_dim,
                num_layers=1,
                batch_first=False,
            )

            for name, param in self.lstm.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0.0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

            policy_input_dim += lstm_dim

        self.value_fc1 = init_linear(nn.Linear(policy_input_dim, value_hidden_dims[0]), gain=1.414)
        self.value_fc2 = init_linear(nn.Linear(value_hidden_dims[0], value_hidden_dims[1]), gain=1.414)
        self.value_out = init_linear(nn.Linear(value_hidden_dims[1], 1), gain=1.0)

        self.value_norm1 = nn.LayerNorm(value_hidden_dims[0])
        self.value_norm2 = nn.LayerNorm(value_hidden_dims[1])

        self.relu = nn.ReLU()

    def encode(self, obs, hidden_state=None, done_mask=None):
        """
        Encode observation and optionally pass through LSTM.
        """
        latent = self.state_encoder(obs)

        if self.use_lstm:
            if hidden_state is None:
                batch_size = latent.shape[0]
                h0 = torch.zeros(1, batch_size, self.lstm_dim, device=latent.device)
                c0 = torch.zeros(1, batch_size, self.lstm_dim, device=latent.device)
                hidden_state = (h0, c0)

            if done_mask is not None:
                # done_mask expected shape: (B, 1)
                h, c = hidden_state
                h = h * done_mask.view(1, -1, 1)
                c = c * done_mask.view(1, -1, 1)
                hidden_state = (h, c)

            lstm_out, next_hidden = self.lstm(latent.unsqueeze(0), hidden_state)
            lstm_out = lstm_out.squeeze(0)
            features = torch.cat([latent, lstm_out], dim=-1)
        else:
            next_hidden = hidden_state
            features = latent

        return features, next_hidden

    def value(self, features):
        """
        Predict scalar state value.
        """
        x = self.value_fc1(features)
        x = self.relu(self.value_norm1(x))

        x = self.value_fc2(x)
        x = self.relu(self.value_norm2(x))

        return self.value_out(x)

    def forward(self, obs, action_masks=None, hidden_state=None, done_mask=None):
        """
        Forward pass returning:
        - value estimate
        - action distributions
        - next recurrent hidden state
        """
        features, next_hidden = self.encode(
            obs,
            hidden_state=hidden_state,
            done_mask=done_mask,
        )

        values = self.value(features)
        action_outputs = self.action_heads(features, action_masks=action_masks)

        return values, action_outputs, next_hidden

    def act(
        self,
        obs,
        action_masks=None,
        hidden_state=None,
        done_mask=None,
        deterministic=False,
    ):
        """
        Sample or choose deterministic action.
        """
        values, action_outputs, next_hidden = self.forward(
            obs,
            action_masks=action_masks,
            hidden_state=hidden_state,
            done_mask=done_mask,
        )

        if deterministic:
            actions = self.action_heads.mode(action_outputs)
        else:
            actions = self.action_heads.sample(action_outputs)

        action_log_probs = self.action_heads.log_probs(action_outputs, actions)
        entropy = self.action_heads.entropy(action_outputs)

        return values, actions, action_log_probs, next_hidden, entropy

    def evaluate_actions(
        self,
        obs,
        actions,
        action_masks=None,
        hidden_state=None,
        done_mask=None,
    ):
        """
        Evaluate provided actions for PPO-style training.
        """
        values, action_outputs, next_hidden = self.forward(
            obs,
            action_masks=action_masks,
            hidden_state=hidden_state,
            done_mask=done_mask,
        )

        action_log_probs = self.action_heads.log_probs(action_outputs, actions)
        entropy = self.action_heads.entropy(action_outputs)

        return values, action_log_probs, entropy, next_hidden

    def get_value(self, obs, hidden_state=None, done_mask=None):
        """
        Convenience method for value-only calls.
        """
        features, _ = self.encode(
            obs,
            hidden_state=hidden_state,
            done_mask=done_mask,
        )
        return self.value(features)