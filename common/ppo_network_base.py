"""Base actor-critic network for PPO inference (no training-specific code)."""


from typing import Optional, Tuple

import torch
import torch.nn as nn


class PPONetworkBase(nn.Module):
    """
    Base actor-critic network for inference and simulation.
    Contains only the core network structure and forward pass, without training-specific features.
    """

    def __init__(
        self,
        base_obs_dim: int = 275,
        action_dim: int = 2,
        hidden_dim: int = 256,
        cognitive_param_dim: int = 0,
        cognitive_mask_dim: int = 4,
        cognitive_modulation: str = "none",
        checkpoint_obs_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if cognitive_modulation is None:
            cognitive_modulation = "none"
        elif isinstance(cognitive_modulation, str):
            cognitive_modulation = cognitive_modulation.lower()

        valid_modulations = {"none", "concat"}
        if cognitive_modulation not in valid_modulations:
            raise ValueError(f"Unsupported cognitive modulation: {cognitive_modulation}")

        self.base_obs_dim = base_obs_dim
        self.cognitive_param_dim = max(cognitive_param_dim, 0)
        self.cognitive_mask_dim = max(cognitive_mask_dim, 0)
        self.original_modulation = cognitive_modulation if self.cognitive_param_dim else "none"

        if self.cognitive_param_dim == 0:
            resolved_modulation = "none"
        elif self.original_modulation == "none":
            resolved_modulation = "concat"
        else:
            resolved_modulation = self.original_modulation

        self.cognitive_modulation = resolved_modulation
        self.action_dim = action_dim
        self.checkpoint_obs_dim = checkpoint_obs_dim

        if self.cognitive_modulation == "concat" and self.cognitive_param_dim:
            self.raw_obs_dim = (
                self.base_obs_dim
                + self.cognitive_param_dim
                + self.cognitive_mask_dim
            )
        else:
            self.raw_obs_dim = self.base_obs_dim
        self.obs_dim = self._compute_network_input_dim()
        self.extra_feature_dim = max(self.obs_dim - self.base_obs_dim, 0)

        # Actor / critic networks
        use_norm_layers = self.original_modulation != "none"

        self.obs_norm = (
            nn.LayerNorm(self.base_obs_dim) if self.base_obs_dim > 0 and use_norm_layers else None
        )
        self.cognitive_norm = None
        if self.cognitive_param_dim and use_norm_layers:
            self.cognitive_norm = nn.LayerNorm(self.cognitive_param_dim)

        self.actor_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2)

        self.critic_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

        self.tanh = nn.Tanh()

        self._init_weights()

    # ---------------------------------------------------------------------
    # Build inputs
    # ---------------------------------------------------------------------
    def _compute_network_input_dim(self) -> int:
        if self.cognitive_modulation == "concat" and self.cognitive_param_dim:
            return (
                self.base_obs_dim
                + self.cognitive_param_dim
                + self.cognitive_mask_dim
            )
        return self.base_obs_dim

    def _split_observation(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        base = obs[..., :self.base_obs_dim]
        offset = self.base_obs_dim

        params = None
        if self.cognitive_param_dim and self.cognitive_modulation == "concat":
            params = obs[..., offset:offset + self.cognitive_param_dim]
            offset += self.cognitive_param_dim

        mask = None
        if self.cognitive_mask_dim and self.cognitive_modulation == "concat":
            mask = obs[..., offset:offset + self.cognitive_mask_dim]

        return base, params, mask

    def _compose_input(self, obs: torch.Tensor) -> torch.Tensor:
        base, params, mask = self._split_observation(obs)

        if self.obs_norm is not None:
            base = self.obs_norm(base)

        if self.cognitive_param_dim == 0:
            return base

        if params is None:
            return base

        if self.cognitive_norm is not None:
            params = self.cognitive_norm(params)

        if mask is not None:
            params = params * mask[..., :self.cognitive_param_dim]

        features = [base, params]
        if mask is not None:
            features.append(mask)
        combined = torch.cat(features, dim=-1)
        return combined

    # ---------------------------------------------------------------------
    # Forward and policy interfaces
    # ---------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Initialize network weights"""
        for layer in [self.actor_fc1, self.actor_fc2, self.critic_fc1, self.critic_fc2]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)
        nn.init.constant_(self.critic_out.bias, 0.0)

        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.constant_(self.actor_out.bias, 0.0)

        with torch.no_grad():
            action_dim = self.actor_out.out_features // 2
            self.actor_out.bias[action_dim:].fill_(-0.5)
            self.actor_out.bias[1] = 0.1

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            obs: observation tensor [batch_size, obs_dim]
        Returns:
            action_logits: action logits [batch_size, action_dim * 2] (mean + log_std)
            value: value estimate [batch_size, 1]
        """
        network_input = self._compose_input(obs)

        x_actor = self.tanh(self.actor_fc1(network_input))
        x_actor = self.tanh(self.actor_fc2(x_actor))
        action_logits = self.actor_out(x_actor)

        x_critic = self.tanh(self.critic_fc1(network_input))
        x_critic = self.tanh(self.critic_fc2(x_critic))
        value = self.critic_out(x_critic)

        return action_logits, value

    def act_deterministic(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action selection for inference and simulation

        Args:
            obs_tensor: observation tensor [batch_size, obs_dim]
        Returns:
            action: deterministic action [batch_size, action_dim]
        """
        action_logits, _ = self.forward(obs_tensor)
        action_mean, _ = torch.chunk(action_logits, 2, dim=-1)
        return torch.tanh(action_mean)

    def get_action_stats(self, obs_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action statistics for steering and acceleration

        Returns:
            steering: steering action [batch_size]
            acceleration: acceleration action [batch_size]
        """
        action_logits, _ = self.forward(obs_tensor)
        action_mean, _ = torch.chunk(action_logits, 2, dim=-1)
        action_tanh = torch.tanh(action_mean)
        return action_tanh[:, 0], action_tanh[:, 1]

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified get_action_and_value for inference without training-time safety checks
        Note: this is a simplified version; use the full PPONetwork implementation during training.

        Args:
            obs: observation tensor
            action: provided action used to compute log_prob; sample a new action when None
            eps: numerical stability parameter

        Returns:
            action: action tensor [batch_size, action_dim]
            log_prob: action log-probability [batch_size]
            entropy: policy entropy [batch_size]
            value: value estimate [batch_size]
        """
        action_logits, value = self.forward(obs)

        action_mean, action_log_std = torch.chunk(action_logits, 2, dim=-1)
        action_log_std = torch.clamp(action_log_std, -5.0, 2.0)
        action_std = torch.exp(action_log_std)

        base_dist = torch.distributions.Normal(action_mean, action_std)

        if action is None:
            u = base_dist.rsample()
        else:
            a = action.clamp(-1 + eps, 1 - eps)
            u = 0.5 * (torch.log1p(a) - torch.log1p(-a))

        a = torch.tanh(u)
        log_prob = base_dist.log_prob(u) - torch.log(1 - a.pow(2) + eps)
        log_prob = log_prob.sum(dim=-1)
        entropy = base_dist.entropy().sum(dim=-1)

        return a, log_prob, entropy, value.squeeze(-1)
