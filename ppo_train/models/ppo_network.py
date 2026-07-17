"""Actor-critic network definition used for PPO expert reproduction."""


import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPONetwork(nn.Module):
    """Actor-critic network with concat cognitive parameters."""

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

        valid_modulations = {"none", "concat"}
        if cognitive_modulation not in valid_modulations:
            raise ValueError(f"Unsupported cognitive modulation: {cognitive_modulation}")

        self.base_obs_dim = base_obs_dim
        self.cognitive_param_dim = max(cognitive_param_dim, 0)
        self.cognitive_mask_dim = max(cognitive_mask_dim, 0)
        self.cognitive_modulation = cognitive_modulation if self.cognitive_param_dim else "none"
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

        self.is_progressive_training = (
            self.cognitive_modulation == "concat"
            and self.cognitive_param_dim > 0
            and checkpoint_obs_dim is not None
            and checkpoint_obs_dim == self.base_obs_dim
            and self.obs_dim > self.base_obs_dim
        )

        # Actor / critic networks
        self.obs_norm = nn.LayerNorm(self.base_obs_dim) if self.base_obs_dim > 0 else None
        self.cognitive_norm = None
        if self.cognitive_param_dim:
            self.cognitive_norm = nn.LayerNorm(self.cognitive_param_dim)

        self.actor_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2)

        self.critic_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

        self.tanh = nn.Tanh()
        self.training_stage = 1  # 1: frozen stage, 2: unfrozen stage
        self.freeze_threshold_steps = 3_000_000

        self._init_weights()

        if self.is_progressive_training:
            self._setup_progressive_training()

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
        for layer in [self.actor_fc1, self.actor_fc2, self.critic_fc1, self.critic_fc2]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.critic_out.weight, gain=1.0)
        nn.init.constant_(self.critic_out.bias, 0.0)

        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)
        nn.init.constant_(self.actor_out.bias, 0.0)

        with torch.no_grad():
            action_dim = self.actor_out.out_features // 2
            self.actor_out.bias[action_dim:].fill_(-1.0)
            self.actor_out.bias[1] = 0.1

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        network_input = self._compose_input(obs)

        x_actor = self.tanh(self.actor_fc1(network_input))
        x_actor = self.tanh(self.actor_fc2(x_actor))
        action_logits = self.actor_out(x_actor)

        x_critic = self.tanh(self.critic_fc1(network_input))
        x_critic = self.tanh(self.critic_fc2(x_critic))
        value = self.critic_out(x_critic)

        return action_logits, value

    @staticmethod
    def _squash_log_abs_det_jacobian(pre_tanh_action: torch.Tensor) -> torch.Tensor:
        """Return log|d tanh(u) / du| without evaluating tanh near saturation."""
        return 2.0 * (
            math.log(2.0)
            - pre_tanh_action
            - F.softplus(-2.0 * pre_tanh_action)
        )

    @classmethod
    def _squashed_log_prob(
        cls,
        base_dist: torch.distributions.Normal,
        pre_tanh_action: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = base_dist.log_prob(pre_tanh_action)
        log_prob -= cls._squash_log_abs_det_jacobian(pre_tanh_action)
        return log_prob.sum(dim=-1)

    @classmethod
    def _squashed_entropy_estimate(
        cls,
        base_dist: torch.distributions.Normal,
        pre_tanh_sample: torch.Tensor,
    ) -> torch.Tensor:
        """One-sample reparameterized entropy estimate in bounded action space."""
        entropy = base_dist.entropy()
        entropy += cls._squash_log_abs_det_jacobian(pre_tanh_sample)
        return entropy.sum(dim=-1)

    def _action_distribution_and_value(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        action_logits, value = self.forward(obs)
        action_mean, action_log_std = torch.chunk(action_logits, 2, dim=-1)
        action_log_std = torch.clamp(action_log_std, -5.0, 2.0)
        action_std = torch.exp(action_log_std)
        return torch.distributions.Normal(action_mean, action_std), value.squeeze(-1)

    def sample_action_and_value(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a bounded action while retaining the exact latent draw for PPO."""
        base_dist, value = self._action_distribution_and_value(obs)
        pre_tanh_action = base_dist.rsample()
        action = torch.tanh(pre_tanh_action)
        log_prob = self._squashed_log_prob(base_dist, pre_tanh_action)
        entropy = self._squashed_entropy_estimate(base_dist, pre_tanh_action)
        return action, pre_tanh_action, log_prob, entropy, value

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        pre_tanh_action: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        estimate_entropy: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if action is not None and pre_tanh_action is not None:
            raise ValueError("action and pre_tanh_action are mutually exclusive")

        base_dist, value = self._action_distribution_and_value(obs)
        sampled_from_current = action is None and pre_tanh_action is None

        if sampled_from_current:
            pre_tanh_action = base_dist.rsample()
            action = torch.tanh(pre_tanh_action)
        elif pre_tanh_action is not None:
            action = torch.tanh(pre_tanh_action)
        else:
            # Compatibility-only path for bounded external actions. PPO rollout
            # replay uses pre_tanh_action and never enters this lossy inverse.
            bounded_action = action.clamp(-1 + eps, 1 - eps)
            pre_tanh_action = 0.5 * (
                torch.log1p(bounded_action) - torch.log1p(-bounded_action)
            )
            action = bounded_action

        log_prob = self._squashed_log_prob(base_dist, pre_tanh_action)
        if estimate_entropy:
            entropy_sample = (
                pre_tanh_action if sampled_from_current else base_dist.rsample()
            )
            entropy = self._squashed_entropy_estimate(base_dist, entropy_sample)
        else:
            entropy = torch.zeros_like(log_prob)

        return action, log_prob, entropy, value

    def act_deterministic(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        action_logits, _ = self.forward(obs_tensor)
        action_mean, _ = torch.chunk(action_logits, 2, dim=-1)
        return torch.tanh(action_mean)

    def get_action_stats(self, obs_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_logits, _ = self.forward(obs_tensor)
        action_mean, _ = torch.chunk(action_logits, 2, dim=-1)
        action_tanh = torch.tanh(action_mean)
        return action_tanh[:, 0], action_tanh[:, 1]

    # ---------------------------------------------------------------------
    # Progressive-training support
    # ---------------------------------------------------------------------
    def _setup_progressive_training(self) -> None:
        if not self.is_progressive_training:
            return

        print("Configuring progressive training mode:")
        print(f"   Checkpoint dimension: {self.checkpoint_obs_dim}")
        print(f"   Target dimension: {self.obs_dim}")
        print(f"   Freeze the first {self.base_obs_dim} dimensions of the weights; train only the new {self.obs_dim - self.base_obs_dim} dimensions and downstream layers")

        with torch.no_grad():
            self.actor_fc1.weight[:, : self.base_obs_dim].requires_grad_(False)
            self.critic_fc1.weight[:, : self.base_obs_dim].requires_grad_(False)

        self.training_stage = 1
        print(f"   Current training stage: {self.training_stage} (frozen stage)")

    def update_training_stage(self, global_step: int) -> None:
        if not self.is_progressive_training:
            return

        if self.training_stage == 1 and global_step >= self.freeze_threshold_steps:
            self.training_stage = 2
            self._unfreeze_weights()
            print(f"Training stage switched to: {self.training_stage} (unfrozen stage)")
            print("   Earlier-layer weights are now unfrozen so the model can learn interactions with cognitive features.")

    def _unfreeze_weights(self) -> None:
        if not self.is_progressive_training:
            return

        with torch.no_grad():
            self.actor_fc1.weight[:, : self.base_obs_dim].requires_grad_(True)
            self.critic_fc1.weight[:, : self.base_obs_dim].requires_grad_(True)

        print("   Unfroze the weights corresponding to the base observation features")

    # ---------------------------------------------------------------------
    # Weight migration utilities
    # ---------------------------------------------------------------------
    def _load_and_extend_weights(self, checkpoint_state_dict):
        current_dim = self.actor_fc1.in_features

        checkpoint_dim = None
        for key, value in checkpoint_state_dict.items():
            if key in ["actor_fc1.weight", "critic_fc1.weight"]:
                checkpoint_dim = value.shape[1]
                break

        print(f"Expanding weights: {checkpoint_dim} dims -> {current_dim} dims...")
        new_state_dict = {}

        for key, value in checkpoint_state_dict.items():
            if key in ["actor_fc1.weight", "critic_fc1.weight"]:
                new_weight = torch.zeros(value.shape[0], current_dim, device=value.device)
                new_weight[:, :checkpoint_dim] = value
                if current_dim > checkpoint_dim:
                    nn.init.orthogonal_(new_weight[:, checkpoint_dim:], gain=0.01)
                new_state_dict[key] = new_weight
                head = "actor" if "actor" in key else "critic"
                print(f"   Extended {head}_fc1 weights: {checkpoint_dim} -> {current_dim}")
            else:
                new_state_dict[key] = value

        print("Weight expansion complete")
        return new_state_dict

    def _load_and_truncate_weights(self, checkpoint_state_dict):
        current_dim = self.actor_fc1.in_features

        checkpoint_dim = None
        for key, value in checkpoint_state_dict.items():
            if key in ["actor_fc1.weight", "critic_fc1.weight"]:
                checkpoint_dim = value.shape[1]
                break

        print(f"Truncating weights: {checkpoint_dim} dims -> {current_dim} dims...")
        new_state_dict = {}

        for key, value in checkpoint_state_dict.items():
            if key in ["actor_fc1.weight", "critic_fc1.weight"]:
                new_state_dict[key] = value[:, :current_dim]
                head = "actor" if "actor" in key else "critic"
                print(f"   Truncated {head}_fc1 weights: {checkpoint_dim} -> {current_dim}")
            else:
                new_state_dict[key] = value

        print("Weight truncation complete")
        return new_state_dict

    # ---------------------------------------------------------------------
    # Monitoring utilities
    # ---------------------------------------------------------------------
    def get_frozen_weight_stats(self):
        if not self.is_progressive_training:
            return {}

        stats = {}
        with torch.no_grad():
            actor_frozen = self.actor_fc1.weight[:, : self.base_obs_dim]
            stats["actor_frozen_mean"] = actor_frozen.mean().item()
            stats["actor_frozen_std"] = actor_frozen.std().item()
            stats["actor_frozen_norm"] = torch.norm(actor_frozen).item()

            critic_frozen = self.critic_fc1.weight[:, : self.base_obs_dim]
            stats["critic_frozen_mean"] = critic_frozen.mean().item()
            stats["critic_frozen_std"] = critic_frozen.std().item()
            stats["critic_frozen_norm"] = torch.norm(critic_frozen).item()

            actor_new = self.actor_fc1.weight[:, self.base_obs_dim :]
            stats["actor_new_mean"] = actor_new.mean().item()
            stats["actor_new_std"] = actor_new.std().item()
            stats["actor_new_norm"] = torch.norm(actor_new).item()

            critic_new = self.critic_fc1.weight[:, self.base_obs_dim :]
            stats["critic_new_mean"] = critic_new.mean().item()
            stats["critic_new_std"] = critic_new.std().item()
            stats["critic_new_norm"] = torch.norm(critic_new).item()

        return stats
