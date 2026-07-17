"""PPO update mixin for PPO expert training."""


import copy

import numpy as np
import torch
import torch.nn as nn


_KL_BACKTRACK_FACTOR = 0.5
_MAX_KL_BACKTRACKS = 16


class PPOUpdateMixin:
    """Mixin holding methods split from PPOExpertReproduction."""

    @staticmethod
    def _mean_or_zero(values) -> float:
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _approx_kl(log_ratio: torch.Tensor) -> torch.Tensor:
        ratio = torch.exp(log_ratio)
        return ((ratio - 1.0) - log_ratio).mean()

    def _measure_policy_kl(
        self,
        obs: torch.Tensor,
        pre_tanh_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            _, new_log_probs, _, _ = self.network.get_action_and_value(
                obs,
                pre_tanh_action=pre_tanh_actions,
                estimate_entropy=False,
            )
            return self._approx_kl(new_log_probs - old_log_probs)

    @staticmethod
    def _restore_parameters(parameters, snapshots) -> None:
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)

    def _transactional_optimizer_step(
        self,
        batch_obs: torch.Tensor,
        batch_pre_tanh_actions: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
    ):
        """Apply one Adam step only after its committed KL is within contract."""
        target_kl = self.args.target_kl
        if not target_kl:
            self.optimizer.step()
            post_kl = self._measure_policy_kl(
                batch_obs,
                batch_pre_tanh_actions,
                batch_old_log_probs,
            )
            return post_kl, post_kl, 1.0, False, 0

        parameters = tuple(self.network.parameters())
        parameter_before = [parameter.detach().clone() for parameter in parameters]
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())

        self.optimizer.step()
        candidate_parameters = [
            parameter.detach().clone() for parameter in parameters
        ]
        candidate_kl = self._measure_policy_kl(
            batch_obs,
            batch_pre_tanh_actions,
            batch_old_log_probs,
        )

        if torch.isfinite(candidate_kl) and candidate_kl <= target_kl:
            return candidate_kl, candidate_kl, 1.0, False, 0

        parameter_deltas = [
            candidate - before
            for candidate, before in zip(candidate_parameters, parameter_before)
        ]
        for backtrack_count in range(1, _MAX_KL_BACKTRACKS + 1):
            step_scale = _KL_BACKTRACK_FACTOR ** backtrack_count
            with torch.no_grad():
                for parameter, before, delta in zip(
                    parameters,
                    parameter_before,
                    parameter_deltas,
                ):
                    parameter.copy_(before + step_scale * delta)

            committed_kl = self._measure_policy_kl(
                batch_obs,
                batch_pre_tanh_actions,
                batch_old_log_probs,
            )
            if torch.isfinite(committed_kl) and committed_kl <= target_kl:
                # Adam consumed this gradient exactly once. Keeping its candidate
                # moment state is equivalent to accepting the same update with a
                # smaller one-step learning rate; only the parameter delta changes.
                return (
                    candidate_kl,
                    committed_kl,
                    step_scale,
                    False,
                    backtrack_count,
                )

        self._restore_parameters(parameters, parameter_before)
        self.optimizer.load_state_dict(optimizer_before)
        return candidate_kl, self._measure_policy_kl(
            batch_obs,
            batch_pre_tanh_actions,
            batch_old_log_probs,
        ), 0.0, True, _MAX_KL_BACKTRACKS

    def compute_gae(
        self,
        rewards,
        values,
        dones,
        truncations,
        timeout_values,
        last_values,
    ):
        """Compute GAE while distinguishing true termination from TimeLimit truncation."""
        # rewards: [n_steps, n_envs]
        # values:  [n_steps, n_envs]
        # last_values: [n_envs], the bootstrap value estimate for s_T.
        n_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards, device=self.device)
        last_values = torch.as_tensor(last_values, dtype=torch.float32, device=self.device)
        last_adv = torch.zeros(n_envs, device=self.device)

        if torch.any(truncations > dones):
            raise ValueError("truncations must be a subset of dones")

        for t in reversed(range(n_steps)):
            episode_continues = 1.0 - dones[t]
            terminated = dones[t] * (1.0 - truncations[t])
            rollout_next_value = last_values if t == n_steps - 1 else values[t + 1]
            bootstrap_value = torch.where(
                truncations[t].bool(),
                timeout_values[t],
                rollout_next_value,
            )
            delta = (
                rewards[t]
                + self.args.gamma * bootstrap_value * (1.0 - terminated)
                - values[t]
            )
            # VecEnv has already reset after done; both done types must stop GAE from
            # flowing backward into the previous episode.
            last_adv = (
                delta
                + self.args.gamma
                * self.args.gae_lambda
                * episode_continues
                * last_adv
            )
            advantages[t] = last_adv

        returns = advantages + values
        return advantages, returns

    def update_policy(
        self,
        obs,
        pre_tanh_actions,
        old_log_probs,
        advantages,
        returns,
    ):
        """Update the policy."""
        # Normalize advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare the training batch.
        batch_size = obs.shape[0] * obs.shape[1]  # n_steps * n_envs
        obs = obs.view(batch_size, -1)
        pre_tanh_actions = pre_tanh_actions.view(batch_size, -1)
        old_log_probs = old_log_probs.view(batch_size)
        advantages = advantages.view(batch_size)
        returns = returns.view(batch_size)

        expected_obs_dim = self.raw_obs_dim
        if obs.shape[1] != expected_obs_dim:
            raise ValueError(
                f"PPO batch observation width mismatch: actual={obs.shape[1]}, expected={expected_obs_dim}"
            )
        print(f"   Observation width is valid: {obs.shape[1]} (expected {expected_obs_dim})")

        # Multi-epoch update.
        policy_losses = []
        value_losses = []
        entropy_losses = []
        entropies = []
        pre_step_kls = []
        candidate_post_step_kls = []
        committed_post_step_kls = []
        clipfracs = []
        grad_norms = []
        step_scales = []
        kl_backtrack_count = 0
        kl_rejected_steps = 0
        kl_precheck_stops = 0

        # Compute the explained variance of the initial value predictions.
        with torch.no_grad():
            _, _, _, initial_values = self.network.get_action_and_value(
                obs,
                estimate_entropy=False,
            )
            y_pred = initial_values
            y_true = returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

        _kl_early_stop = False
        for epoch in range(self.args.n_epochs):
            if _kl_early_stop:
                break
            # Shuffle the batch.
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.args.batch_size):
                end = start + self.args.batch_size
                batch_indices = indices[start:end]

                batch_obs = obs[batch_indices]
                batch_pre_tanh_actions = pre_tanh_actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass.
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch_obs,
                    pre_tanh_action=batch_pre_tanh_actions,
                )

                # Compute the ratio.
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)

                # Compute the clip fraction.
                clip_fraction = torch.mean(((ratio - 1.0).abs() > self.args.clip_range).float())
                clipfracs.append(clip_fraction.item())

                # KL is the precondition for whether this minibatch may update.
                # Once it is over the limit, we must not perform another optimizer step.
                with torch.no_grad():
                    pre_step_kl = self._approx_kl(log_ratio)
                    pre_step_kls.append(pre_step_kl.item())
                if not torch.isfinite(pre_step_kl):
                    raise FloatingPointError(
                        "non-finite PPO KL before optimizer step"
                    )
                if self.args.target_kl and pre_step_kl > self.args.target_kl:
                    _kl_early_stop = True
                    kl_precheck_stops += 1
                    break

                # PPO loss.
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.args.clip_range, 1 + self.args.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss.
                value_loss = nn.MSELoss()(new_values, batch_returns)

                # Entropy loss.
                entropy_loss = -entropy.mean()

                # Total loss.
                total_loss = policy_loss + self.args.vf_coef * value_loss + self.current_entropy_coef * entropy_loss

                # Backpropagation.
                self.optimizer.zero_grad()
                total_loss.backward()

                # Compute the gradient norm.
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.max_grad_norm)
                grad_norms.append(grad_norm.item())

                (
                    candidate_post_step_kl,
                    committed_post_step_kl,
                    step_scale,
                    step_rejected,
                    backtrack_count,
                ) = self._transactional_optimizer_step(
                    batch_obs,
                    batch_pre_tanh_actions,
                    batch_old_log_probs,
                )
                candidate_post_step_kls.append(candidate_post_step_kl.item())
                committed_post_step_kls.append(committed_post_step_kl.item())
                step_scales.append(step_scale)
                kl_backtrack_count += backtrack_count
                if step_rejected:
                    kl_rejected_steps += 1
                    _kl_early_stop = True
                    break

                # Record statistics.
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                entropies.append(entropy.mean().item())

        return {
            "policy_loss": self._mean_or_zero(policy_losses),
            "value_loss": self._mean_or_zero(value_losses),
            "entropy_loss": self._mean_or_zero(entropy_losses),
            "entropy": self._mean_or_zero(entropies),
            "approx_kl": self._mean_or_zero(committed_post_step_kls),
            "max_pre_step_kl": max(pre_step_kls, default=0.0),
            "max_candidate_kl": max(candidate_post_step_kls, default=0.0),
            "max_committed_kl": max(committed_post_step_kls, default=0.0),
            "mean_step_scale": self._mean_or_zero(step_scales),
            "kl_backtrack_count": kl_backtrack_count,
            "kl_rejected_steps": kl_rejected_steps,
            "kl_precheck_stops": kl_precheck_stops,
            "clipfrac": self._mean_or_zero(clipfracs),
            "explained_variance": explained_var.item(),
            "grad_norm": self._mean_or_zero(grad_norms),
        }
