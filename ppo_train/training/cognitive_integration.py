"""Cognitive-module integration mixin for PPO expert training."""


from typing import Any, Dict

import numpy as np
import torch

from common.cognitive_input import cognitive_mask_values
from ppo_train.config.defaults import (
    BIAS_INVERSE_TTA_COEF_DEFAULT,
    DELAY_STEPS_DEFAULT,
    PERCEPTION_SIGMA0_DEFAULT,
    PERCEPTION_SIGMA_MAX_DEFAULT,
)
from ppo_train.models.ppo_network import PPONetwork


class CognitiveIntegrationMixin:
    """Mixin holding methods split from PPOExpertReproduction."""

    def _determine_network_input_dim(self) -> int:
        """Return the network input width after concatenating base obs, cognitive params, and mask."""
        return self.base_obs_dim + self.cognitive_param_dim + self.cognitive_mask_dim

    def _create_network(self, checkpoint_obs_dim: int = None) -> PPONetwork:
        """Create the PPO network from the current configuration."""
        network = PPONetwork(
            base_obs_dim=self.base_obs_dim,
            action_dim=2,
            hidden_dim=256,
            cognitive_param_dim=self.cognitive_param_dim,
            cognitive_mask_dim=self.cognitive_mask_dim,
            cognitive_modulation=self.cognitive_modulation,
            checkpoint_obs_dim=checkpoint_obs_dim
        ).to(self.device)
        self.network_input_dim = network.obs_dim
        return network

    def _attach_cognitive_modules_to_env(self, env):
        """Attach cognitive modules to the environment."""
        if not self.use_cognitive_modules:
            return

        # Attach the perception module before the bias module.
        if self.cognitive_perception_module:
            self.cognitive_perception_module.reset()
            self.cognitive_perception_module.attach_to_env(env)
            print("Attached the cognitive-perception module to the environment; sensor-layer noise will be injected automatically")

            # Validate the environment config to avoid double-applying noise.
            lidar_config = env.config.get("vehicle_config", {}).get("lidar", {})
            gaussian_noise = lidar_config.get("gaussian_noise", 0.0)
            dropout_prob = lidar_config.get("dropout_prob", 0.0)

            if gaussian_noise > 0.0 or dropout_prob > 0.0:
                print("Warning: detected extra noise in the environment lidar config")
                print(f"   gaussian_noise: {gaussian_noise}")
                print(f"   dropout_prob: {dropout_prob}")
                print("   This can cause a double-noise issue; set both values to 0.0")
            else:
                print("Lidar noise config is correct (gaussian_noise=0.0, dropout_prob=0.0)")

        # Attach the bias module.
        if self.cognitive_bias_module:
            success = self.cognitive_bias_module.attach_to_env(env)
            if success:
                print("Attached the cognitive-bias module to the environment; rewards will be adjusted dynamically from TTA")
            else:
                print("Failed to attach the cognitive-bias module")

    def _detach_cognitive_modules_from_env(self):
        """Detach cognitive modules from the environment."""
        if not self.use_cognitive_modules:
            return

        if self.cognitive_perception_module:
            self.cognitive_perception_module.detach_from_env()
            print("Detached the cognitive-perception module from the environment")

        if self.cognitive_bias_module:
            try:
                self.cognitive_bias_module.detach_from_env()
                print("Detached the cognitive-bias module from the environment")
            except Exception as e:
                print(f"Failed to detach the cognitive-bias module: {e}")

    def _apply_cognitive_parameters(self, new_params: Dict[str, Any]):
        """
        Apply newly sampled cognitive parameters to the corresponding modules.

        Args:
            new_params: New parameter dictionary.
        """

        if self.cognitive_bias_module and 'bias_inverse_tta_coef' in new_params:
            self.cognitive_bias_module.inverse_tta_coef = new_params['bias_inverse_tta_coef']
            if hasattr(self, "envs") and hasattr(self.envs, "env_method"):
                self.envs.env_method(
                    "set_cognitive_bias_inverse_tta_coef",
                    float(new_params['bias_inverse_tta_coef']),
                )

        if self.cognitive_perception_module:
            perception_params_changed = False
            noise_lidar = getattr(self.cognitive_perception_module, 'noise_lidar', None)
            if 'perception_sigma0' in new_params:
                new_sigma0 = float(new_params['perception_sigma0'])
                self.cognitive_perception_module.sigma0 = new_sigma0
                self.cognitive_perception_module.noise_config['sigma0'] = new_sigma0
                if noise_lidar:
                    noise_lidar.sigma0 = new_sigma0
                perception_params_changed = True

            if 'perception_sigma_max' in new_params:
                new_sigma_max = new_params['perception_sigma_max']
                self.cognitive_perception_module.sigma_max = new_sigma_max
                self.cognitive_perception_module.noise_config['sigma_max'] = new_sigma_max
                if noise_lidar:
                    noise_lidar.sigma_max = new_sigma_max
                perception_params_changed = True

            if perception_params_changed:
                derived_k = None
                if noise_lidar and hasattr(noise_lidar, 'recompute_k_from_sigma_max'):
                    derived_k = noise_lidar.recompute_k_from_sigma_max()
                elif hasattr(self.cognitive_perception_module, 'recompute_k_from_sigma_max'):
                    derived_k = self.cognitive_perception_module.recompute_k_from_sigma_max()
                if noise_lidar and derived_k is not None:
                    noise_lidar.k = derived_k

                if hasattr(self, "envs") and hasattr(self.envs, "env_method"):
                    self.envs.env_method(
                        "set_cognitive_perception_parameters",
                        float(self.cognitive_perception_module.noise_config['sigma0']),
                        float(self.cognitive_perception_module.noise_config['sigma_max']),
                    )

        if self.cognitive_delay_module and 'delay_steps' in new_params:
            self.cognitive_delay_module.update_config(delay_steps=new_params['delay_steps'])

    def _concatenate_cognitive_params(self, obs, cognitive_params):
        """
        Concatenate cognitive parameters and their mask onto the observation.

        Args:
            obs: Raw observation ``[n_envs, base_dim]`` or ``[batch_size, base_dim]``.
            cognitive_params: Cognitive-parameter dictionary.

        Returns:
            Expanded observation ``[n_envs, raw_dim]`` or ``[batch_size, raw_dim]``,
            where ``raw_dim = base_dim + cognitive_param_dim + mask_dim``.
        """
        if torch.is_tensor(obs):
            obs_np = obs.cpu().numpy()
        else:
            obs_np = np.asarray(obs)

        if obs_np.ndim != 2 or obs_np.shape[1] not in {
            self.base_obs_dim,
            self.raw_obs_dim,
        }:
            raise ValueError(
                "Observation shape does not match the 283-dim paper contract: "
                f"actual={obs_np.shape}, expected_second_dim="
                f"{self.base_obs_dim} or {self.raw_obs_dim}"
            )

        base_obs = obs_np[:, :self.base_obs_dim]
        if self.use_cognitive_modules:
            cognitive_values = [
                cognitive_params.get('bias_inverse_tta_coef', BIAS_INVERSE_TTA_COEF_DEFAULT),
                cognitive_params.get('perception_sigma0', PERCEPTION_SIGMA0_DEFAULT),
                cognitive_params.get('perception_sigma_max', PERCEPTION_SIGMA_MAX_DEFAULT),
                cognitive_params.get('delay_steps', DELAY_STEPS_DEFAULT),
            ]
        else:
            cognitive_values = [0.0] * self.cognitive_param_dim

        mask_values = cognitive_mask_values(
            effects_enabled=self.use_cognitive_modules,
            bias_enabled=getattr(self.args, "use_cognitive_bias", False),
            perception_enabled=getattr(self.args, "use_cognitive_perception", False),
            delay_enabled=getattr(self.args, "use_cognitive_delay", False),
        )

        cognitive_vector = np.tile(
            np.asarray(cognitive_values, dtype=np.float32),
            (base_obs.shape[0], 1),
        )
        cognitive_mask = np.tile(
            np.asarray(mask_values, dtype=np.float32),
            (base_obs.shape[0], 1),
        )
        obs_with_cognitive = np.concatenate(
            [base_obs, cognitive_vector, cognitive_mask],
            axis=1,
        )

        if obs_with_cognitive.shape[1] != self.raw_obs_dim:
            raise ValueError(
                f"Cognitive observation width mismatch: {obs_with_cognitive.shape[1]} "
                f"!= expected {self.raw_obs_dim}"
            )

        return obs_with_cognitive
