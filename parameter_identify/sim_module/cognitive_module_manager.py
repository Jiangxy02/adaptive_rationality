"""
Cognitive module manager utilities.

This module is responsible for initializing, configuring, coordinating, and
managing the lifecycle of cognitive modules. It extracts the cognitive-module
logic from the original ``PPOCheckpointSimulator`` implementation.
"""


import argparse
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path
import sys

from common.cognitive_input import cognitive_mask_values

# Import cognitive modules.
from cognitive_module.cognitive_bias_module import CognitiveBiasModule
from cognitive_module.cognitive_delay_module import CognitiveDelayModule
from cognitive_module.cognitive_perception_module import CognitivePerceptionModule


class CognitiveModuleManager:
    """Cognitive module manager.

    Responsibilities:
    1. Initialize and configure cognitive modules.
    2. Manage their lifecycle when attaching to or detaching from environments.
    3. Run the cognitive processing pipeline for observations, actions, and rewards.
    4. Build and manage concatenated cognitive-parameter vectors.
    """

    def __init__(self, args: Optional[argparse.Namespace] = None):
        """
        Initialize the cognitive module manager.

        Args:
            args: Command-line arguments that include cognitive-module config.
        """
        self.args = args
        self.use_cognitive_modules = args and getattr(args, 'use_cognitive_modules', False)
        self.network_signature = None

        # Cognitive-module instances.
        self.cognitive_bias_module = None
        self.cognitive_delay_module = None
        self.cognitive_perception_module = None

        # Cognitive-parameter bookkeeping.
        self._cognitive_params_logged = False
        self._extra_param_warning_logged = False
        self._extra_mask_warning_logged = False

        if self.use_cognitive_modules:
            self._initialize_modules()
        else:
            print("CognitiveModuleManager: cognitive modules are disabled")

    def _initialize_modules(self):
        """Initialize every enabled cognitive module."""
        # Initialize perception first because the bias module depends on it.
        if self.args and getattr(self.args, 'use_cognitive_perception', False):
            self._initialize_perception_module()

        # Initialize the bias module with the perception-module reference.
        if self.args and getattr(self.args, 'use_cognitive_bias', False):
            self._initialize_bias_module()

        # Initialize the delay module.
        if self.args and getattr(self.args, 'use_cognitive_delay', False):
            self._initialize_delay_module()

    def _initialize_perception_module(self):
        """Initialize the cognitive-perception module."""
        perception_config = {
            'sigma0': getattr(self.args, 'perception_sigma0', 0.1),
            'sigma_max': getattr(self.args, 'perception_sigma_max', 0.8),
            'far_distance': 150.0,
            'use_ar1': True,
            'rho': 0.8,
            'use_kf': getattr(self.args, 'perception_use_kf', True),
            'kf_dt': getattr(self.args, 'perception_kf_dt', 0.1),
            'kf_q_scale': getattr(self.args, 'perception_kf_q_scale', 100.0),
            'random_seed': getattr(self.args, 'perception_random_seed', None),
        }
        self.cognitive_perception_module = CognitivePerceptionModule(noise_config=perception_config)

        # Enable radar-beam visualization when requested.
        if getattr(self.args, 'enable_radar_beam_viz', False):
            self.cognitive_perception_module.enable_radar_visualization(True)

    def _initialize_bias_module(self):
        """Initialize the cognitive-bias module."""
        bias_config = {
            'inverse_tta_coef': getattr(self.args, 'bias_inverse_tta_coef', 1.5),
            'tta_threshold': getattr(self.args, 'bias_tta_threshold', 0.1),
            'visual_detection_distance': getattr(self.args, 'bias_visual_distance', 50.0),
            'verbose': True
        }
        self.cognitive_bias_module = CognitiveBiasModule(
            bias_config=bias_config,
            cognitive_perception_module=self.cognitive_perception_module
        )

    def _initialize_delay_module(self):
        """Initialize the cognitive-delay module."""
        delay_steps = int(getattr(self.args, 'delay_steps', 2))
        self.cognitive_delay_module = CognitiveDelayModule(
            delay_steps=delay_steps,
            enable_smoothing=False,
            smoothing_factor=0.3,
            enable_visualization=True
        )

    def attach_to_env(self, env) -> bool:
        """
        Attach cognitive modules to an environment.

        Args:
            env: MetaDrive environment instance.

        Returns:
            Whether every enabled module attached successfully.
        """
        if not self.use_cognitive_modules:
            return False

        success_count = 0
        total_modules = 0

        # Attach the cognitive-perception module.
        if self.cognitive_perception_module:
            total_modules += 1
            self.cognitive_perception_module.attach_to_env(env)
            success_count += 1

        # Attach the cognitive-bias module.
        if self.cognitive_bias_module:
            total_modules += 1
            success = self.cognitive_bias_module.attach_to_env(env)
            if success:
                success_count += 1
            else:
                print("Failed to attach the cognitive-bias module")

        return success_count == total_modules

    def detach_from_env(self):
        """Detach cognitive modules from the environment."""
        if not self.use_cognitive_modules:
            return

        if self.cognitive_perception_module:
            try:
                self.cognitive_perception_module.detach_from_env()
            except Exception as e:
                print(f"Failed to detach the cognitive-perception module: {e}")

        if self.cognitive_bias_module:
            try:
                self.cognitive_bias_module.detach_from_env()
            except Exception as e:
                print(f"Failed to detach the cognitive-bias module: {e}")

    def reset_modules(self):
        """Reset all cognitive-module state."""
        if not self.use_cognitive_modules:
            return

        if self.cognitive_bias_module:
            self.cognitive_bias_module.reset()
        if self.cognitive_delay_module:
            self.cognitive_delay_module.reset()
        if self.cognitive_perception_module:
            self.cognitive_perception_module.reset()

    def process_observation(self, obs: np.ndarray, theta: Dict[str, float] = None) -> np.ndarray:
        """
        Compatibility entry point that builds observations from the current
        network signature for legacy call paths.
        """
        signature = self.network_signature or {}
        base_obs_dim = int(signature.get('base_obs_dim', 275))

        cognitive_modulation = signature.get('cognitive_modulation', getattr(self.args, 'cognitive_modulation', 'none') if self.args else 'none')
        if cognitive_modulation == 'auto':
            cognitive_modulation = 'none'
        cognitive_param_dim = int(signature.get('cognitive_param_dim', getattr(self.args, 'cognitive_param_dim', 0) if self.args else 0) or 0)
        cognitive_mask_dim = int(signature.get('cognitive_mask_dim', getattr(self.args, 'cognitive_mask_dim', 0) if self.args else 0) or 0)

        merged_obs = self.build_network_inputs(
            obs=obs,
            theta=theta,
            base_obs_dim=base_obs_dim,
            cognitive_modulation=cognitive_modulation,
            cognitive_param_dim=cognitive_param_dim,
            cognitive_mask_dim=cognitive_mask_dim,
        )
        return merged_obs

    def set_network_signature(self, signature: Dict[str, int]) -> None:
        """Cache the checkpoint network signature for compatibility paths."""
        self.network_signature = dict(signature or {})

    def build_network_inputs(
        self,
        obs: np.ndarray,
        theta: Dict[str, float],
        base_obs_dim: int,
        cognitive_modulation: str,
        cognitive_param_dim: int,
        cognitive_mask_dim: int,
    ) -> np.ndarray:
        """Build the concatenated observation from the network signature."""

        raw_obs_dim = self._expected_raw_dim(
            base_obs_dim,
            cognitive_modulation,
            cognitive_param_dim,
            cognitive_mask_dim,
        )
        allowed_input_dims = {base_obs_dim, raw_obs_dim}
        base_obs = self._ensure_base_observation(
            obs,
            base_obs_dim,
            allowed_input_dims=allowed_input_dims,
            context="CognitiveModuleManager.build_network_inputs",
        )

        if cognitive_modulation == "none" or cognitive_param_dim == 0:
            target_dim = base_obs_dim
            return self._match_target_dim(base_obs, target_dim)

        if self.use_cognitive_modules:
            cognitive_params = self._build_cognitive_param_vector(theta, cognitive_param_dim)
        else:
            cognitive_params = np.zeros(cognitive_param_dim, dtype=np.float32)

        if cognitive_modulation == "concat":
            cognitive_mask = self._build_cognitive_mask_vector(cognitive_mask_dim)
            return np.concatenate(
                [base_obs, cognitive_params, cognitive_mask],
                axis=-1,
            )

        raise ValueError(f"Unsupported cognitive modulation mode: {cognitive_modulation}")

    def process_action(self, action: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process an action through the cognitive-delay module.

        Args:
            action: Raw action.

        Returns:
            ``(processed_action, delay_applied)``.
        """
        if not self.use_cognitive_modules or not self.cognitive_delay_module:
            return action, False

        delayed_action = self.cognitive_delay_module.process_action(action, is_ppo_mode=True)
        return delayed_action, True

    def process_reward(self, reward: float, env, info: dict) -> Tuple[float, bool, Dict]:
        """
        Process a reward through the cognitive-bias module.

        Args:
            reward: Original reward.
            env: Environment instance.
            info: Environment info dictionary.

        Returns:
            ``(processed_reward, bias_applied, bias_info)``.
        """
        if not self.use_cognitive_modules or not self.cognitive_bias_module:
            return reward, False, {}

        reward_result = self.cognitive_bias_module.process_reward(
            original_reward=reward,
            env=env,
            info=info,
            is_ppo_mode=True
        )
        if not isinstance(reward_result, (tuple, list)) or len(reward_result) != 2:
            raise TypeError(
                "CognitiveBiasModule.process_reward() must return "
                "(adjusted_reward, bias_info)"
            )

        adjusted_reward, bias_info = reward_result
        if not isinstance(bias_info, dict):
            raise TypeError("CognitiveBiasModule bias_info must be a dict")

        adjusted_reward = float(adjusted_reward)
        bias_amount = bias_info.get('bias_applied', 0.0)
        bias_active = bias_info.get('bias_active', False)
        if bias_active and abs(bias_amount) > 1e-6:
            return adjusted_reward, True, bias_info
        return reward, False, bias_info

    def _ensure_base_observation(
        self,
        obs: np.ndarray,
        base_obs_dim: int,
        allowed_input_dims: Optional[set] = None,
        context: str = "CognitiveModuleManager",
    ) -> np.ndarray:
        """Convert the observation to a 1D vector with the requested base width."""
        if isinstance(obs, np.ndarray):
            obs_array = obs
        elif isinstance(obs, (list, tuple)):
            obs_array = np.array(obs, dtype=np.float32)
            if obs_array.dtype == object:
                obs_array = np.array(self._flatten_sequence(obs), dtype=np.float32)
        else:
            obs_array = np.array([obs], dtype=np.float32)

        if obs_array.ndim > 1:
            obs_array = obs_array.flatten()

        obs_array = obs_array.astype(np.float32, copy=False)

        if allowed_input_dims is not None and obs_array.shape[0] not in allowed_input_dims:
            allowed = sorted(dim for dim in allowed_input_dims if dim is not None)
            raise ValueError(
                f"{context}: observation width does not match the checkpoint signature: "
                f"expected_one_of={allowed}, actual={obs_array.shape[0]}"
            )

        if obs_array.shape[0] < base_obs_dim:
            raise ValueError(
                f"{context}: base observation width is too small: "
                f"expected_at_least={base_obs_dim}, actual={obs_array.shape[0]}"
            )

        return obs_array[:base_obs_dim]

    def _match_target_dim(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        if obs.shape[0] == target_dim:
            return obs.astype(np.float32, copy=False)
        raise ValueError(
            "Observation width does not match the checkpoint signature: "
            f"expected={target_dim}, actual={obs.shape[0]}"
        )

    def _expected_raw_dim(
        self,
        base_obs_dim: int,
        cognitive_modulation: str,
        cognitive_param_dim: int,
        cognitive_mask_dim: int,
    ) -> int:
        if cognitive_modulation == "concat" and cognitive_param_dim:
            return base_obs_dim + cognitive_param_dim + cognitive_mask_dim
        return base_obs_dim

    def _build_cognitive_mask_vector(self, dim: int) -> np.ndarray:
        args = getattr(self, 'args', None)
        values = cognitive_mask_values(
            effects_enabled=bool(self.use_cognitive_modules),
            bias_enabled=bool(getattr(args, 'use_cognitive_bias', False)) if args else False,
            perception_enabled=bool(getattr(args, 'use_cognitive_perception', False)) if args else False,
            delay_enabled=bool(getattr(args, 'use_cognitive_delay', False)) if args else False,
        )
        if dim != len(values):
            raise ValueError(
                "Checkpoint cognitive-mask width does not match the paper input contract: "
                f"cognitive_mask_dim={dim}, expected={len(values)}"
            )
        return np.asarray(values, dtype=np.float32)

    def _build_cognitive_param_vector(self, theta: Optional[Dict[str, float]], dim: int) -> np.ndarray:
        defaults = {
            'bias_inverse_tta_coef': getattr(self.args, 'bias_inverse_tta_coef', 1.5) if self.args else 1.5,
            'perception_sigma0': getattr(self.args, 'perception_sigma0', 0.1) if self.args else 0.1,
            'perception_sigma_max': getattr(self.args, 'perception_sigma_max', 0.8) if self.args else 0.8,
            'delay_steps': getattr(self.args, 'delay_steps', 2) if self.args else 2.0,
        }

        values = [
            (theta or {}).get('bias_inverse_tta_coef', defaults['bias_inverse_tta_coef']),
            (theta or {}).get('perception_sigma0', defaults['perception_sigma0']),
            (theta or {}).get('perception_sigma_max', defaults['perception_sigma_max']),
            float((theta or {}).get('delay_steps', defaults['delay_steps'])),
        ]

        if dim > len(values):
            raise ValueError(
                "Checkpoint cognitive-parameter width exceeds what the current runtime can construct: "
                f"cognitive_param_dim={dim}, supported={len(values)}"
            )

        vector = np.zeros(dim, dtype=np.float32)
        limit = min(dim, len(values))
        for idx in range(limit):
            vector[idx] = values[idx]

        return vector

    def _flatten_sequence(self, seq) -> list:
        result = []

        def _recursive(item):
            if isinstance(item, (list, tuple)):
                for sub in item:
                    _recursive(sub)
            elif isinstance(item, np.ndarray):
                result.extend(item.astype(np.float32).flatten().tolist())
            else:
                try:
                    result.append(float(item))
                except (ValueError, TypeError):
                    result.append(0.0)

        _recursive(seq)
        return result

    def generate_visualizations(self, save_dir: str, env=None):
        """Generate cognitive-module visualizations."""
        if not self.use_cognitive_modules:
            return

        # Generate cognitive-perception visualizations.
        if self.cognitive_perception_module:
            try:
                self.cognitive_perception_module.generate_visualization(save_dir=save_dir, env=env)
            except Exception as e:
                print(f"Failed to generate cognitive-perception visualizations: {e}")

        # Generate cognitive-bias visualizations.
        if self.cognitive_bias_module:
            try:
                self.cognitive_bias_module.generate_visualization(env=env, save_dir=save_dir)
            except Exception as e:
                print(f"Failed to generate cognitive-bias visualizations: {e}")

    def get_statistics(self) -> Dict:
        """Return cognitive-module statistics."""
        stats = {
            'use_cognitive_modules': self.use_cognitive_modules,
            'modules_enabled': {}
        }

        if not self.use_cognitive_modules:
            return stats

        # Cognitive-bias statistics.
        if self.cognitive_bias_module:
            try:
                if hasattr(self.cognitive_bias_module, 'get_statistics'):
                    stats['modules_enabled']['bias'] = self.cognitive_bias_module.get_statistics()
                else:
                    stats['modules_enabled']['bias'] = {'status': 'active'}
            except Exception as e:
                stats['modules_enabled']['bias'] = {'error': str(e)}

        # Cognitive-delay statistics.
        if self.cognitive_delay_module:
            try:
                if hasattr(self.cognitive_delay_module, 'get_statistics'):
                    stats['modules_enabled']['delay'] = self.cognitive_delay_module.get_statistics()
                else:
                    delay_steps = getattr(self.cognitive_delay_module, 'delay_steps', 0)
                    stats['modules_enabled']['delay'] = {'delay_steps': delay_steps, 'status': 'active'}
            except Exception as e:
                stats['modules_enabled']['delay'] = {'error': str(e)}

        # Cognitive-perception statistics.
        if self.cognitive_perception_module:
            try:
                if hasattr(self.cognitive_perception_module, 'get_statistics'):
                    stats['modules_enabled']['perception'] = self.cognitive_perception_module.get_statistics()
                else:
                    noise_config = getattr(self.cognitive_perception_module, 'noise_config', {})
                    stats['modules_enabled']['perception'] = {'noise_config': noise_config, 'status': 'active'}
            except Exception as e:
                stats['modules_enabled']['perception'] = {'error': str(e)}

        return stats

    def get_closest_beam_info(self) -> Dict:
        """Return the nearest radar-beam info for visualization."""
        if self.cognitive_perception_module and hasattr(self.cognitive_perception_module, 'get_closest_beam_info'):
            return self.cognitive_perception_module.get_closest_beam_info()

        return {'original_distance': 0.0, 'noisy_distance': 0.0, 'noise_level': 0.0, 'beam_index': 0}

    def get_bias_info(self) -> Dict:
        """Return bias info for visualization."""
        if self.cognitive_bias_module and hasattr(self.cognitive_bias_module, 'get_bias_info'):
            return self.cognitive_bias_module.get_bias_info()

        return {'bias_strength': 0.0, 'bias_active': False}

    def get_delay_info(self) -> Dict:
        """Return delay info for visualization."""
        if self.cognitive_delay_module:
            if hasattr(self.cognitive_delay_module, 'get_delay_info'):
                return self.cognitive_delay_module.get_delay_info()
            else:
                return {'current_delay': getattr(self.cognitive_delay_module, 'delay_steps', 0)}

        return {'current_delay': 0}
