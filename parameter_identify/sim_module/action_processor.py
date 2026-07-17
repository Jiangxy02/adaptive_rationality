"""
Action processor utilities.

This module is responsible for PPO policy action generation and post-processing.
It extracts action-handling logic from the original
``PPOCheckpointSimulator`` implementation.
"""


import torch
import numpy as np
from typing import Optional, TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from common.ppo_network_base import PPONetworkBase as PPONetwork
    from .cognitive_module_manager import CognitiveModuleManager


class ActionProcessor:
    """PPO action processor.

    Responsibilities:
    1. Generate actions from the PPO policy.
    2. Concatenate cognitive parameters into observations.
    3. Clamp and post-process actions.
    4. Coordinate with cognitive modules.
    """

    def __init__(self, network: 'PPONetwork', cognitive_manager: Optional['CognitiveModuleManager'] = None, device: str = "cpu"):
        """
        Initialize the action processor.

        Args:
            network: PPO network instance.
            cognitive_manager: Cognitive module manager.
            device: Compute device.
        """
        self.network = network
        self.cognitive_manager = cognitive_manager
        self.device = torch.device(device)

        # Ensure the network lives on the requested device.
        if hasattr(self.network, 'to'):
            self.network = self.network.to(self.device)

        # Always run the policy in evaluation mode here.
        if hasattr(self.network, 'eval'):
            self.network.eval()

        self._refresh_network_signature()

        print("Action processor initialized")
        print(f"   Network observation dimension: {self._network_input_dim}")
        print(f"   Compute device: {self.device}")
        print(f"   Cognitive modules: {'enabled' if cognitive_manager else 'disabled'}")

    def get_action(self, observation: np.ndarray, deterministic: bool = True, env=None, step_count: int = 0, theta: Dict[str, float] = None) -> np.ndarray:
        """
        Get a PPO policy action from the observation.

        Args:
            observation: Environment observation.
            deterministic: Whether to use deterministic inference.
            env: Environment instance, used by cognitive-module processing.
            step_count: Current step number, used for debug output.
            theta: Optional cognitive-parameter dictionary for the current particle.

        Returns:
            Action array ``[steering, acceleration]``.
        """
        # Step 1: process the observation through the cognitive pipeline.
        obs_input = self._prepare_network_inputs(
            observation, env, step_count, theta
        )

        # Step 2: get the raw action from the PPO network.
        raw_action = self._get_raw_action(
            obs_input,
            deterministic=deterministic,
            step_count=step_count,
        )

        # Step 3: process the action through cognitive modules such as delay.
        final_action = self._process_action(raw_action, step_count)

        # Step 4: clamp the action into the valid range.
        final_action = self._clip_action(final_action)

        return final_action

    def _prepare_network_inputs(
        self,
        observation: np.ndarray,
        env=None,
        step_count: int = 0,
        theta: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Prepare the concatenated input vector for the current network configuration."""

        # Refresh the cached network signature as a safety net.
        if self._network_input_dim is None and self.network is not None:
            self._refresh_network_signature()

        # Resolve the expected input width.
        target_dim = self._network_input_dim
        if target_dim is None and self.network is not None:
            target_dim = getattr(self.network, 'obs_dim', None)

        # Fail hard if the target dimension cannot be determined.
        if target_dim is None:
            raise ValueError(f"_prepare_network_inputs: unable to determine the network input dimension. observation_shape={observation.shape if hasattr(observation, 'shape') else 'unknown'}, network={type(self.network).__name__ if self.network else 'None'}")

        if self._cognitive_param_dim > 0 and self._cognitive_modulation == "concat":
            if self.cognitive_manager:
                obs_input = self.cognitive_manager.build_network_inputs(
                    obs=observation,
                    theta=theta or {},
                    base_obs_dim=self._base_obs_dim,
                    cognitive_modulation=self._cognitive_modulation,
                    cognitive_param_dim=self._cognitive_param_dim,
                    cognitive_mask_dim=self._cognitive_mask_dim,
                )
            else:
                obs_array = self._ensure_numpy_array(observation)
                allowed_dims = {self._base_obs_dim, target_dim}
                if obs_array.shape[0] not in allowed_dims:
                    raise ValueError(
                        "_prepare_network_inputs: observation width without cognitive effects does not match the signature: "
                        f"expected_one_of={sorted(allowed_dims)}, actual={obs_array.shape[0]}"
                    )
                base_obs = obs_array[:self._base_obs_dim]
                obs_input = np.concatenate([
                    base_obs,
                    np.zeros(self._cognitive_param_dim, dtype=np.float32),
                    np.zeros(self._cognitive_mask_dim, dtype=np.float32),
                ])
            if step_count < 3:
                print(
                    f"    ActionProcessor observation transform: {np.shape(observation)} -> {obs_input.shape}, target_dim={target_dim}"
                )
            obs_final = self._ensure_observation_dim(obs_input, target_dim, "_prepare_network_inputs")
            return obs_final

        obs_array = self._ensure_numpy_array(observation)
        obs_prepared = self._ensure_observation_dim(obs_array, target_dim, "_prepare_network_inputs")
        return obs_prepared

    def _flatten_observation(self, obs):
        """Recursively flatten an observation sequence."""
        result = []

        def _recursive_flatten(item):
            if isinstance(item, (list, tuple)):
                for sub_item in item:
                    _recursive_flatten(sub_item)
            elif isinstance(item, np.ndarray):
                for sub_item in item.flatten():
                    result.append(float(sub_item))
            else:
                try:
                    result.append(float(item))
                except (ValueError, TypeError):
                    result.append(0.0)

        _recursive_flatten(obs)
        return result

    def _get_raw_action(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
        step_count: int = 0,
    ) -> np.ndarray:
        """Get the raw action from the PPO network."""

        # Ensure the observation is a NumPy array.
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
        observation = observation.flatten()

        # Refresh the cached network signature as a safety net.
        if self._network_input_dim is None and self.network is not None:
            self._refresh_network_signature()

        # Enforce an exact input-width match.
        target_dim = self._network_input_dim
        if target_dim is None and self.network is not None:
            target_dim = getattr(self.network, 'obs_dim', None)

        # Fail hard if the target dimension is still unavailable.
        if target_dim is None:
            raise ValueError(f"Unable to determine the network input dimension. observation_dim={observation.shape[0]}, network={type(self.network).__name__}")

        if observation.shape[0] != target_dim:
            raise ValueError(
                "PPO input width does not match the checkpoint signature: "
                f"expected_raw_input_dim={target_dim}, actual={observation.shape[0]}, "
                f"network_obs_dim={getattr(self.network, 'obs_dim', None)}, "
                f"modulation={self._cognitive_modulation}"
            )

        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Final tensor-dimension check.
        if obs_tensor.shape[1] != target_dim:
            raise ValueError(f"Tensor dimension mismatch: expected {target_dim}, got {obs_tensor.shape[1]}")

        with torch.no_grad():
            if deterministic and hasattr(self.network, 'act_deterministic'):
                action_tensor = self.network.act_deterministic(obs_tensor)
                mode = "deterministic"
            else:
                action_tensor, _, _, _ = self.network.get_action_and_value(obs_tensor)
                mode = "stochastic"

        action = action_tensor.squeeze(0).cpu().numpy()

        if step_count < 3:
            print(f"    Policy sampling mode: {mode}")
            print(f"    Network output: [{action[0]:.3f}, {action[1]:.3f}]")

        return action

    def _process_action(self, action: np.ndarray, step_count: int = 0) -> np.ndarray:
        """Process the action through cognitive modules such as delay."""
        if self.cognitive_manager:
            processed_action, delay_applied = self.cognitive_manager.process_action(action)

            if step_count < 3 and delay_applied:
                print(f"    Delay processing: [{action[0]:.3f}, {action[1]:.3f}] -> [{processed_action[0]:.3f}, {processed_action[1]:.3f}]")

            return processed_action
        else:
            return action

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        """Clamp the action into the valid range."""
        clipped_action = np.clip(action, -1.0, 1.0)
        return clipped_action

    def update_network(self, new_network: 'PPONetwork'):
        """Update the wrapped network instance."""
        self.network = new_network
        if hasattr(self.network, 'to'):
            self.network = self.network.to(self.device)
        if hasattr(self.network, 'eval'):
            self.network.eval()

        self._refresh_network_signature()

        print("Action processor network updated")
        print(f"   New network observation dimension: {self._network_input_dim}")

    def update_cognitive_manager(self, new_cognitive_manager: Optional['CognitiveModuleManager']):
        """Update the cognitive-module manager."""
        self.cognitive_manager = new_cognitive_manager
        print(f"Action processor cognitive manager updated: {'enabled' if new_cognitive_manager else 'disabled'}")

    def get_network_info(self) -> dict:
        """Return network metadata."""
        return {
            'obs_dim': getattr(self.network, 'obs_dim', 'Unknown'),
            'action_dim': getattr(self.network, 'action_dim', 'Unknown'),
            'device': str(self.device),
            'network_type': type(self.network).__name__ if self.network else 'None',
            'cognitive_enabled': self.cognitive_manager is not None
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _refresh_network_signature(self) -> None:
        self._network_input_dim = getattr(self.network, 'raw_obs_dim', getattr(self.network, 'obs_dim', None))
        self._base_obs_dim = getattr(self.network, 'base_obs_dim', self._network_input_dim)
        self._cognitive_param_dim = getattr(self.network, 'cognitive_param_dim', 0)
        self._cognitive_mask_dim = getattr(self.network, 'cognitive_mask_dim', 0)
        self._cognitive_modulation = getattr(self.network, 'cognitive_modulation', 'none')

        if self._network_input_dim is None and self._base_obs_dim is not None:
            self._network_input_dim = self._base_obs_dim

    def _ensure_numpy_array(self, observation: np.ndarray) -> np.ndarray:
        if isinstance(observation, np.ndarray):
            obs_array = observation
        elif isinstance(observation, (list, tuple)):
            try:
                obs_array = np.array(observation, dtype=np.float32)
            except ValueError:
                obs_array = np.array(self._flatten_observation(observation), dtype=np.float32)
        else:
            obs_array = np.array([observation], dtype=np.float32)

        if obs_array.ndim > 1:
            obs_array = obs_array.flatten()

        return obs_array.astype(np.float32, copy=False)

    def _ensure_observation_dim(self, obs: np.ndarray, target_dim: Optional[int], context: str) -> np.ndarray:
        if target_dim is None:
            raise ValueError(f"{context}: unable to determine the network input dimension from the checkpoint signature")
        if obs.shape[0] != target_dim:
            raise ValueError(
                f"{context}: PPO input width does not match the checkpoint signature: "
                f"expected_raw_input_dim={target_dim}, actual={obs.shape[0]}, "
                f"network_obs_dim={getattr(self.network, 'obs_dim', None)}, "
                f"base_obs_dim={self._base_obs_dim}, "
                f"cognitive_param_dim={self._cognitive_param_dim}, "
                f"cognitive_mask_dim={self._cognitive_mask_dim}, "
                f"cognitive_modulation={self._cognitive_modulation}"
            )
        return obs.astype(np.float32, copy=False)

    def _match_observation_dim(self, obs: np.ndarray, target_dim: Optional[int], step_count: int) -> np.ndarray:
        """Compatibility path for older internal callers; padding and truncation are no longer allowed."""
        return self._ensure_observation_dim(obs, target_dim, "_match_observation_dim")
