
import torch
import numpy as np
import sys
import os
from pathlib import Path
import importlib.util

from common.ppo_network_base import PPONetworkBase

class PPOController:
    """Load PPO checkpoints and provide action inference."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        use_cognitive_modules: bool = False,
        cognitive_modulation_override: str = None,
    ):
        """Initialize the PPO controller."""
        self.checkpoint_path = checkpoint_path
        self.use_cognitive_modules = use_cognitive_modules
        self.cognitive_modulation_override = cognitive_modulation_override

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print("Initializing PPO controller...")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Cognitive modules: {'enabled' if use_cognitive_modules else 'disabled'}")

        self._debug_printed = False

        self._load_sim_modules()

        self._load_network()

    def _load_sim_modules(self):
        """Load the shared network base and the sim-module checkpoint loader."""
        current_dir = Path(__file__).parent.absolute()
        sim_module_path = current_dir.parent / "sim_module"

        self.PPONetwork = PPONetworkBase

        self.CheckpointLoader = self._import_checkpoint_loader(sim_module_path)

    def _import_checkpoint_loader(self, sim_module_path):
        """Import the checkpoint loader with fallbacks."""
        try:
            from parameter_identify.sim_module.checkpoint_loader import CheckpointLoader
            print("Imported the native CheckpointLoader")
            return CheckpointLoader

        except Exception as e1:
            print(f"Strategy 1 failed: {e1}")

            try:
                checkpoint_loader_spec = importlib.util.spec_from_file_location(
                    "checkpoint_loader", sim_module_path / "checkpoint_loader.py"
                )
                checkpoint_loader_module = importlib.util.module_from_spec(checkpoint_loader_spec)
                checkpoint_loader_spec.loader.exec_module(checkpoint_loader_module)
                CheckpointLoader = checkpoint_loader_module.CheckpointLoader
                print("Imported CheckpointLoader with importlib")
                return CheckpointLoader

            except Exception as e2:
                print(f"Strategy 2 failed: {e2}")
                print("Falling back to the simplified checkpoint loader...")
                return self._create_fallback_checkpoint_loader()

    def _create_fallback_checkpoint_loader(self):
        """Create a simplified fallback checkpoint loader."""
        class FallbackCheckpointLoader:
            def __init__(self, checkpoint_path: str, device: str = "auto"):
                self.checkpoint_path = checkpoint_path
                self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
                self.checkpoint = None
                self.network = None

                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"checkpoint file does not exist: {checkpoint_path}")

            def load_checkpoint(self, use_cognitive_modules: bool = False):
                print(f"Loading checkpoint: {os.path.basename(self.checkpoint_path)}")

                checkpoint_loader = self.CheckpointLoader(self.checkpoint_path, str(self.device))
                self.network = checkpoint_loader.load_checkpoint(use_cognitive_modules)
                self.checkpoint = checkpoint_loader.checkpoint

                self.network.eval()

                print("Checkpoint load complete")
                return self.network

            def get_checkpoint_info(self):
                if self.checkpoint:
                    return {
                        'iteration': self.checkpoint.get('iteration', 'Unknown'),
                        'global_step': self.checkpoint.get('global_step', 'Unknown')
                    }
                return {'iteration': 'Unknown', 'global_step': 'Unknown'}

            def get_ppo_network_class(self):
                return self._ppo_network_class

        print("Using the simplified CheckpointLoader")
        return FallbackCheckpointLoader

    def _load_network(self):
        """Load the PPO network."""
        try:
            checkpoint_loader = self.CheckpointLoader(self.checkpoint_path, str(self.device))

            if hasattr(checkpoint_loader, '_ppo_network_class'):
                checkpoint_loader._ppo_network_class = self.PPONetwork

            modulation_override = getattr(self, 'cognitive_modulation_override', None)
            self.network = checkpoint_loader.load_checkpoint(
                self.use_cognitive_modules,
                cognitive_modulation_override=modulation_override
            )
            self.checkpoint_info = checkpoint_loader.get_checkpoint_info()
            self.network_signature = {
                'obs_dim': getattr(self.network, 'obs_dim', None),
                'raw_obs_dim': getattr(self.network, 'raw_obs_dim', getattr(self.network, 'obs_dim', None)),
                'base_obs_dim': getattr(self.network, 'base_obs_dim', None),
                'cognitive_param_dim': getattr(self.network, 'cognitive_param_dim', 0),
                'cognitive_mask_dim': getattr(self.network, 'cognitive_mask_dim', 0),
                'cognitive_modulation': getattr(self.network, 'cognitive_modulation', 'none'),
            }

            print("PPO network loaded successfully")
            print(f"Training iteration: {self.checkpoint_info.get('iteration', 'Unknown')}")
            print(f"Global step: {self.checkpoint_info.get('global_step', 'Unknown')}")

        except Exception as e:
            print(f"Failed to load PPO network: {e}")
            raise

    def get_action(self, obs, deterministic: bool = True):
        """Get an action from the PPO network."""
        try:
            obs_array = self._process_observation(obs)

            obs_for_net = self._prepare_network_inputs(obs_array)

            obs_tensor = torch.as_tensor(obs_for_net, dtype=torch.float32, device=self.device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

            if not self._debug_printed:
                print("Observation shape debug:")
                print(f"   Original observation type: {type(obs)}")
                print(f"   Processed observation shape: {obs_tensor.shape}")
                print(f"   Expected network dimension: {self.network.obs_dim}")
                self._debug_printed = True

            with torch.no_grad():
                if deterministic and hasattr(self.network, 'act_deterministic'):
                    action_tensor = self.network.act_deterministic(obs_tensor)
                else:
                    action_tensor, _, _, _ = self.network.get_action_and_value(obs_tensor)

            action_np = action_tensor.cpu().numpy().flatten()

            return action_np

        except Exception as e:
            raise RuntimeError(
                f"failed to get PPO action: {e}; observation_type={type(obs)}, "
                f"observation_shape={getattr(obs, 'shape', None)}"
            ) from e

    def _process_observation(self, obs):
        """Convert supported observation formats into one NumPy array."""
        if isinstance(obs, tuple):
            obs_array = obs[0] if len(obs) > 0 else obs
            print(f"Detected tuple observation; using the first element: {type(obs_array)}")
        elif isinstance(obs, list):
            obs_array = np.array(obs)
            print(f"Detected list observation; converted to NumPy: {obs_array.shape}")
        elif isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.array(obs)
            print(f"Detected {type(obs)} observation; converted to NumPy: {obs_array.shape}")

        if not isinstance(obs_array, np.ndarray):
            obs_array = np.array(obs_array)

        if obs_array.ndim > 1:
            obs_array = obs_array.flatten()

        obs_array = obs_array.astype(np.float32, copy=False)

        return obs_array

    def _prepare_network_inputs(self, obs_array):
        modulation = getattr(self.network, 'cognitive_modulation', 'none')
        base_dim = int(getattr(self.network, 'base_obs_dim', obs_array.shape[0]))
        param_dim = int(getattr(self.network, 'cognitive_param_dim', 0))
        mask_dim = int(getattr(self.network, 'cognitive_mask_dim', 0))
        raw_dim = int(getattr(self.network, 'raw_obs_dim', getattr(self.network, 'obs_dim', obs_array.shape[0])))

        if modulation == 'none' or param_dim == 0:
            self._require_dim(obs_array, raw_dim, 'PPOController raw observation')
            return obs_array

        if modulation == 'concat':
            base_obs = self._extract_base_observation(
                obs_array,
                base_dim,
                allowed_lengths={base_dim, raw_dim},
            )
            params_np = (
                self._default_cognitive_params(param_dim)
                if self.use_cognitive_modules
                else np.zeros(param_dim, dtype=np.float32)
            )
            masks_np = (
                np.ones(mask_dim, dtype=np.float32)
                if self.use_cognitive_modules
                else np.zeros(mask_dim, dtype=np.float32)
            )
            return np.concatenate([base_obs, params_np, masks_np], axis=0).astype(np.float32)

        raise ValueError(f"unsupported cognitive modulation mode: {modulation}")

    def _extract_base_observation(self, obs_array, base_dim, allowed_lengths=None):
        if allowed_lengths is not None and obs_array.shape[0] not in allowed_lengths:
            allowed = sorted(dim for dim in allowed_lengths if dim is not None)
            raise ValueError(
                "PPOController observation dimension does not match the checkpoint signature: "
                f"expected_one_of={allowed}, actual={obs_array.shape[0]}"
            )
        if obs_array.shape[0] < base_dim:
            raise ValueError(
                "PPOController base observation dimension is too small: "
                f"expected_at_least={base_dim}, actual={obs_array.shape[0]}"
            )
        return obs_array[:base_dim].astype(np.float32, copy=False)

    def _require_dim(self, obs_array, expected_dim, context):
        if obs_array.shape[0] != expected_dim:
            raise ValueError(f"{context}: expected_dim={expected_dim}, actual={obs_array.shape[0]}")

    def _default_cognitive_params(self, dim):
        values = [
            getattr(self, '_current_bias_coef', 1.5),
            getattr(self, '_current_sigma0', 0.1),
            getattr(self, '_current_sigma_max', 0.8),
            float(getattr(self, '_current_delay', 2.0)),
        ]
        if dim > len(values):
            raise ValueError(
                "checkpoint cognitive-parameter dimension exceeds what PPOController can construct: "
                f"cognitive_param_dim={dim}, supported={len(values)}"
            )
        vector = np.zeros(dim, dtype=np.float32)
        limit = min(dim, len(values))
        vector[:limit] = np.asarray(values[:limit], dtype=np.float32)
        return vector

    def get_checkpoint_info(self):
        """Return cached checkpoint metadata."""
        return getattr(self, 'checkpoint_info', {'iteration': 'Unknown', 'global_step': 'Unknown'})
