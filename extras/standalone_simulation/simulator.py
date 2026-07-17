#!/usr/bin/env python3
"""
PPO checkpoint simulator
Control the ego vehicle in MetaDrive with a trained PPO checkpoint.
Supports loading a chosen checkpoint, visualization runs, and performance evaluation.
Integrates cognitive modules: bias, delay, and perception.
"""


import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from collections import defaultdict, deque

from common.cognitive_input import PERCEPTION_SIGMA_UNIT

from extras.standalone_simulation.cognitive_viz import CognitiveVizMixin
from extras.standalone_simulation.speed_control_viz import SpeedControlVizMixin
from parameter_identify.sim_module.checkpoint_signature import (
    load_current_public_network,
)

# Import cognitive modules
from cognitive_module.cognitive_bias_module import CognitiveBiasModule
from cognitive_module.cognitive_delay_module import CognitiveDelayModule
from cognitive_module.cognitive_perception_module import CognitivePerceptionModule

# Import environment classes
import metadrive
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.obs.state_obs import LidarStateObservation
import sys
# Import the speed-control environment when available
try:
    from ppo_train.envs.speed_control_env import SpeedControlMetaDriveEnv
    SPEED_CONTROL_AVAILABLE = True
    print("Imported SpeedControlMetaDriveEnv successfully")
except ImportError:
    SPEED_CONTROL_AVAILABLE = False
    print("Failed to import SpeedControlMetaDriveEnv; falling back to the standard MetaDriveEnv")

class PPOCheckpointSimulator(CognitiveVizMixin, SpeedControlVizMixin):
    """PPO checkpoint simulator"""

    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None, device: str = "auto", args: Optional[argparse.Namespace] = None):
        """
        Initialize the simulator

        Args:
            checkpoint_path: checkpoint file path
            config_path: config file path (optional)
            device: compute device
            args: command-line arguments used for lane-change penalties and cognitive modules
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.args = args  # Preserve the command-line arguments.

        # === Initialize cognitive modules ===
        self.use_cognitive_modules = args and getattr(args, 'use_cognitive_modules', False)
        self.cognitive_bias_module = None
        self.cognitive_delay_module = None
        self.cognitive_perception_module = None

        if self.use_cognitive_modules:
            print("Initializing cognitive modules...")

            # Initialize the cognitive perception module first because the bias module may reference it.
            if args and getattr(args, 'use_cognitive_perception', False):
                perception_config = {
                    'sigma0': getattr(args, 'perception_sigma0', 0.1),
                    'sigma_max': getattr(args, 'perception_sigma_max', 0.8),
                    'distance_unit': PERCEPTION_SIGMA_UNIT,
                    'far_distance': 150.0,
                    'use_ar1': True,
                    'rho': 0.8,
                    'use_kf': True,
                    'kf_dt': 0.1
                }
                self.cognitive_perception_module = CognitivePerceptionModule(noise_config=perception_config)

                # Enable radar-beam visualization when requested.
                if getattr(args, 'enable_radar_beam_viz', False):
                    self.cognitive_perception_module.enable_radar_visualization(True)

                print(f"   Cognitive perception module enabled")

            # Initialize the cognitive bias module and pass the perception-module reference.
            if args and getattr(args, 'use_cognitive_bias', False):
                bias_config = {
                    'inverse_tta_coef': getattr(args, 'bias_inverse_tta_coef', 1.5),
                    'tta_threshold': getattr(args, 'bias_tta_threshold', 0.1),
                    'visual_detection_distance': getattr(args, 'bias_visual_distance', 50.0),
                    'verbose': True
                }
                # Pass the perception-module reference.
                self.cognitive_bias_module = CognitiveBiasModule(
                    bias_config=bias_config,
                    cognitive_perception_module=self.cognitive_perception_module
                )
                print(f"   Cognitive bias module enabled")
                if self.cognitive_perception_module:
                    print(f"      Connected to the cognitive perception module")

            # Initialize the cognitive delay module
            if args and getattr(args, 'use_cognitive_delay', False):
                self.cognitive_delay_module = CognitiveDelayModule(
                    delay_steps=int(getattr(args, 'delay_steps', 2)),  # Ensure an integer value.
                    enable_smoothing=False,
                    smoothing_factor=0.3,
                    enable_visualization=True
                )
                print(f"   Cognitive delay module enabled (delay={getattr(args, 'delay_steps', 2)} steps)")

        # Load the checkpoint
        self._load_checkpoint()

        # Load the config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # Use the default config
            self.config = self._get_default_config()

        # Track lane-change cooldown state.
        self._last_lane_change_step = {}  # Store the time step of the last lane change.
        self._last_lane_index = {}        # Store the previous lane index for each agent.
        # Derive cooldown steps dynamically from the environment frequency.
        self._lane_change_cooldown_steps = None  # Filled in after the environment is created.

        print(f"PPO checkpoint simulator initialization complete")
        print(f"Checkpoint: {os.path.basename(checkpoint_path)}")
        print(f"Device: {self.device}")
        print(f"Training iteration: {self.checkpoint.get('iteration', 'Unknown')}")
        print(f"Global steps: {self.checkpoint.get('global_step', 'Unknown')}")
        print(f"Cognitive modules: {'enabled' if self.use_cognitive_modules else 'disabled'}")

        # === Collect cognitive-visualization data ===
        self.enable_cognitive_visualization = args and getattr(args, 'enable_cognitive_viz', False)
        if self.enable_cognitive_visualization and self.use_cognitive_modules:
            self.cognitive_viz_data = {
                'timestamps': [],
                'bias_strength': [],
                'bias_applied': [],
                'delay_steps': [],
                'delay_applied': [],
                'perception_noise': [],
                'perception_applied': [],
                'original_rewards': [],
                'modified_rewards': [],
                'original_actions': [],
                'delayed_actions': [],
                'original_observations': [],
                'noisy_observations': [],
                'step_count': []
            }
            print(f"Cognitive visualization: enabled")
        else:
            self.cognitive_viz_data = None

        # === Collect speed-control reward data ===
        self.enable_speed_control_visualization = args and getattr(args, 'use_speed_control_reward', False)
        if self.enable_speed_control_visualization:
            self.speed_control_viz_data = {
                'step_count': [],
                'speed_control_total': [],
                'speed_control_tracking': [],
                'speed_control_soft_wall': [],
                'speed_control_behavior_guidance': [],
                'vehicle_speeds': [],
                'speed_references': [],
                'speed_deviations': []
            }
            print(f"Speed-control visualization: enabled")
        else:
            self.speed_control_viz_data = None

    def _setup_lane_change_cooldown(self, env):
        """Dynamically set the number of lane-change cooldown steps"""
        try:
            if env and hasattr(env, 'config'):
                # Read the physics step size and decision repeat count.
                physics_step_size = env.config.get('physics_world_step_size', 0.02)
                decision_repeat = env.config.get('decision_repeat', 5)

                # Compute the effective control frequency.
                effective_time_step = physics_step_size * decision_repeat
                effective_frequency = 1.0 / effective_time_step

                # Convert the cooldown duration into environment steps.
                lc_cooldown_s = getattr(self, 'args', None) and getattr(self.args, 'lc_cooldown_s', 4.0) or 4.0
                self._lane_change_cooldown_steps = int(lc_cooldown_s * effective_frequency)

                print(f"Lane-change cooldown configuration:")
                print(f"   Physics step: {physics_step_size:.3f}s")
                print(f"   Decision repeat: {decision_repeat}")
                print(f"   Effective frequency: {effective_frequency:.1f}Hz")
                print(f"   Cooldown: {lc_cooldown_s}s → {self._lane_change_cooldown_steps} steps")
            else:
                # Fall back to defaults if the environment config is unavailable.
                lc_cooldown_s = getattr(self, 'args', None) and getattr(self.args, 'lc_cooldown_s', 4.0) or 4.0
                self._lane_change_cooldown_steps = int(lc_cooldown_s * 10)
                print(f" Could not read the environment config; assuming the default 10 Hz control loop")
                print(f"   Cooldown: {lc_cooldown_s}s → {self._lane_change_cooldown_steps} steps")

        except Exception as e:
            # Fall back to defaults when an exception occurs.
            lc_cooldown_s = getattr(self, 'args', None) and getattr(self.args, 'lc_cooldown_s', 4.0) or 4.0
            self._lane_change_cooldown_steps = int(lc_cooldown_s * 10)
            print(f" Failed to configure cooldown: {e}; assuming the default 10 Hz control loop")
            print(f"   Cooldown: {lc_cooldown_s}s → {self._lane_change_cooldown_steps} steps")

    def _load_checkpoint(self):
        """Load checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {self.checkpoint_path}")

        self.checkpoint, self.signature, self.network = load_current_public_network(
            self.checkpoint_path,
            map_location=self.device,
        )
        print(
            "Checkpoint signature: "
            f"observation_dim={self.signature.observation_dim}, "
            f"base={self.signature.base_obs_dim}, "
            f"cognitive_param_dim={self.signature.cognitive_param_dim}, "
            f"cognitive_mask_dim={self.signature.cognitive_mask_dim}, "
            f"cognitive_modulation={self.signature.cognitive_modulation}"
        )

        print(f"Loaded checkpoint: {os.path.basename(self.checkpoint_path)}")

    def _concatenate_cognitive_params(self, obs: np.ndarray) -> np.ndarray:
        """
        Build the raw network input from the checkpoint signature

        Args:
            obs: raw observation [275 dims]

        Returns:
            network input matching the checkpoint signature
        """
        return self._prepare_network_inputs(obs)

    def _prepare_network_inputs(self, obs: np.ndarray):
        obs_array = np.asarray(obs, dtype=np.float32).flatten()
        base_dim = int(getattr(self.network, 'base_obs_dim', 275))
        param_dim = int(getattr(self.network, 'cognitive_param_dim', 0))
        mask_dim = int(getattr(self.network, 'cognitive_mask_dim', 0))
        raw_dim = int(getattr(self.network, 'raw_obs_dim', getattr(self.network, 'obs_dim', obs_array.shape[0])))
        modulation = getattr(self.network, 'cognitive_modulation', 'none')

        if modulation == 'none' or param_dim == 0:
            if obs_array.shape[0] != raw_dim:
                raise ValueError(f"Inference observation dimension does not match the checkpoint signature: expected={raw_dim}, actual={obs_array.shape[0]}")
            return obs_array

        if modulation == 'concat':
            if obs_array.shape[0] not in {base_dim, raw_dim}:
                raise ValueError(f"concatInference observation dimension does not match the checkpoint signature: expected {base_dim} or {raw_dim}, actual={obs_array.shape[0]}")
            base_obs = obs_array[:base_dim]
            params = (
                self._build_cognitive_param_vector(param_dim)
                if self.use_cognitive_modules
                else np.zeros(param_dim, dtype=np.float32)
            )
            mask = np.asarray(
                cognitive_mask_values(
                    effects_enabled=bool(self.use_cognitive_modules),
                    bias_enabled=bool(getattr(self.args, 'use_cognitive_bias', False)) if self.args else False,
                    perception_enabled=bool(getattr(self.args, 'use_cognitive_perception', False)) if self.args else False,
                    delay_enabled=bool(getattr(self.args, 'use_cognitive_delay', False)) if self.args else False,
                ),
                dtype=np.float32,
            )
            if mask.shape[0] != mask_dim:
                raise ValueError(
                    "Checkpoint cognitive-mask dimension does not match the paper input contract: "
                    f"cognitive_mask_dim={mask_dim}, actual={mask.shape[0]}"
                )
            return np.concatenate([base_obs, params, mask], axis=-1).astype(np.float32)

        raise ValueError(f"Unsupported cognitive modulation: {modulation}")

    def _build_cognitive_param_vector(self, dim: int) -> np.ndarray:
        values = [
            getattr(self.args, 'bias_inverse_tta_coef', 1.5) if self.args else 1.5,
            getattr(self.args, 'perception_sigma0', 0.1) if self.args else 0.1,
            getattr(self.args, 'perception_sigma_max', 0.8) if self.args else 0.8,
            float(getattr(self.args, 'delay_steps', 2.0) if self.args else 2.0),
        ]
        if dim > len(values):
            raise ValueError(
                "Checkpoint cognitive-parameter dimension exceeds what the inference script can construct: "
                f"cognitive_param_dim={dim}, supported={len(values)}"
            )
        vector = np.zeros(dim, dtype=np.float32)
        limit = min(dim, len(values))
        vector[:limit] = np.asarray(values[:limit], dtype=np.float32)
        return vector

    def _get_default_config(self):
        """Get the default environment config consistent with training"""
        return {
            # Base environment configuration consistent with training.
            "num_scenarios": 1,
            "traffic_density": 0.08,
            "random_traffic": True,
            "random_agent_model": False,
            "horizon": 1000,
            "map": "SSSSSSS",
            # "map": 301,
            "start_seed": 8888,
                 # Use a dynamic number of straight road segments.



            # Reward configuration exactly matching training.
            "success_reward": 100.0,
            "driving_reward": 0.4,
            "speed_reward": 0,
            "use_lateral_reward": True,

            # Speed-control reward settings are handled inside SpeedControlMetaDriveEnv instead of the base config.

            # Penalty configuration consistent with training.
            "out_of_road_penalty": 8.0,
            "crash_vehicle_penalty": 8.0,
            "crash_object_penalty": 8.0,
            "crash_sidewalk_penalty": 8.0,

            # Termination conditions consistent with training.
            "out_of_road_done": True,
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,

            # Vehicle configuration exactly matching training (ego vehicle).
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 240,
                    "distance": 50,
                    "num_others": 4,
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0
                },
                "side_detector": {
                    "num_lasers": 0,
                    "distance": 50,
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0
                },
                "lane_line_detector": {
                    "num_lasers": 0,
                    "distance": 20,
                    "gaussian_noise": 0.0,
                    "dropout_prob": 0.0
                }
            },

            # Traffic-vehicle-only configuration
            "traffic_vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": False,
                "enable_reverse": False,
                "show_lidar": False,
                "show_lane_line_detector": False,
                "show_side_detector": False,
            },

            # Rendering configuration
            "use_render": True
        }

    def create_environment(self, render: bool = True, scenario_seed: Optional[int] = None) -> MetaDriveEnv:
        """Create a MetaDrive simulation environment"""
        env_config = self.config.copy()
        env_config["use_render"] = render

        # Set the seed so background-vehicle initial state is reproducible.
        if scenario_seed is not None:
            env_config["start_seed"] = scenario_seed
            # Apply stricter randomness control for the first scenario.
            if scenario_seed == 8888:
                env_config["random_traffic"] = False  # Keep the first scenario traffic state fixed.
                print(f"Pinned the first scenario seed: {scenario_seed} (background-vehicle state will remain consistent)")

        # Configure observations carefully to avoid double noise with the cognitive perception module.
        env_config["vehicle_config"]["lidar"] = {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 4,
            # Critical: force these values to 0 to avoid stacking noise on top of the cognitive perception module.
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0
        }

        # Select the environment type from the config.
        use_speed_control = env_config.get("use_speed_control_reward", False)

        # Respect the use_speed_control_reward CLI flag when present.
        if hasattr(self, 'args') and self.args and getattr(self.args, 'use_speed_control_reward', False):
            use_speed_control = True
            print(f"Enabled the speed-control reward from the command-line arguments")

        if use_speed_control and SPEED_CONTROL_AVAILABLE:
            # Use the speed-control environment.
            print("Creating a SpeedControlMetaDriveEnv environment")

            # Create a config that includes speed-control parameters; SpeedControlMetaDriveEnv strips them internally.
            speed_control_config = env_config.copy()

            # Ensure use_speed_control_reward is enabled.
            speed_control_config["use_speed_control_reward"] = True

            # Add speed-control parameters to the config.
            if hasattr(self, 'args') and self.args:
                speed_control_config.update({
                    "speed_control_k": getattr(self.args, 'speed_control_k', 1.0),
                    "speed_control_kappa": getattr(self.args, 'speed_control_kappa', 0.5),
                    "speed_control_mu": getattr(self.args, 'speed_control_mu', 0.3),
                    "speed_control_nu": getattr(self.args, 'speed_control_nu', 0.2),
                    "speed_control_v_tolerance": getattr(self.args, 'speed_control_v_tolerance', 1.0),
                    "speed_control_v_ref": getattr(self.args, 'speed_control_v_ref', 15.0)
                })

            # Disable the original speed reward to avoid double counting.
            speed_control_config["speed_reward"] = 0.0

            print(f"   Configuration update:")
            print(f"      use_speed_control_reward: {speed_control_config['use_speed_control_reward']}")
            print(f"      speed_reward: {speed_control_config['speed_reward']}")
            print(f"      speed_control_k: {speed_control_config.get('speed_control_k', 'N/A')}")
            print(f"      speed_control_v_ref: {speed_control_config.get('speed_control_v_ref', 'N/A')}")

            # Create the SpeedControlMetaDriveEnv instance; it removes custom speed-control keys internally.
            env = SpeedControlMetaDriveEnv(speed_control_config)

            # Apply submodule enable flags after the environment is created.
            if hasattr(self, 'args') and self.args:
                tracking_enabled = getattr(self.args, 'speed_control_enable_tracking', False)
                soft_wall_enabled = getattr(self.args, 'speed_control_enable_soft_wall', False)
                behavior_enabled = getattr(self.args, 'speed_control_enable_behavior_guidance', False)

                # Set the environment attributes directly.
                env.enable_tracking = tracking_enabled
                env.enable_soft_wall = soft_wall_enabled
                env.enable_behavior_guidance = behavior_enabled

                print(f"   Submodule switches:")
                print(f"      enable_tracking: {tracking_enabled}")
                print(f"      enable_soft_wall: {soft_wall_enabled}")
                print(f"      enable_behavior_guidance: {behavior_enabled}")

                # Verify the switches were applied.
                print(f"   Verify submodule switches:")
                print(f"      env.enable_tracking: {getattr(env, 'enable_tracking', 'N/A')}")
                print(f"      env.enable_soft_wall: {getattr(env, 'enable_soft_wall', 'N/A')}")
                print(f"      env.enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', 'N/A')}")
            else:
                print("   Using the default submodule configuration")
                print(f"      enable_tracking: {getattr(env, 'enable_tracking', True)}")
                print(f"      enable_soft_wall: {getattr(env, 'enable_soft_wall', True)}")
                print(f"      enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', True)}")

        else:
            # Use the standard MetaDrive environment.
            if use_speed_control and not SPEED_CONTROL_AVAILABLE:
                print("Speed-control environment unavailable; using the standard MetaDriveEnv")
            else:
                print("Using the standard MetaDriveEnv")

            env = MetaDriveEnv(env_config)

        # Configure lane-change cooldown dynamically.
        if not hasattr(self, '_lane_change_cooldown_steps') or self._lane_change_cooldown_steps is None:
            self._setup_lane_change_cooldown(env)

        return env

    def get_action(self, observation: np.ndarray, deterministic: bool = True, env=None, step_count: int = 0) -> np.ndarray:
        """
        Compute a PPO policy action from an observation

        Args:
            observation: environment observation
            deterministic: whether to use a deterministic policy
            env: environment instance used by cognitive-module processing
            step_count: current step count used for visualization

        Returns:
            action array [steering, acceleration]
        """
        original_obs = observation.copy() if self.enable_cognitive_visualization and self.cognitive_viz_data is not None else None

        # === Cognitive perception: noise is already injected at the sensor layer ===
        # The cognitive perception module already injects noise at the sensor layer via attach_to_env().
        # The observation already includes that noise, so no extra processing is needed here.
        processed_obs = observation
        perception_applied = bool(self.use_cognitive_modules and self.cognitive_perception_module)

        processed_obs = self._prepare_network_inputs(processed_obs)

        # Convert to a tensor.
        obs_tensor = torch.FloatTensor(processed_obs).unsqueeze(0).to(self.device)
        # Query the policy action.
        with torch.no_grad():
            if deterministic and hasattr(self.network, 'act_deterministic'):
                action = self.network.act_deterministic(obs_tensor)
            else:
                action, _, _, _ = self.network.get_action_and_value(obs_tensor)

        # Convert back to NumPy.
        action = action.cpu().numpy().flatten()

        # Clamp the action range.
        action = np.clip(action, -1.0, 1.0)

        original_action = action.copy() if self.enable_cognitive_visualization else None

        # === Cognitive delay: process delayed actions ===
        delay_applied = False
        if self.use_cognitive_modules and self.cognitive_delay_module:
            try:
                action = self.cognitive_delay_module.process_action(action, is_ppo_mode=True)
                delay_applied = True
            except Exception as e:
                print(f"Cognitive delay processing failed: {e}")

        # === Collect visualization data ===
        if self.enable_cognitive_visualization and self.cognitive_viz_data is not None:
            # Perception data: read the live front-beam radar sample.
            front_beam_data = {'original_distance': 0.0, 'noisy_distance': 0.0, 'noise_level': 0.0}

            if self.cognitive_perception_module and perception_applied:
                # Read the live front-beam radar sample.
                front_beam_data = self.cognitive_perception_module.get_front_beam_info()
                noise_level = front_beam_data.get('noise_level', 0.0)
            else:
                noise_level = 0.0

            # Delay-module data
            if self.cognitive_delay_module and delay_applied:
                # Read delay information when available, otherwise use the default value.
                if hasattr(self.cognitive_delay_module, 'get_delay_info'):
                    delay_info = self.cognitive_delay_module.get_delay_info()
                    current_delay = delay_info.get('current_delay', 0)
                else:
                    current_delay = self.cognitive_delay_module.delay_steps
            else:
                current_delay = 0


            # Record visualization data.
            self.cognitive_viz_data['timestamps'].append(time.time())
            self.cognitive_viz_data['step_count'].append(step_count)
            self.cognitive_viz_data['perception_noise'].append(noise_level)
            self.cognitive_viz_data['perception_applied'].append(perception_applied)
            self.cognitive_viz_data['delay_steps'].append(current_delay)
            self.cognitive_viz_data['delay_applied'].append(delay_applied)

            # Record the raw and noisy front-beam distances in meters.
            self.cognitive_viz_data['original_observations'].append(front_beam_data['original_distance'])
            self.cognitive_viz_data['noisy_observations'].append(front_beam_data['noisy_distance'])

            if original_action is not None:
                self.cognitive_viz_data['original_actions'].append(original_action.copy())
                self.cognitive_viz_data['delayed_actions'].append(action.copy())

        return action

    def run_single_episode(self, render: bool = True, max_steps: int = 1000,
                          scenario_seed: Optional[int] = None, deterministic: bool = True) -> Dict:
        """
        Run a single simulation episode

        Args:
            render: whether to render
            max_steps: maximum number of steps
            scenario_seed: scenario seed
            deterministic: whether to use a deterministic policy

        Returns:
            episode statistics
        """
        # Create the environment.
        env = self.create_environment(render=render, scenario_seed=scenario_seed)

        # Reset the environment.
        obs, info = env.reset()

        # === Cognitive modules: reset state and attach to the environment ===
        if self.use_cognitive_modules:
            if self.cognitive_bias_module:
                self.cognitive_bias_module.reset()
            if self.cognitive_delay_module:
                self.cognitive_delay_module.reset()
            if self.cognitive_perception_module:
                self.cognitive_perception_module.reset()
                # Attach the noisy radar to the environment so it replaces the original sensor.
                self.cognitive_perception_module.attach_to_env(env)
                print("Attached the cognitive perception module to the environment; noise will now be injected at the sensor layer")

                # Verify the environment config to avoid double noise.
                lidar_config = env.config.get("vehicle_config", {}).get("lidar", {})
                gaussian_noise = lidar_config.get("gaussian_noise", 0.0)
                dropout_prob = lidar_config.get("dropout_prob", 0.0)

                if gaussian_noise > 0.0 or dropout_prob > 0.0:
                    print(f"Warning: detected extra noise configured in the environment lidar settings!")
                    print(f"   gaussian_noise: {gaussian_noise}")
                    print(f"   dropout_prob: {dropout_prob}")
                    print(f"   This can cause double-noise behavior; set the values to 0.0 to avoid it.")
                else:
                    print(f"Environment lidar noise settings are correct (gaussian_noise=0.0, dropout_prob=0.0)")

            # Attach the cognitive bias module to the environment.
            if self.cognitive_bias_module:
                success = self.cognitive_bias_module.attach_to_env(env)
                if success:
                    print("Attached the cognitive bias module to the environment; rewards will now be adjusted dynamically from TTA")
                else:
                    print("Failed to attach the cognitive bias module")

        # Episode statistics
        episode_stats = {
            "total_reward": 0.0,
            "episode_length": 0,
            "success": False,
            "collision": False,
            "out_of_road": False,
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "path_completion": 0.0,
            "actions": [],
            "speeds": [],
            "rewards": [],
            # Lane-change penalty statistics
            "lane_change_penalties": [],
            "lane_change_speed_ratios": [],
            "cooldown_violations": [],
            "lane_changes": 0,
            # Cognitive-module statistics
            "cognitive_bias_info": [],
            "cognitive_delay_info": [],
            "cognitive_perception_info": []
        }

        speeds = []
        step_count = 0

        try:
            for step in range(max_steps):
                # Query the PPO action.
                action = self.get_action(obs, deterministic=deterministic, env=env, step_count=step_count)
                episode_stats["actions"].append(action.copy())

                # Debug: print the first few actions.
                if step < 5:
                    print(f"   Step {step}: action=[{action[0]:.3f}, {action[1]:.3f}]")

                # Step the environment.
                obs, reward, terminated, truncated, info = env.step(action)

                original_reward = reward if self.enable_cognitive_visualization and self.cognitive_viz_data is not None else None

                # === Cognitive bias: process reward bias ===
                bias_applied = False
                if self.use_cognitive_modules and self.cognitive_bias_module:
                    try:
                        # Call process_reward with the documented arguments.
                        if hasattr(self.cognitive_bias_module, 'process_reward'):
                            reward_result = self.cognitive_bias_module.process_reward(
                                original_reward=reward,
                                env=env,
                                info=info,
                                is_ppo_mode=True
                            )

                            # Handle the documented return shape: (adjusted_reward, bias_info).
                            orig_reward_debug = reward
                            if isinstance(reward_result, (tuple, list)) and len(reward_result) >= 2:
                                adjusted_reward, bias_info = reward_result[0], reward_result[1]
                                reward = float(adjusted_reward)

                                # Record bias information for visualization.
                                if isinstance(bias_info, dict):
                                    bias_amount = bias_info.get('bias_applied', 0.0)
                                    inverse_tta = bias_info.get('inverse_tta', 0.0)
                                    bias_active = bias_info.get('bias_active', False)

                                    if bias_active and abs(bias_amount) > 1e-6:
                                        print(f"Cognitive bias: {orig_reward_debug:.3f} → {reward:.3f} (bias: {bias_amount:+.3f}, TTA⁻¹: {inverse_tta:.3f})")
                                        bias_applied = True
                                    else:
                                        # Ignore negligible or inactive bias adjustments.
                                        bias_applied = False
                                else:
                                    bias_applied = True
                            else:
                                # Compatibility path for a single return value.
                                reward = float(reward_result) if reward_result is not None else reward
                                bias_applied = abs(reward - orig_reward_debug) > 1e-6
                        else:
                            # Apply a simple bias fallback if process_reward is unavailable.
                            orig_reward_debug = reward
                            if reward > 0:
                                reward *= 0.9  # Slightly reduce positive rewards.
                            else:
                                reward *= 1.1  # Amplify negative rewards.



                            bias_applied = True
                    except Exception as e:
                        print(f"Cognitive bias processing failed: {e}")

                # === Collect bias-visualization data ===
                if self.enable_cognitive_visualization and self.cognitive_viz_data is not None:
                    # Bias-module data
                    try:
                        if self.cognitive_bias_module and bias_applied:
                            if hasattr(self.cognitive_bias_module, 'get_bias_info'):
                                bias_info = self.cognitive_bias_module.get_bias_info()
                                # Normalize complex bias_info structures.
                                if isinstance(bias_info, dict):
                                    bias_strength = float(bias_info.get('bias_strength', 0.0))
                                elif isinstance(bias_info, (list, tuple)):
                                    bias_strength = float(bias_info[0]) if len(bias_info) > 0 else 0.0
                                else:
                                    bias_strength = float(bias_info) if bias_info is not None else 0.0
                            else:
                                # Estimate a simple bias strength if get_bias_info is unavailable.
                                bias_strength = abs(reward - original_reward) if original_reward is not None else 0.0
                        else:
                            bias_strength = 0.0
                    except Exception as e:
                        print(f"Bias-data collection failed: {e}")
                        bias_strength = 0.0

                    # Record bias-related data.
                    self.cognitive_viz_data['bias_strength'].append(bias_strength)
                    self.cognitive_viz_data['bias_applied'].append(bias_applied)

                    if original_reward is not None:
                        self.cognitive_viz_data['original_rewards'].append(float(original_reward))
                        self.cognitive_viz_data['modified_rewards'].append(float(reward))

                # Collect speed-control reward data.
                if self.enable_speed_control_visualization and self.speed_control_viz_data is not None:
                    # Record the step count.
                    self.speed_control_viz_data['step_count'].append(step_count)

                    try:
                        # Check the environment type and step_infos attribute.
                        if hasattr(env, 'step_infos') and isinstance(env.step_infos, dict):
                            # Resolve the correct agent ID.
                            agent_id = None
                            if hasattr(env, 'agent') and hasattr(env.agent, 'id'):
                                agent_id = env.agent.id
                            elif hasattr(env, 'agents'):
                                # If env.agent is unavailable, try the first entry in env.agents.
                                agent_ids = list(env.agents.keys())
                                if agent_ids:
                                    agent_id = agent_ids[0]

                            # If agent_id is still missing, fall back to the first key in step_infos.
                            if agent_id is None and env.step_infos:
                                # Use the first agent ID from step_infos.
                                agent_id = list(env.step_infos.keys())[0]
                                print(f"    Resolved agent_id from step_infos: {agent_id}")

                            if agent_id and agent_id in env.step_infos:
                                step_info = env.step_infos[agent_id]

                                # Debug: print step_info contents for the first few steps.
                                if step < 5:
                                    print(f"    Speed-control debug - agent_id: {agent_id}")
                                    print(f"    Speed-control debug - step_info keys: {list(step_info.keys())}")
                                    if 'sc_r_total' in step_info:
                                        print(f"    Found speed-control reward: {step_info['sc_r_total']:.4f}")
                                    else:
                                        print(f"    Speed-control reward key not found")

                                # Collect speed-control reward data.
                                if 'sc_r_total' in step_info:
                                    self.speed_control_viz_data['speed_control_total'].append(float(step_info['sc_r_total']))
                                    self.speed_control_viz_data['speed_control_tracking'].append(float(step_info.get('sc_r_track', 0.0)))
                                    self.speed_control_viz_data['speed_control_soft_wall'].append(float(step_info.get('sc_r_wall', 0.0)))
                                    self.speed_control_viz_data['speed_control_behavior_guidance'].append(float(step_info.get('sc_r_act_over', 0.0)))

                                    # Record speed-related information.
                                    if 'sc_v' in step_info:
                                        self.speed_control_viz_data['vehicle_speeds'].append(float(step_info['sc_v']))
                                    if 'sc_v_ref' in step_info:
                                        self.speed_control_viz_data['speed_references'].append(float(step_info['sc_v_ref']))
                                    if 'sc_dv' in step_info:
                                        self.speed_control_viz_data['speed_deviations'].append(float(step_info['sc_dv']))

                                    # Debug: print successfully collected data.
                                    if step < 5:
                                        print(f"    Collected speed-control data successfully: sc_r_total={step_info['sc_r_total']:.4f}")
                                else:
                                    # If speed-control data is missing, try computing it directly from the environment.
                                    if hasattr(env, '_compute_speed_control_reward') and hasattr(env, 'agent'):
                                        try:
                                            # Call the speed-control reward helper directly.
                                            sc_reward = env._compute_speed_control_reward(env.agent, action)
                                            # Read the latest data back out of step_infos.
                                            if agent_id in env.step_infos:
                                                step_info = env.step_infos[agent_id]
                                                if 'sc_r_total' in step_info:
                                                    self.speed_control_viz_data['speed_control_total'].append(float(step_info['sc_r_total']))
                                                    self.speed_control_viz_data['speed_control_tracking'].append(float(step_info.get('sc_r_track', 0.0)))
                                                    self.speed_control_viz_data['speed_control_soft_wall'].append(float(step_info.get('sc_r_wall', 0.0)))
                                                    self.speed_control_viz_data['speed_control_behavior_guidance'].append(float(step_info.get('sc_r_act_over', 0.0)))

                                                    if 'sc_v' in step_info:
                                                        self.speed_control_viz_data['vehicle_speeds'].append(float(step_info['sc_v']))
                                                    if 'sc_v_ref' in step_info:
                                                        self.speed_control_viz_data['speed_references'].append(float(step_info['sc_v_ref']))
                                                    if 'sc_dv' in step_info:
                                                        self.speed_control_viz_data['speed_deviations'].append(float(step_info['sc_dv']))
                                                else:
                                                    # Fill default values.
                                                    self._fill_default_speed_control_data()
                                            else:
                                                self._fill_default_speed_control_data()
                                        except Exception as e:
                                            print(f"   Direct speed-control reward calculation failed: {e}")
                                            self._fill_default_speed_control_data()
                                    else:
                                        # Fill default values.
                                        self._fill_default_speed_control_data()
                            else:
                                # Debug: print environment information.
                                if step < 5:
                                    print(f"    Speed-control debug - environment info:")
                                    print(f"       hasattr(env, 'step_infos'): {hasattr(env, 'step_infos')}")
                                    print(f"       env.step_infostype: {type(env.step_infos)}")
                                    print(f"       env.step_infoscontents: {env.step_infos}")
                                    if hasattr(env, 'agent') and hasattr(env.agent, 'id'):
                                        print(f"       agent.id: {env.agent.id}")
                                    elif hasattr(env, 'agents'):
                                        print(f"       agent keys: {list(env.agents.keys())}")

                                # Fill default values when step_info is unavailable.
                                self._fill_default_speed_control_data()
                        else:
                            # Debug: print environment type information.
                            if step < 5:
                                print(f"    Speed-control debug - environment type check:")
                                print(f"       environmenttype: {type(env)}")
                                print(f"       Environment class: {env.__class__.__name__}")
                                print(f"       Has step_infos attribute: {hasattr(env, 'step_infos')}")
                                print(f"       step_infostype: {type(getattr(env, 'step_infos', None))}")

                            # Fill default values when the environment does not support speed control.
                            self._fill_default_speed_control_data()
                    except Exception as e:
                        print(f"Speed-control reward data collection failed: {e}")
                        # Fill default values after exceptions.
                        self._fill_default_speed_control_data()

                # Update statistics.
                episode_stats["total_reward"] += reward
                episode_stats["episode_length"] += 1
                episode_stats["rewards"].append(reward)

                # Read speed directly from the agent object.
                if hasattr(env.agent, 'speed'):
                    speed = env.agent.speed
                    speeds.append(speed)
                    episode_stats["speeds"].append(speed)
                    episode_stats["max_speed"] = max(episode_stats["max_speed"], speed)

                    # Debug: print speed information for the first few steps.
                    if step < 5:
                        print(f"    Speed: {speed:.3f} m/s, reward: {reward:.3f}")
                else:
                    if step < 5:
                        print(f"   Unable to read speed information, reward: {reward:.3f}")

                # Lane-change detection and penalty calculation
                lane_change_detected = False

                # Detect lane changes from lane-index transitions.
                if hasattr(env.agent, 'lane_index'):
                    current_lane_index = env.agent.lane_index
                    agent_id = getattr(env.agent, 'id', id(env.agent))

                    # Check whether a previous lane index is already stored.
                    if agent_id in self._last_lane_index:
                        if self._last_lane_index[agent_id] != current_lane_index:
                            lane_change_detected = True
                            episode_stats["lane_changes"] += 1
                    else:
                        # On the first step, store the initial lane index.
                        self._last_lane_index[agent_id] = current_lane_index

                # Compute the lane-change penalty.
                if lane_change_detected:
                    # Read the current speed.
                    current_speed = abs(speed) if 'speed' in locals() else 0.0

                    # Compute the speed ratio using CLI values or defaults.
                    v_limit = getattr(self, 'args', None) and getattr(self.args, 'v_limit', 15.0) or 15.0
                    speed_ratio = current_speed / v_limit
                    speed_ratio = min(speed_ratio, 2.0)  # Cap the maximum ratio.

                    # Base lane-change penalty using CLI values or defaults.
                    w_lc = getattr(self, 'args', None) and getattr(self.args, 'w_lc', 0.6) or 0.6
                    base_penalty = w_lc

                    # High-speed amplification using CLI values or defaults.
                    k_speed = getattr(self, 'args', None) and getattr(self.args, 'k_speed', 1.0) or 1.0
                    speed_penalty = base_penalty * (1 + k_speed * speed_ratio)

                    # Check the cooldown window.
                    current_step = step_count

                    if agent_id in self._last_lane_change_step:
                        steps_since_last_change = current_step - self._last_lane_change_step[agent_id]
                        if steps_since_last_change < self._lane_change_cooldown_steps:
                            # Add an extra penalty during cooldown using CLI values or defaults.
                            w_lc_cool = getattr(self, 'args', None) and getattr(self.args, 'w_lc_cool', 0.6) or 0.6
                            speed_penalty += w_lc_cool
                            episode_stats["cooldown_violations"].append(1)
                        else:
                            episode_stats["cooldown_violations"].append(0)
                    else:
                        episode_stats["cooldown_violations"].append(0)

                    # Update the last lane-change step.
                    self._last_lane_change_step[agent_id] = current_step

                    # Record lane-change penalty statistics.
                    episode_stats["lane_change_penalties"].append(speed_penalty)
                    episode_stats["lane_change_speed_ratios"].append(speed_ratio)

                    # Apply the penalty to the reward.
                    episode_stats["total_reward"] -= speed_penalty

                    # Debug: print lane-change penalty information.
                    if step < 5:
                        print(f"    Lane change detected! Penalty: {speed_penalty:.3f}, speed ratio: {speed_ratio:.3f}")

                # Update the stored lane index regardless of whether a lane change occurred.
                if hasattr(env.agent, 'lane_index'):
                    current_lane_index = env.agent.lane_index
                    agent_id = getattr(env.agent, 'id', id(env.agent))
                    self._last_lane_index[agent_id] = current_lane_index

                # === Cognitive modules: collect statistics ===
                if self.use_cognitive_modules:
                    if self.cognitive_bias_module:
                        try:
                            if hasattr(self.cognitive_bias_module, 'get_bias_info'):
                                bias_info = self.cognitive_bias_module.get_bias_info()
                                # Store a simplified copy and avoid complex objects.
                                if isinstance(bias_info, dict):
                                    simplified_info = {
                                        'bias_strength': float(bias_info.get('bias_strength', 0.0)),
                                        'bias_active': bool(bias_info.get('bias_active', False))
                                    }
                                else:
                                    simplified_info = {'status': 'active', 'value': str(bias_info)[:50]}
                                episode_stats["cognitive_bias_info"].append(simplified_info)
                            else:
                                episode_stats["cognitive_bias_info"].append({'status': 'active'})
                        except Exception as e:
                            print(f"Cognitive bias statistics collection failed: {e}")

                    if self.cognitive_delay_module:
                        try:
                            if hasattr(self.cognitive_delay_module, 'get_delay_info'):
                                delay_info = self.cognitive_delay_module.get_delay_info()
                                if isinstance(delay_info, dict):
                                    simplified_info = {k: v for k, v in delay_info.items() if isinstance(v, (int, float, str, bool))}
                                else:
                                    simplified_info = {'delay_steps': self.cognitive_delay_module.delay_steps}
                                episode_stats["cognitive_delay_info"].append(simplified_info)
                            else:
                                episode_stats["cognitive_delay_info"].append({'delay_steps': self.cognitive_delay_module.delay_steps})
                        except Exception as e:
                            print(f"Cognitive delay statistics collection failed: {e}")

                    if self.cognitive_perception_module:
                        try:
                            if hasattr(self.cognitive_perception_module, 'get_perception_info'):
                                perception_info = self.cognitive_perception_module.get_perception_info()
                                if isinstance(perception_info, dict):
                                    simplified_info = {k: v for k, v in perception_info.items() if isinstance(v, (int, float, str, bool))}
                                else:
                                    simplified_info = {'status': 'active'}
                                episode_stats["cognitive_perception_info"].append(simplified_info)
                            else:
                                episode_stats["cognitive_perception_info"].append({'status': 'active'})
                        except Exception as e:
                            print(f"Cognitive perception statistics collection failed: {e}")

                # Check termination conditions.
                if terminated or truncated:
                    # Reached the destination.
                    if info.get("arrive_dest", False):
                        episode_stats["success"] = True

                    # Collision.
                    if info.get("crash", False) or info.get("crash_vehicle", False):
                        episode_stats["collision"] = True

                    # Off-road event.
                    if info.get("out_of_road", False):
                        episode_stats["out_of_road"] = True

                    break

                step_count += 1

                # Add a short delay when rendering.
                if render:
                    time.sleep(0.05)  # 50msdelay，for easier viewing

        finally:
            # Fetch final path-completion information.
            try:
                # First try the last info dictionary.
                if "route_completion" in info:
                    episode_stats["path_completion"] = info["route_completion"]
                # Then try to read it directly from the navigation module.
                elif hasattr(env.agent, 'navigation') and hasattr(env.agent.navigation, 'route_completion'):
                    episode_stats["path_completion"] = env.agent.navigation.route_completion
                # Finally, try fallback accessors.
                elif hasattr(env.agent, 'navigation'):
                    nav = env.agent.navigation
                    if hasattr(nav, 'get_current_lane_progress'):
                        episode_stats["path_completion"] = nav.get_current_lane_progress()

            except Exception as e:
                print(f" Failed to read path completion: {e}")
                episode_stats["path_completion"] = 0.0

            # Compute the average speed.
            if speeds:
                episode_stats["avg_speed"] = np.mean(speeds)

            # === Generate cognitive visualization ===
            # Define the output directory for cognitive-module visualization.
            str_save_dir = str(f"cognitive_visualization/cognitive_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            if self.enable_cognitive_visualization and self.cognitive_viz_data:
                try:
                    viz_path = self.generate_cognitive_visualization(episode_stats, str_save_dir)
                    episode_stats["cognitive_visualization_path"] = viz_path

                    # Clear data before the next episode.
                    self.clear_cognitive_visualization_data()
                except Exception as e:
                    print(f"Cognitive visualization generation failed: {e}")

            # === Generate speed-control visualization ===
            if self.enable_speed_control_visualization and self.speed_control_viz_data:
                try:
                    viz_path = self.generate_speed_control_visualization(episode_stats)
                    episode_stats["speed_control_visualization_path"] = viz_path

                    # Clear data before the next episode.
                    self.clear_speed_control_visualization_data()
                except Exception as e:
                    print(f"Speed-control visualization generation failed: {e}")

            # === Cognitive modules: detach from the environment ===
            if self.use_cognitive_modules and self.cognitive_perception_module:
                try:
                    # Generate the visualization before closing the environment and before detaching sensors.
                    self.cognitive_perception_module.generate_visualization(save_dir=str_save_dir, env=env)
                    print("Generated cognitive perception visualization")
                except Exception as e:
                    print(f"Cognitive perception visualization generation failed: {e}")
                finally:
                    # Detach the noisy radar and restore the original sensor.
                    self.cognitive_perception_module.detach_from_env()
                    print("Detached the cognitive perception module from the environment")

            # Generate and detach the cognitive bias module visualization.
            if self.cognitive_bias_module:
                try:
                    # Generate cognitive bias visualization.
                    self.cognitive_bias_module.generate_visualization(env=env, save_dir=str_save_dir)
                    print("Generated cognitive bias visualization")
                except Exception as e:
                    print(f"Cognitive bias visualization generation failed: {e}")
                finally:
                    try:
                        self.cognitive_bias_module.detach_from_env()
                        print("Detached the cognitive bias module from the environment")
                    except Exception as e:
                        print(f"Failed to detach the cognitive bias module: {e}")

            env.close()

        return episode_stats

    def run_simulation(self, num_episodes: int = 5, render: bool = True,
                      max_steps: int = 1000, deterministic: bool = True) -> List[Dict]:
        """
        Run multiple simulation episodes

        Args:
            num_episodes: number of episodes
            render: whether to render
            max_steps: maximum steps per episode
            deterministic: whether to use a deterministic policy

        Returns:
            list of per-episode statistics
        """
        print(f"\nStarting PPO simulation ({num_episodes} episodes)")
        print("=" * 60)

        all_stats = []

        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")

            # Pin the first scenario seed so the background traffic starts reproducibly; later scenarios may differ.
            if episode == 0:
                scenario_seed = 8888  # Always use a fixed seed for the first scenario.
            else:
                scenario_seed = 8888 + episode  # Use different seeds for later scenarios.

            # Run the episode.
            stats = self.run_single_episode(
                render=render,
                max_steps=max_steps,
                scenario_seed=scenario_seed,
                deterministic=deterministic
            )

            all_stats.append(stats)

            # Print episode results.
            print(f"Episode {episode + 1} results:")
            print(f"   Total reward: {stats['total_reward']:.2f}")
            print(f"   Episode length: {stats['episode_length']}")
            print(f"   Reached destination: {'yes' if stats['success'] else 'no'}")
            print(f"   Collision: {'yes' if stats['collision'] else 'no'}")
            print(f"   Off-road: {'yes' if stats['out_of_road'] else 'no'}")
            print(f"   Maximum speed: {stats['max_speed']:.2f} m/s")
            print(f"   Average speed: {stats['avg_speed']:.2f} m/s")
            print(f"   Path completion: {stats['path_completion']:.1%}")

            # Lane-change penalty summary
            if stats['lane_changes'] > 0:
                avg_penalty = np.mean(stats['lane_change_penalties']) if stats['lane_change_penalties'] else 0
                avg_speed_ratio = np.mean(stats['lane_change_speed_ratios']) if stats['lane_change_speed_ratios'] else 0
                cooldown_violations = np.sum(stats['cooldown_violations']) if stats['cooldown_violations'] else 0
                print(f"   Lane changes: {stats['lane_changes']}")
                print(f"   Average lane-change penalty: {avg_penalty:.3f}")
                print(f"   Average lane-change speed ratio: {avg_speed_ratio:.3f}")
                print(f"   Cooldown violations: {cooldown_violations}")
            else:
                print(f"   Lane changes: 0")

            # Cognitive visualization output
            if 'cognitive_visualization_path' in stats and stats['cognitive_visualization_path']:
                print(f"   Cognitive visualization: {stats['cognitive_visualization_path']}")

            # Speed-control visualization output
            if 'speed_control_visualization_path' in stats and stats['speed_control_visualization_path']:
                print(f"   Speed-control visualization: {stats['speed_control_visualization_path']}")

        # Print aggregate statistics.
        self._print_summary_stats(all_stats)

        return all_stats

    def _print_summary_stats(self, all_stats: List[Dict]):
        """Print aggregate statistics"""
        if not all_stats:
            return

        print(f"\nAggregate statistics ({len(all_stats)} episodes)")
        print("=" * 60)

        # Compute aggregate metrics.
        success_rate = np.mean([s['success'] for s in all_stats])
        collision_rate = np.mean([s['collision'] for s in all_stats])
        out_of_road_rate = np.mean([s['out_of_road'] for s in all_stats])

        avg_reward = np.mean([s['total_reward'] for s in all_stats])
        avg_length = np.mean([s['episode_length'] for s in all_stats])
        avg_speed = np.mean([s['avg_speed'] for s in all_stats])
        avg_completion = np.mean([s['path_completion'] for s in all_stats])

        # Compute lane-change metrics.
        total_lane_changes = np.sum([s['lane_changes'] for s in all_stats])
        avg_lane_change_penalty = np.mean([np.mean(s['lane_change_penalties']) if s['lane_change_penalties'] else 0 for s in all_stats])
        avg_lane_change_speed_ratio = np.mean([np.mean(s['lane_change_speed_ratios']) if s['lane_change_speed_ratios'] else 0 for s in all_stats])
        total_cooldown_violations = np.sum([np.sum(s['cooldown_violations']) if s['cooldown_violations'] else 0 for s in all_stats])

        print(f"Success rate: {success_rate:.1%}")
        print(f"Collision rate: {collision_rate:.1%}")
        print(f" Off-road rate: {out_of_road_rate:.1%}")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Average episode length: {avg_length:.1f}")
        print(f"Average speed: {avg_speed:.2f} m/s")
        print(f"Average path completion: {avg_completion:.1%}")

        # Lane-change metrics
        print(f"Total lane changes: {total_lane_changes}")
        print(f"Average lane-change penalty: {avg_lane_change_penalty:.3f}")
        print(f"Average lane-change speed ratio: {avg_lane_change_speed_ratio:.3f}")
        print(f"Total cooldown violations: {total_cooldown_violations}")

        # Cognitive-module statistics output
        if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
            print(f"\nCognitive-module statistics:")

            if self.cognitive_bias_module:
                try:
                    if hasattr(self.cognitive_bias_module, 'get_statistics'):
                        bias_stats = self.cognitive_bias_module.get_statistics()
                        print(f"   Cognitive bias:")
                        print(f"      Average bias strength: {bias_stats.get('average_bias', 0.0):.3f}")
                        print(f"      Bias activation count: {bias_stats.get('active_steps', 0)}")
                    else:
                        print(f"   Cognitive bias: running in basic mode")
                except Exception as e:
                    print(f"  Failed to read cognitive bias statistics: {e}")

            if self.cognitive_delay_module:
                try:
                    # The delay module has no get_statistics method, so show basic information instead.
                    delay_steps = getattr(self.cognitive_delay_module, 'delay_steps', 0)
                    print(f"   Cognitive delay:")
                    print(f"      Configured delay steps: {delay_steps}")
                    print(f"      Status: active")
                except Exception as e:
                    print(f"  Failed to read cognitive delay statistics: {e}")

            if self.cognitive_perception_module:
                try:
                    # The perception module has no get_statistics method, so show config information instead.
                    noise_config = getattr(self.cognitive_perception_module, 'noise_config', {})
                    sigma0_value = noise_config.get('sigma0', 0.01)
                    # Ensure the retrieved value is numeric.
                    if isinstance(sigma0_value, (int, float)):
                        base_noise = float(sigma0_value)
                    else:
                        base_noise = 0.01  # Default value
                    print(f"  Cognitive perception:")
                    print(f"      Base noise level: {base_noise:.3f}")
                    print(f"      Status: active")
                except Exception as e:
                    print(f"  Failed to read cognitive perception statistics: {e}")

    def evaluate_model(self, num_episodes: int = 20, render: bool = False) -> Dict:
        """
        Evaluate model performance

        Args:
            num_episodes: number of evaluation episodes
            render: whether to render

        Returns:
            evaluation result dictionary
        """
        print(f"\nModel performance evaluation ({num_episodes} episodes)")
        print("=" * 60)

        all_stats = self.run_simulation(
            num_episodes=num_episodes,
            render=render,
            max_steps=1000,
            deterministic=True
        )

        # Compute detailed evaluation metrics.
        evaluation = {
            "num_episodes": len(all_stats),
            "success_rate": np.mean([s['success'] for s in all_stats]),
            "collision_rate": np.mean([s['collision'] for s in all_stats]),
            "out_of_road_rate": np.mean([s['out_of_road'] for s in all_stats]),
            "avg_reward": np.mean([s['total_reward'] for s in all_stats]),
            "std_reward": np.std([s['total_reward'] for s in all_stats]),
            "avg_episode_length": np.mean([s['episode_length'] for s in all_stats]),
            "avg_speed": np.mean([s['avg_speed'] for s in all_stats]),
            "avg_path_completion": np.mean([s['path_completion'] for s in all_stats]),
            # Lane-change evaluation metrics
            "total_lane_changes": np.sum([s['lane_changes'] for s in all_stats]),
            "avg_lane_change_penalty": np.mean([np.mean(s['lane_change_penalties']) if s['lane_change_penalties'] else 0 for s in all_stats]),
            "avg_lane_change_speed_ratio": np.mean([np.mean(s['lane_change_speed_ratios']) if s['lane_change_speed_ratios'] else 0 for s in all_stats]),
            "total_cooldown_violations": np.sum([np.sum(s['cooldown_violations']) if s['cooldown_violations'] else 0 for s in all_stats]),
            "checkpoint_path": self.checkpoint_path,
            "training_iteration": self.checkpoint.get('iteration', 'Unknown'),
            "training_global_step": self.checkpoint.get('global_step', 'Unknown')
        }

        # Cognitive-module evaluation metrics
        if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
            evaluation["cognitive_modules"] = {}

            if self.cognitive_bias_module:
                try:
                    bias_stats = self.cognitive_bias_module.get_statistics()
                    evaluation["cognitive_modules"]["bias"] = bias_stats
                except Exception as e:
                    print(f"Failed to read cognitive bias evaluation statistics: {e}")

            if self.cognitive_delay_module:
                try:
                    delay_stats = self.cognitive_delay_module.get_statistics()
                    evaluation["cognitive_modules"]["delay"] = delay_stats
                except Exception as e:
                    print(f"Failed to read cognitive delay evaluation statistics: {e}")

            if self.cognitive_perception_module:
                try:
                    perception_stats = self.cognitive_perception_module.get_statistics()
                    evaluation["cognitive_modules"]["perception"] = perception_stats
                except Exception as e:
                    print(f"Failed to read cognitive perception evaluation statistics: {e}")

        return evaluation
from common.cognitive_input import cognitive_mask_values
