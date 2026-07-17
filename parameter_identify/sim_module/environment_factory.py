"""
Environment factory helpers.

This module handles MetaDrive environment creation, configuration, and
management. It extracts environment-construction logic from the original
``PPOCheckpointSimulator`` implementation.
"""


import argparse
from typing import Dict, Optional
from pathlib import Path
import sys


# Import environment classes.
from metadrive.envs.metadrive_env import MetaDriveEnv

# Import the speed-control environment if available.
try:
    from ppo_train.envs.speed_control_env import SpeedControlMetaDriveEnv
    SPEED_CONTROL_AVAILABLE = True
    print("SpeedControlMetaDriveEnv imported successfully")
except ImportError:
    SPEED_CONTROL_AVAILABLE = False
    print("Failed to import SpeedControlMetaDriveEnv; falling back to standard MetaDriveEnv")


class EnvironmentFactory:
    """MetaDrive environment factory.

    Responsibilities:
    1. Manage and build environment configs.
    2. Create standard ``MetaDriveEnv`` and ``SpeedControlMetaDriveEnv`` instances.
    3. Apply dynamic environment parameter overrides.
    4. Configure speed-control submodules.
    """

    def __init__(self, config: Optional[Dict] = None, args: Optional[argparse.Namespace] = None):
        """
        Initialize the environment factory.

        Args:
            config: Environment configuration dictionary.
            args: Command-line arguments.
        """
        self.config = config if config is not None else self._get_default_config()
        self.args = args

        print("Environment factory initialized")
        print(f"   Base map: {self.config.get('map', 'SSSSSSS')}")
        print(f"   Traffic density: {self.config.get('traffic_density', 0.08)}")
        print(f"   Max horizon: {self.config.get('horizon', 1000)}")

    def create_environment(self, render: bool = True, scenario_seed: Optional[int] = None) -> MetaDriveEnv:
        """
        Create a MetaDrive simulation environment.

        Args:
            render: Whether rendering is enabled.
            scenario_seed: Optional scenario seed.

        Returns:
            A configured ``MetaDriveEnv`` instance.
        """
        env_config = self.config.copy()
        env_config["use_render"] = render

        # Set the seed so background-vehicle initial states are reproducible.
        if scenario_seed is not None:
            env_config["start_seed"] = scenario_seed
            if scenario_seed == 8888:
                env_config["random_traffic"] = True
                print(f"Using fixed first-scenario seed: {scenario_seed} (background-vehicle state will stay reproducible)")

        # Configure observations to avoid double-applying perception noise.
        env_config["vehicle_config"]["lidar"] = {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 4,
            # These must stay at zero to avoid stacking with cognitive-perception noise.
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0
        }

        # Configure the custom IDM policy when traffic-speed settings are present.
        self._configure_custom_idm_policy(env_config)

        # Choose the environment type from config.
        use_speed_control = self._should_use_speed_control(env_config)

        if use_speed_control and SPEED_CONTROL_AVAILABLE:
            env = self._create_speed_control_env(env_config)
        else:
            env = self._create_standard_env(env_config, use_speed_control)

        return env

    def _should_use_speed_control(self, env_config: Dict) -> bool:
        """Return whether the speed-control environment should be used."""
        use_speed_control = env_config.get("use_speed_control_reward", False)

        if hasattr(self, 'args') and self.args and getattr(self.args, 'use_speed_control_reward', False):
            use_speed_control = True
            print("Enabled speed-control reward from command-line arguments")

        return use_speed_control

    def _create_speed_control_env(self, env_config: Dict) -> 'SpeedControlMetaDriveEnv':
        """Create a speed-control environment."""
        print("Creating SpeedControlMetaDriveEnv")

        # Build a config that includes speed-control parameters.
        speed_control_config = env_config.copy()

        # Ensure the speed-control reward path is enabled.
        speed_control_config["use_speed_control_reward"] = True

        # Inject speed-control parameters into the config.
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

        print("   Config updates:")
        print(f"      use_speed_control_reward: {speed_control_config['use_speed_control_reward']}")
        print(f"      speed_reward: {speed_control_config['speed_reward']}")
        print(f"      speed_control_k: {speed_control_config.get('speed_control_k', 'N/A')}")
        print(f"      speed_control_v_ref: {speed_control_config.get('speed_control_v_ref', 'N/A')}")

        # Create the environment instance. It cleans up speed-control-only keys itself.
        env = SpeedControlMetaDriveEnv(speed_control_config)

        # Configure submodule toggles after environment construction.
        self._configure_speed_control_submodules(env)

        return env

    def _create_standard_env(self, env_config: Dict, use_speed_control: bool) -> MetaDriveEnv:
        """Create a standard MetaDrive environment."""
        if use_speed_control and not SPEED_CONTROL_AVAILABLE:
            print("Speed-control environment is unavailable; using standard MetaDriveEnv")
        else:
            print("Using standard MetaDriveEnv")

        env = MetaDriveEnv(env_config)
        return env

    def _configure_speed_control_submodules(self, env):
        """Configure speed-control submodules."""
        if not hasattr(self, 'args') or not self.args:
            print("   Using default submodule configuration")
            print(f"      enable_tracking: {getattr(env, 'enable_tracking', True)}")
            print(f"      enable_soft_wall: {getattr(env, 'enable_soft_wall', True)}")
            print(f"      enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', True)}")
            return

        tracking_enabled = getattr(self.args, 'speed_control_enable_tracking', False)
        soft_wall_enabled = getattr(self.args, 'speed_control_enable_soft_wall', False)
        behavior_enabled = getattr(self.args, 'speed_control_enable_behavior_guidance', False)

        # Set environment attributes directly.
        env.enable_tracking = tracking_enabled
        env.enable_soft_wall = soft_wall_enabled
        env.enable_behavior_guidance = behavior_enabled

        print("   Submodule toggle configuration:")
        print(f"      enable_tracking: {tracking_enabled}")
        print(f"      enable_soft_wall: {soft_wall_enabled}")
        print(f"      enable_behavior_guidance: {behavior_enabled}")

        print("   Verified submodule toggles:")
        print(f"      env.enable_tracking: {getattr(env, 'enable_tracking', 'N/A')}")
        print(f"      env.enable_soft_wall: {getattr(env, 'enable_soft_wall', 'N/A')}")
        print(f"      env.enable_behavior_guidance: {getattr(env, 'enable_behavior_guidance', 'N/A')}")

    def _get_default_config(self) -> Dict:
        """Return the default environment configuration used by training."""
        config = {
            # Base environment config, kept aligned with training.
            "num_scenarios": 1,
            "traffic_density": 0.08,
            "random_traffic": True,
            "traffic_mode": "basic",
            # "random_agent_model": False,
            "horizon": 1000,
            "map": "SSSSSSS",
            "start_seed": 8888,

            # Reward config, kept aligned with training.
            "success_reward": 20.0,
            "driving_reward": 1.25,
            "speed_reward": 0,
            "use_lateral_reward": True,

            # Penalty config, kept aligned with training.
            "out_of_road_penalty": 8.0,
            "crash_vehicle_penalty": 8.0,
            "crash_object_penalty": 8.0,
            "crash_sidewalk_penalty": 2.0,

            # Termination conditions, kept aligned with training.
            "out_of_road_done": True,
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "on_continuous_line_done": False,
            "on_broken_line_done": False,

            # Vehicle config, kept aligned with training (ego vehicle).
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

            # Background-traffic-only config.
            "traffic_vehicle_config": {
                "show_navi_mark": False,
                "show_dest_mark": False,
                "enable_reverse": False,
                "show_lidar": False,
                "show_lane_line_detector": False,
                "show_side_detector": False,
            },

            # Rendering config.
            "use_render": True,

            # Camera and rendering parameters.
            "camera_height": 80,
            "camera_dist": 15,  # Camera distance.
            "camera_pitch": -15,
            "camera_smooth": True,
            "window_size": (1200, 900)
        }

        # Add background-traffic speed configuration.
        self._add_traffic_speed_config(config)

        return config

    def _add_traffic_speed_config(self, config: Dict):
        """Add background-traffic speed-control settings."""
        if not hasattr(self, 'args') or not self.args:
            return

        # Read background-traffic speed settings.
        traffic_normal_speed = getattr(self.args, 'traffic_normal_speed', 80.0)
        traffic_max_speed = getattr(self.args, 'traffic_max_speed', 100.0)
        traffic_target_speed_min = getattr(self.args, 'traffic_target_speed_min', 60.0)
        traffic_target_speed_max = getattr(self.args, 'traffic_target_speed_max', 80.0)
        traffic_spawn_with_speed = getattr(self.args, 'traffic_spawn_with_speed', False)

        # Save the speed config into the environment config.
        config["traffic_speed_config"] = {
            "normal_speed": traffic_normal_speed,
            "max_speed": traffic_max_speed,
            "target_speed_range": [traffic_target_speed_min, traffic_target_speed_max],
            "spawn_with_speed": traffic_spawn_with_speed
        }

        print("Background-traffic speed configuration:")
        print(f"   Normal speed: {traffic_normal_speed} km/h")
        print(f"   Max speed: {traffic_max_speed} km/h")
        print(f"   Target speed range: [{traffic_target_speed_min}, {traffic_target_speed_max}] km/h")
        print(f"   Assign speed at spawn: {'yes' if traffic_spawn_with_speed else 'no'}")

    def _configure_custom_idm_policy(self, env_config: Dict):
        """Configure the custom IDM policy."""
        if "traffic_speed_config" not in env_config:
            return

        try:
            from .custom_idm_policy import ConfigurableIDMPolicy

            if "traffic_vehicle_config" not in env_config:
                env_config["traffic_vehicle_config"] = {}

            env_config["traffic_vehicle_config"]["policy"] = ConfigurableIDMPolicy

            env_config["agent_policy"] = ConfigurableIDMPolicy

            print("Configured the custom IDM policy for background-vehicle speed control")
            print(f"   Policy class: {ConfigurableIDMPolicy.__name__}")

        except ImportError as e:
            print(f"Could not import the custom IDM policy; using the default policy: {e}")
        except Exception as e:
            print(f"Failed to configure the custom IDM policy; using the default policy: {e}")

    def update_config(self, new_config: Dict):
        """Update the environment configuration."""
        self.config.update(new_config)
        print(f"Environment config updated: {list(new_config.keys())}")

    def get_config(self) -> Dict:
        """Return the current environment configuration."""
        return self.config.copy()
