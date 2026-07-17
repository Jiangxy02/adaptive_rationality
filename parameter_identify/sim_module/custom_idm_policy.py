"""
Custom IDM policy helpers.

This module provides configurable IDM policies whose background-vehicle speed
parameters can be adjusted dynamically from the environment configuration.
Unlike the standard ``IDMPolicy``, these policies read speed parameters from
runtime config.
"""


import numpy as np
from metadrive.policy.idm_policy import IDMPolicy


class ConfigurableIDMPolicy(IDMPolicy):
    """
    Configurable IDM policy.

    This policy inherits from the standard ``IDMPolicy`` but allows speed
    parameters to be set dynamically from environment config:
    - ``normal_speed``: nominal speed in km/h
    - ``max_speed``: max speed in km/h
    - ``target_speed_range``: target-speed range ``[min, max]`` in km/h
    - ``spawn_with_speed``: whether to assign speed immediately on spawn
    """

    def __init__(self, control_object, random_seed):
        # Read environment config before parent initialization.
        self._load_speed_config_from_env(control_object)

        # Initialize the parent policy.
        super(ConfigurableIDMPolicy, self).__init__(control_object, random_seed)

        # Sample the target speed from config.
        self._set_target_speed_from_config()

        # Track whether spawn velocity still needs to be applied.
        self._need_set_spawn_velocity = getattr(self, '_spawn_with_speed', False)
        self._spawn_velocity_set = False

    def _load_speed_config_from_env(self, control_object):
        """Load speed parameters from environment config."""
        try:
            # Read environment config.
            env_config = control_object.engine.global_config
            speed_config = env_config.get("traffic_speed_config", {})

            # Pull speed-related config values.
            self._normal_speed = speed_config.get("normal_speed", 80.0)
            self._max_speed = speed_config.get("max_speed", 100.0)
            self._target_speed_range = speed_config.get("target_speed_range", [60.0, 80.0])
            self._spawn_with_speed = speed_config.get("spawn_with_speed", False)

            # Update class-level defaults used by IDM internals.
            self.NORMAL_SPEED = self._normal_speed
            self.MAX_SPEED = self._max_speed

        except Exception as e:
            print(f"Could not load background-vehicle speed config; using defaults: {e}")
            self._normal_speed = 80.0
            self._max_speed = 100.0
            self._target_speed_range = [60.0, 80.0]
            self._spawn_with_speed = False

    def _set_target_speed_from_config(self):
        """Sample a target speed from the configured range."""
        min_speed, max_speed = self._target_speed_range
        self.target_speed = self.np_random.uniform(min_speed, max_speed)

    def _set_spawn_velocity(self):
        """Assign the initial vehicle speed at spawn time."""
        try:
            # Convert the target speed from km/h to m/s.
            target_speed_ms = self.target_speed / 3.6

            print(f"Setting initial background-vehicle speed: {self.target_speed:.1f} km/h ({target_speed_ms:.1f} m/s)")

            if hasattr(self.control_object, 'set_velocity'):
                # Set initial speed along the heading direction.
                direction = [np.cos(self.control_object.heading_theta),
                            np.sin(self.control_object.heading_theta)]
                self.control_object.set_velocity(direction, target_speed_ms)
            elif hasattr(self.control_object, 'velocity'):
                # Set the velocity vector directly.
                self.control_object.velocity = np.array([
                    target_speed_ms * np.cos(self.control_object.heading_theta),
                    target_speed_ms * np.sin(self.control_object.heading_theta)
                ])
            else:
                print("Could not set the initial vehicle speed because the vehicle object has no velocity interface")

        except Exception as e:
            print(f"Failed to set initial background-vehicle speed: {e}")

    def act(self, *args, **kwargs):
        """Override ``act`` to apply spawn velocity on first use."""
        if self._need_set_spawn_velocity and not self._spawn_velocity_set:
            self._set_spawn_velocity()
            self._spawn_velocity_set = True

        return super().act(*args, **kwargs)

    def reset(self):
        """Reset the policy state."""
        super().reset()
        self._set_target_speed_from_config()
        self._spawn_velocity_set = False


class ConfigurableTrajectoryIDMPolicy(ConfigurableIDMPolicy):
    """
    Configurable trajectory IDM policy.

    This policy is used for background vehicles in trajectory-replay
    environments. It keeps trajectory-following behavior while honoring the
    configurable speed settings.
    """

    def __init__(self, control_object, random_seed, traj_to_follow, policy_index=None):
        # Load speed config first.
        self._load_speed_config_from_env(control_object)

        # Initialize IDMPolicy directly and skip ConfigurableIDMPolicy.__init__.
        IDMPolicy.__init__(self, control_object, random_seed)

        # Set trajectory-specific attributes.
        self.policy_index = policy_index
        from metadrive.component.lane.point_lane import PointLane
        assert isinstance(traj_to_follow, PointLane), "Trajectory of IDM policy should be in PointLane Class"
        self.traj_to_follow = traj_to_follow
        self.routing_target_lane = self.traj_to_follow
        self.destination = np.asarray(self.traj_to_follow.end)
        self.available_routing_index_range = None
        self.overtake_timer = self.np_random.randint(0, self.LANE_CHANGE_FREQ)
        self.enable_lane_change = False

        # Reinitialize PID controllers.
        from metadrive.utils.pid_controller import PIDController
        self.heading_pid = PIDController(1.2, 0.1, 3.5)
        self.lateral_pid = PIDController(0.3, .0, 0.0)

        self.last_action = [0, 0]

        # Sample the target speed from config.
        self._set_target_speed_from_config()

        # Apply speed immediately when configured to do so.
        if getattr(self, '_spawn_with_speed', False):
            self._set_spawn_velocity()


def create_traffic_policy_class(use_configurable_idm=True):
    """
    Factory for background-traffic policy classes.

    Args:
        use_configurable_idm: Whether to use the configurable IDM policy.

    Returns:
        The policy class.
    """
    if use_configurable_idm:
        return ConfigurableIDMPolicy
    else:
        return IDMPolicy


def create_trajectory_policy_class(use_configurable_idm=True):
    """
    Factory for trajectory-replay traffic policy classes.

    Args:
        use_configurable_idm: Whether to use the configurable IDM policy.

    Returns:
        The trajectory policy class.
    """
    if use_configurable_idm:
        return ConfigurableTrajectoryIDMPolicy
    else:
        from metadrive.policy.idm_policy import TrajectoryIDMPolicy
        return TrajectoryIDMPolicy
