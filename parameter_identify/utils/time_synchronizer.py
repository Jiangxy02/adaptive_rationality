
import numpy as np

class TimeSynchronizer:
    """
    Time synchronization manager.

    Responsibilities:
    - Keep simulation time aligned with trajectory time.
    - Interpolate and match trajectory timestamps.
    - Provide time-based trajectory state lookup.
    - Detect when a trajectory has finished.
    """

    def __init__(self, env):
        """
        Initialize the time synchronizer.

        Args:
            env: TrajectoryReplayEnv instance.
        """
        self.env = env
        self._trajectory_start_time = None  # Start timestamp of the trajectory data.

    def initialize_trajectory_start_time(self, custom_start_timestamp=None):
        """
        Initialize the trajectory start timestamp for time alignment.

        Args:
            custom_start_timestamp: Optional custom start timestamp.
        """
        if custom_start_timestamp is not None:
            # Use the caller-specified start timestamp.
            self._trajectory_start_time = custom_start_timestamp
            print(f"Using custom trajectory start time: {self._trajectory_start_time:.3f} seconds")
            return

        if not self.env.main_vehicle_trajectory:
            self._trajectory_start_time = 0.0
            return

        # Use the first ego-trajectory timestamp as the start time.
        if "timestamp" in self.env.main_vehicle_trajectory[0]:
            self._trajectory_start_time = self.env.main_vehicle_trajectory[0]["timestamp"]
        else:
            self._trajectory_start_time = 0.0

        print(f"Trajectory start time: {self._trajectory_start_time:.3f} seconds")

    def setup_time_synchronization(self):
        """
        Configure time synchronization and inspect trajectory timing.
        """
        if not self.env.main_vehicle_trajectory:
            print("Warning: No main vehicle trajectory for time synchronization")
            return

        # Check whether timestamps are available in the trajectory data.
        if len(self.env.main_vehicle_trajectory) > 1:
            first_point = self.env.main_vehicle_trajectory[0]
            second_point = self.env.main_vehicle_trajectory[1]

            if "timestamp" in first_point and "timestamp" in second_point:
                csv_dt = second_point["timestamp"] - first_point["timestamp"]
                print(f"\nTime Synchronization Analysis:")
                print(f"  CSV interpolated step size: {csv_dt:.6f} seconds ({1/csv_dt:.1f} Hz)")
                print(f"  MetaDrive physics step: {self.env.physics_world_step_size:.6f} seconds ({1/self.env.physics_world_step_size:.1f} Hz)")

                # Compute the time-step ratio.
                step_ratio = csv_dt / self.env.physics_world_step_size
                print(f"  Time step ratio (CSV/Physics): {step_ratio:.2f}")

                if abs(step_ratio - 1.0) > 0.1:  # Difference exceeds 10%.
                    print(f"  Warning: Significant time step mismatch!")
                    print(f"     This may cause speed inconsistencies.")
                    print(f"     Consider adjusting CSV interpolation or physics step size.")
                else:
                    print(f"  Time steps are well synchronized")

                self.csv_dt = csv_dt
            else:
                print("Warning: Trajectory data missing timestamp information")
                self.csv_dt = 0.05  # Default to 50 ms.
        else:
            print("Warning: Insufficient trajectory data for time analysis")
            self.csv_dt = 0.05

    def get_trajectory_state_at_time(self, trajectory, sim_time):
        """
        Get the trajectory state that corresponds to the simulation time.

        Args:
            trajectory: Vehicle trajectory data.
            sim_time: Current simulation time in seconds since reset.

        Returns:
            Dict: The matching vehicle state, or None if out of range.
        """
        if not trajectory:
            return None

        # Compute the target timestamp.
        target_time = self._trajectory_start_time + sim_time

        # Prefer exact matching against original CSV timestamps when available.
        if "original_timestamp" in trajectory[0] and trajectory[0]["original_timestamp"] != 0:
            return self._find_closest_original_timestamp(trajectory, target_time)
        # Otherwise interpolate using the resampled timestamp field.
        elif "timestamp" in trajectory[0]:
            return self._interpolate_trajectory_by_time(trajectory, target_time)
        else:
            # Fallback to step-index lookup when timestamps are unavailable.
            step_index = int(sim_time / self.env.physics_world_step_size)
            if 0 <= step_index < len(trajectory):
                return trajectory[step_index]
            else:
                return None

    def _find_closest_original_timestamp(self, trajectory, target_time):
        """
        Find the closest point using original CSV timestamps.

        Args:
            trajectory: Trajectory data with original_timestamp fields.
            target_time: Target timestamp.

        Returns:
            Dict: The nearest trajectory state.
        """
        if not trajectory:
            return None

        # Extract the original timestamps.
        original_timestamps = [point["original_timestamp"] for point in trajectory]

        # Clamp to the first or last point when outside the trajectory range.
        if target_time < original_timestamps[0]:
            return self._validate_trajectory_state(trajectory[0])
        elif target_time > original_timestamps[-1]:
            return self._validate_trajectory_state(trajectory[-1])

        # Find the closest timestamp.
        try:
            closest_idx = min(range(len(original_timestamps)),
                             key=lambda i: abs(original_timestamps[i] - target_time))
        except Exception as e:
            print(f"Timestamp lookup failed: {e}; falling back to the first point")
            return self._validate_trajectory_state(trajectory[0])

        closest_point = trajectory[closest_idx].copy()

        # Record the absolute time-matching error.
        time_error = abs(original_timestamps[closest_idx] - target_time)
        closest_point["current_time_error"] = time_error

        return self._validate_trajectory_state(closest_point)

    def _validate_trajectory_state(self, state):
        """
        Validate and sanitize trajectory state values.

        Args:
            state: Trajectory state dictionary.

        Returns:
            Dict: Validated trajectory state.
        """
        if not state:
            return None

        validated_state = state.copy()

        # Clamp speeds to a physically reasonable range.
        MAX_SPEED = 50.0  # Maximum 50 m/s (about 180 km/h).
        MIN_SPEED = 0.0   # Minimum 0 m/s.

        # Validate the scalar speed value.
        if "speed" in validated_state:
            original_speed = validated_state["speed"]
            if original_speed > MAX_SPEED:
                validated_state["speed"] = MAX_SPEED
                print(f"Speed corrected: {original_speed:.1f} -> {MAX_SPEED} m/s")
            elif original_speed < MIN_SPEED:
                validated_state["speed"] = MIN_SPEED

        # Validate speed components.
        if "speed_x" in validated_state and "speed_y" in validated_state:
            speed_x = validated_state["speed_x"]
            speed_y = validated_state["speed_y"]
            speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)

            if speed_magnitude > MAX_SPEED:
                # Keep direction while limiting magnitude.
                scale_factor = MAX_SPEED / speed_magnitude
                validated_state["speed_x"] = speed_x * scale_factor
                validated_state["speed_y"] = speed_y * scale_factor
                print(f"Speed components corrected: magnitude {speed_magnitude:.1f} -> {MAX_SPEED} m/s")

        # Validate positions to avoid unreasonable coordinates.
        MAX_COORD = 50000.0  # Maximum coordinate magnitude: 50 km.
        for coord in ["x", "y"]:
            if coord in validated_state:
                if abs(validated_state[coord]) > MAX_COORD:
                    print(f"Coordinate out of range: {coord}={validated_state[coord]:.1f}")
                    validated_state[coord] = np.clip(validated_state[coord], -MAX_COORD, MAX_COORD)

        # Normalize heading into [-pi, pi].
        if "heading" in validated_state:
            heading = validated_state["heading"]
            while heading > np.pi:
                heading -= 2 * np.pi
            while heading < -np.pi:
                heading += 2 * np.pi
            validated_state["heading"] = heading

        return validated_state

    def _interpolate_trajectory_by_time(self, trajectory, target_time):
        """
        Interpolate a trajectory state by timestamp.

        Args:
            trajectory: Trajectory data with timestamp fields.
            target_time: Target timestamp.

        Returns:
            Dict: Interpolated state data.
        """
        # Find the two points surrounding the target time.
        timestamps = [point["timestamp"] for point in trajectory]

        # Clamp when outside the trajectory range.
        if target_time <= timestamps[0]:
            return self._validate_trajectory_state(trajectory[0])
        elif target_time >= timestamps[-1]:
            return self._validate_trajectory_state(trajectory[-1])

        # Find the interval containing the target time.
        for i in range(len(timestamps) - 1):
            if timestamps[i] <= target_time <= timestamps[i + 1]:
                # Linear interpolation.
                t0, t1 = timestamps[i], timestamps[i + 1]
                p0, p1 = trajectory[i], trajectory[i + 1]

                # Interpolation weight.
                alpha = (target_time - t0) / (t1 - t0) if t1 != t0 else 0.0

                # Interpolate state values.
                interpolated_state = {
                    "x": p0["x"] + alpha * (p1["x"] - p0["x"]),
                    "y": p0["y"] + alpha * (p1["y"] - p0["y"]),
                    "speed": p0["speed"] + alpha * (p1["speed"] - p0["speed"]),
                    "heading": p0["heading"] + alpha * (p1["heading"] - p0["heading"]),
                    "timestamp": target_time
                }

                # Interpolate speed components when available.
                if "speed_x" in p0 and "speed_x" in p1:
                    interpolated_state["speed_x"] = p0["speed_x"] + alpha * (p1["speed_x"] - p0["speed_x"])
                    interpolated_state["speed_y"] = p0["speed_y"] + alpha * (p1["speed_y"] - p0["speed_y"])

                return self._validate_trajectory_state(interpolated_state)

        # If no interval was found, return the closest point.
        closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_time))
        return self._validate_trajectory_state(trajectory[closest_idx])

    def is_trajectory_finished(self, trajectory, sim_time):
        """
        Check whether a trajectory has finished.

        Args:
            trajectory: Vehicle trajectory data.
            sim_time: Current simulation time.

        Returns:
            bool: Whether the trajectory has finished.
        """
        if not trajectory:
            return True

        # Compute the target timestamp.
        target_time = self._trajectory_start_time + sim_time

        # Prefer original timestamps when available.
        if "original_timestamp" in trajectory[0]:
            last_timestamp = trajectory[-1]["original_timestamp"]
            return target_time > last_timestamp
        # Otherwise use interpolated timestamps.
        elif "timestamp" in trajectory[0]:
            last_timestamp = trajectory[-1]["timestamp"]
            return target_time > last_timestamp
        else:
            # Fall back to step-count comparison if timestamps are absent.
            max_steps = len(trajectory)
            current_step = int(sim_time / self.env.physics_world_step_size)
            return current_step >= max_steps

    def get_effective_time_step(self):
        """Get the effective time step after applying decision_repeat."""
        decision_repeat = self.env.engine.global_config.get('decision_repeat', 1)
        return self.env.physics_world_step_size * decision_repeat

    def update_simulation_time(self, time_increment):
        """Advance simulation time."""
        self.env._simulation_time += time_increment

    def reset_simulation_time(self):
        """Reset simulation time."""
        self.env._simulation_time = 0.0

    def get_simulation_time(self):
        """Return the current simulation time."""
        return self.env._simulation_time

    def get_trajectory_start_time(self):
        """Return the trajectory start time."""
        return self._trajectory_start_time
