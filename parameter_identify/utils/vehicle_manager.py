import numpy as np
from metadrive.component.vehicle.vehicle_type import DefaultVehicle


class VehicleManager:
    """
    Background vehicle manager.

    Responsibilities:
    - Create and destroy background vehicles.
    - Update vehicle state from trajectory data.
    - Support both position and dynamics update modes.
    - Clean up vehicles whose trajectories have ended.
    """

    def __init__(self, env):
        """
        Initialize the vehicle manager.

        Args:
            env: TrajectoryReplayEnv instance.
        """
        self.env = env
        self.ghost_vehicles = {}

    def replay_all_vehicles_by_time(self):
        """
        Replay all background vehicles according to simulation time.

        Supported modes:
        1) position: query the current state by simulation time, create or
           update a `DefaultVehicle`, and set position directly in kinematic mode.
        2) dynamics: update vehicles through the physics engine using CSV
           `speed_x` and `speed_y` for more realistic motion.
        """
        if not self.env.enable_background_vehicles:
            return

        for vid, traj in self.env.trajectory_dict.items():
            state = self.env._get_trajectory_state_at_time(traj, self.env._simulation_time)

            if state is None:
                self._remove_vehicle_if_exists(vid)
                continue

            if vid not in self.ghost_vehicles:
                self._create_background_vehicle(vid, state)

            if self.env.background_vehicle_update_mode == "position":
                self._update_vehicle_by_position(self.ghost_vehicles[vid], state)
            elif self.env.background_vehicle_update_mode == "dynamics":
                self._update_vehicle_by_dynamics(self.ghost_vehicles[vid], state, vid)

    def _remove_vehicle_if_exists(self, vid):
        """Remove a vehicle whose trajectory has ended."""
        if vid in self.ghost_vehicles:
            vehicle = self.ghost_vehicles[vid]
            print(f"Background vehicle {vid} trajectory ended; removing vehicle...")
            try:
                vehicle.destroy()
                print(f"Background vehicle {vid} removed successfully")
            except Exception as e:
                print(f"Error while removing background vehicle {vid}: {e}")
            del self.ghost_vehicles[vid]

    def _create_background_vehicle(self, vid, state):
        """Create a background vehicle."""
        vehicle_config = self.env.engine.global_config["vehicle_config"].copy()
        vehicle_config.update({
            "show_navi_mark": False,          # Hide navigation markers.
            "show_dest_mark": False,          # Hide destination markers.
            "show_line_to_dest": False,       # Hide destination lines.
            "show_line_to_navi_mark": False,  # Hide navigation lines.
            "show_navigation_arrow": False,   # Hide navigation arrows.
            "use_special_color": False,       # Avoid colors reserved for the ego vehicle.
        })

        if self.env.background_vehicle_update_mode == "position":
            vehicle_config.update({
                "mass": 1,                  # Minimize physical influence.
                "no_wheel_friction": True,  # Disable wheel friction.
            })
        elif self.env.background_vehicle_update_mode == "dynamics":
            vehicle_config.update({
                "mass": 1100,                # Use a normal vehicle mass.
                "no_wheel_friction": False,  # Enable wheel friction.
            })

        v = self.env.engine.spawn_object(
            DefaultVehicle,
            vehicle_config=vehicle_config,
            position=[0, 0],
            heading=0,
        )
        self._configure_vehicle_physics(v)
        self.ghost_vehicles[vid] = v

    def _configure_vehicle_physics(self, vehicle):
        """Configure vehicle physics."""
        if self.env.background_vehicle_update_mode == "position":
            if hasattr(vehicle, '_body') and hasattr(vehicle._body, 'disable'):
                try:
                    vehicle._body.disable()  # Keep the body out of physics simulation.
                except:
                    pass

            if hasattr(vehicle, '_body') and hasattr(vehicle._body, 'setKinematic'):
                try:
                    vehicle._body.setKinematic(True)
                except:
                    pass
        elif self.env.background_vehicle_update_mode == "dynamics":
            if hasattr(vehicle, '_body') and hasattr(vehicle._body, 'setKinematic'):
                try:
                    vehicle._body.setKinematic(False)
                except:
                    pass

    def _update_vehicle_by_position(self, vehicle, state):
        """
        Update a vehicle in position mode using the precomputed stable heading.
        """
        vehicle.set_position([state["x"], state["y"]])

        heading = state.get("heading", 0.0)
        vehicle.set_heading_theta(heading)

        speed_magnitude = state["speed"]

        MAX_REASONABLE_SPEED = 50.0  # About 180 km/h.
        if speed_magnitude > MAX_REASONABLE_SPEED:
            print(
                f"Unusually high vehicle speed: {speed_magnitude:.1f} m/s; "
                f"clamping to {MAX_REASONABLE_SPEED} m/s"
            )
            speed_magnitude = MAX_REASONABLE_SPEED

        if speed_magnitude > 0.01:
            direction = [np.cos(heading), np.sin(heading)]
            vehicle.set_velocity(direction, speed_magnitude)
        else:
            direction = [np.cos(heading), np.sin(heading)]
            vehicle.set_velocity(direction, 0.0)

    def _update_vehicle_by_dynamics(self, vehicle, state, vehicle_id):
        """
        Update a vehicle in dynamics mode using CSV speed components.
        """
        speed_x = state.get("speed_x", 0.0)
        speed_y = state.get("speed_y", 0.0)

        heading = state.get("heading", 0.0)
        vehicle.set_heading_theta(heading)

        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)

        MAX_REASONABLE_SPEED = 50.0  # About 180 km/h.
        if speed_magnitude > MAX_REASONABLE_SPEED:
            print(
                f"Background vehicle {vehicle_id} has unusually high speed "
                f"{speed_magnitude:.1f} m/s; clamping to {MAX_REASONABLE_SPEED} m/s"
            )
            scale_factor = MAX_REASONABLE_SPEED / speed_magnitude
            speed_x = speed_x * scale_factor
            speed_y = speed_y * scale_factor
            speed_magnitude = MAX_REASONABLE_SPEED

        if speed_magnitude > 0.01:
            direction = [speed_x / speed_magnitude, speed_y / speed_magnitude]
            vehicle.set_velocity(direction, speed_magnitude)
        else:
            vehicle.set_velocity([1.0, 0.0], 0.0)

        current_pos = vehicle.position
        target_pos = [state["x"], state["y"]]
        pos_error = np.sqrt(
            (current_pos[0] - target_pos[0])**2 + (current_pos[1] - target_pos[1])**2
        )

        if pos_error > 2.0:  # Correct drifts larger than 2 meters.
            vehicle.set_position(target_pos)
            print(f"Background vehicle {vehicle_id} position corrected: error={pos_error:.2f}m")

    def cleanup_finished_trajectories(self):
        """
        Remove background vehicles whose trajectories have finished.
        """
        if not self.env.enable_background_vehicles:
            return

        vehicles_to_remove = []

        for vid, vehicle in self.ghost_vehicles.items():
            if vid in self.env.trajectory_dict:
                traj = self.env.trajectory_dict[vid]
                if self.env._is_trajectory_finished(traj, self.env._simulation_time):
                    vehicles_to_remove.append(vid)

        for vid in vehicles_to_remove:
            if vid in self.ghost_vehicles:
                vehicle = self.ghost_vehicles[vid]
                try:
                    vehicle.destroy()
                except Exception as e:
                    print(f"Error while removing background vehicle {vid}: {e}")
                del self.ghost_vehicles[vid]

        if len(self.ghost_vehicles) == 0 and self.env.trajectory_dict:
            print("All background-vehicle trajectories have ended; only the ego vehicle remains in the scene")

    def cleanup_all_vehicles(self):
        """
        Destroy all background-vehicle objects.
        """
        for vid, vehicle in self.ghost_vehicles.items():
            try:
                vehicle.destroy()
            except:
                pass
        self.ghost_vehicles = {}

    def print_speed_comparison(self):
        """
        Print speed comparisons for the ego vehicle and background vehicles.
        """
        decision_repeat = self.env.engine.global_config.get('decision_repeat', 1)
        physics_step_size = self.env.physics_world_step_size
        effective_step_size = physics_step_size * decision_repeat

        print(f"\n=== Speed Comparison (Step {self.env._step_count}, Sim Time: {self.env._simulation_time:.3f}s) ===")
        print(f"Physics step: {physics_step_size:.3f}s, Decision repeat: {decision_repeat}, Effective: {effective_step_size:.3f}s")

        main_actual_speed = self.env.agent.speed
        main_expected_speed = "N/A"
        if self.env.main_vehicle_trajectory:
            main_state = self.env._get_trajectory_state_at_time(
                self.env.main_vehicle_trajectory,
                self.env._simulation_time,
            )
            if main_state:
                main_expected_speed = f"{main_state['speed']:.1f}"

        print(f"Main Car: Actual={main_actual_speed:.1f} m/s, Expected={main_expected_speed} m/s")

        if self.env.enable_background_vehicles and self.ghost_vehicles:
            print(f"Background Vehicles (Updated at {physics_step_size:.3f}s intervals):")
            for vid, traj in self.env.trajectory_dict.items():
                if vid in self.ghost_vehicles:
                    bg_vehicle = self.ghost_vehicles[vid]
                    actual_speed = bg_vehicle.speed if hasattr(bg_vehicle, 'speed') else 0.0

                    expected_state = self.env._get_trajectory_state_at_time(traj, self.env._simulation_time)
                    if expected_state:
                        expected_speed = expected_state["speed"]
                        position = bg_vehicle.position if hasattr(bg_vehicle, 'position') else [0, 0]
                        print(
                            f"  Vehicle {vid}: Actual={actual_speed:.1f} m/s, "
                            f"Expected={expected_speed:.1f} m/s, "
                            f"Pos=({position[0]:.1f}, {position[1]:.1f})"
                        )
                    else:
                        print(f"  Vehicle {vid}: Trajectory ended")
        elif not self.env.enable_background_vehicles:
            print("Background Vehicles: Disabled")

        print("=" * 70)

    def get_vehicle_count(self):
        """Return the current number of background vehicles."""
        return len(self.ghost_vehicles)

    def get_vehicles(self):
        """Return all background-vehicle objects."""
        return self.ghost_vehicles
