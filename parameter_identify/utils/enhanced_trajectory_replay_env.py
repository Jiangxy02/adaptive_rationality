
import pandas as pd
import numpy as np
import torch
import sys
import os
import time
from pathlib import Path
import sys


from metadrive.envs import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.engine.engine_utils import get_global_config
from metadrive.constants import HELP_MESSAGE

from parameter_identify.utils.trajectory_loader import TrajectoryLoader, load_trajectory


from parameter_identify.utils.observation_recorder import ObservationRecorder


from parameter_identify.sim_module.environment_factory import EnvironmentFactory


from parameter_identify.utils.ppo_controller import PPOController
from parameter_identify.utils.navigation_manager import NavigationManager
from parameter_identify.utils.vehicle_manager import VehicleManager
from parameter_identify.utils.time_synchronizer import TimeSynchronizer


class EnhancedTrajectoryReplayEnv(MetaDriveEnv):


    def __init__(self, trajectory_dict, config=None):


        user_config = config.copy() if config else {}
        self._user_config = user_config.copy()


        self._initialize_basic_attributes(user_config, trajectory_dict)


        self._setup_environment_config(user_config)


        self._initialize_ppo_controller(user_config)


        self._process_trajectory_data(trajectory_dict)


        self._initialize_time_control(user_config)


        self._initialize_termination_conditions(user_config)


        self._initialize_observation_recorder(user_config)


        self._add_rendering_safety_config(self._final_config)


        super().__init__(self._final_config)


        self.max_step = 10000


        self._last_observation = None



    def _after_lazy_init(self):

        super()._after_lazy_init()


        self._initialize_physics_parameters()


        self._replace_with_enhanced_engine()


        self._initialize_managers()

    def _replace_with_enhanced_engine(self):

        original_step = self.engine.step
        self.original_engine_step = original_step


        def enhanced_step(engine_self, step_num: int = 1) -> None:

            for i in range(step_num):

                self._update_background_vehicles_for_physics_step(i)


                self._execute_original_physics_step(i, step_num)


                if not hasattr(self, '_physics_step_count'):
                    self._physics_step_count = 0
                self._physics_step_count += 1


        import types
        self.engine.step = types.MethodType(enhanced_step, self.engine)


    def _update_background_vehicles_for_physics_step(self, step_index):

        if not hasattr(self, 'vehicle_manager') or not self.enable_background_vehicles:
            return


        physics_step_size = self.physics_world_step_size
        current_physics_time = self._simulation_time + step_index * physics_step_size


        original_sim_time = self._simulation_time
        self._simulation_time = current_physics_time


        try:
            self.vehicle_manager.replay_all_vehicles_by_time()

        except Exception as e:
            print(f"Physics step {step_index} background-vehicle update failed: {e}")
        finally:

            self._simulation_time = original_sim_time

    def _execute_original_physics_step(self, step_index, total_steps):


        for name, manager in self.engine.managers.items():
            if name != "record_manager":
                manager.step()


        self.engine.step_physics_world()


        if "record_manager" in self.engine.managers and step_index < total_steps - 1:
            self.engine.record_manager.step()


        if self.engine.force_fps.real_time_simulation and step_index < total_steps - 1:
            self.engine.task_manager.step()
        else:

            try:
                self.engine.task_manager.step()
            except Exception:
                pass

    def _initialize_basic_attributes(self, user_config, trajectory_dict):
        self._step_count = 0
        self._simulation_time = 0.0
        self._real_start_time = None
        self._last_step_time = None
        self._real_time_module = time


        self.ppo_checkpoint_path = user_config.pop("ppo_checkpoint_path", None)
        self.ppo_device = user_config.pop("ppo_device", "auto")
        self.use_cognitive_modules = user_config.pop("use_cognitive_modules", False)

        if not self.ppo_checkpoint_path:
            raise ValueError("PPO checkpoint path 'ppo_checkpoint_path' is required")

    def _setup_environment_config(self, user_config):

        env_factory = EnvironmentFactory()
        base_config = env_factory.get_config()



        custom_params = {
            'end_on_crash', 'end_on_horizon', 'enable_background_vehicles',
            'end_on_arrive_dest', 'enable_realtime', 'target_fps',
            'end_on_out_of_road', 'enable_observation_recording',
            'background_vehicle_update_mode', 'recording_session_name',
            'recording_output_dir', 'ppo_device',
            'use_cognitive_modules', '_custom_config', 'seed',
            'data_path', 'scenario_name'
        }

        custom_params.add('cognitive_modulation_override')


        self._final_config = base_config.copy()
        self._final_config["use_render"] = user_config.get("use_render", True)


        for key, value in user_config.items():
            if key not in custom_params and key not in ["vehicle_config"]:
                self._final_config[key] = value






    def _add_rendering_safety_config(self, config):

        if "camera_height" not in config:
            config["camera_height"] = 80
        if "camera_dist" not in config:
            config["camera_dist"] = 15
        if "camera_pitch" not in config:
            config["camera_pitch"] = -15
        if "camera_smooth" not in config:
            config["camera_smooth"] = True
        if "window_size" not in config:
            config["window_size"] = (1200, 900)

    def _initialize_ppo_controller(self, user_config):
        modulation_override = self._user_config.get('cognitive_modulation_override')
        self.ppo_controller = PPOController(
            checkpoint_path=self.ppo_checkpoint_path,
            device=self.ppo_device,
            use_cognitive_modules=self.use_cognitive_modules,
            cognitive_modulation_override=modulation_override if modulation_override != 'auto' else None,
        )

    def _process_trajectory_data(self, trajectory_dict):

        self.enable_background_vehicles = self._user_config.get("enable_background_vehicles", False)


        original_trajectory_dict = trajectory_dict.copy()


        self.main_vehicle_trajectory = None
        if -1 in original_trajectory_dict:
            self.main_vehicle_trajectory = original_trajectory_dict.pop(-1)
        else:
            print("Warning: Vehicle -1 not found in trajectory data")


        if self.enable_background_vehicles:
            self.trajectory_dict = original_trajectory_dict
        else:
            self.trajectory_dict = {}
            print(" Background vehicles disabled - CSV background vehicle data skipped")

    def _initialize_time_control(self, user_config):

        self.enable_realtime = self._user_config.get("enable_realtime", True)
        self.target_fps = self._user_config.get("target_fps", 50.0)


        self.background_vehicle_update_mode = self._user_config.get("background_vehicle_update_mode", "position")
        if self.background_vehicle_update_mode not in ["position", "dynamics"]:
            self.background_vehicle_update_mode = "position"

    def _initialize_termination_conditions(self, user_config):

        self.end_on_crash = self._user_config.get("end_on_crash", True)
        self.end_on_out_of_road = self._user_config.get("end_on_out_of_road", False)
        self.end_on_arrive_dest = self._user_config.get("end_on_arrive_dest", False)
        self.end_on_horizon = self._user_config.get("end_on_horizon", False)

    def _initialize_observation_recorder(self, user_config):

        self.enable_observation_recording = self._user_config.get("enable_observation_recording", False)
        self.observation_recorder = None
        if self.enable_observation_recording:
            session_name = self._user_config.get("recording_session_name", None)
            output_dir = self._user_config.get("recording_output_dir", "observation_logs")
            self.observation_recorder = ObservationRecorder(output_dir=output_dir, session_name=session_name)


    def _initialize_physics_parameters(self):
        try:
            self.physics_world_step_size = self.engine.physics_world.static_world.getPhysicsWorldStepSize()
        except AttributeError:

            self.physics_world_step_size = 0.02  # 50Hz

    def _initialize_managers(self):

        self.time_synchronizer = TimeSynchronizer(self)
        self.time_synchronizer.setup_time_synchronization()


        self.vehicle_manager = VehicleManager(self)


        self.navigation_manager = NavigationManager(self)

    def get_current_observation(self):

        try:

            if hasattr(self.agent, 'get_state'):

                state = self.agent.get_state()
                if hasattr(state, 'observation'):
                    return state.observation


            if hasattr(self, 'observe') and callable(self.observe):
                obs = self.observe(self.agent)
                return obs


            elif hasattr(self.agent, 'observe'):
                obs = self.agent.observe(self.agent)
                return obs


            elif hasattr(self, 'observation_space'):

                from metadrive.obs.observation_base import BaseObservation
                if hasattr(self.agent, 'observation'):
                    obs_manager = self.agent.observation
                    if hasattr(obs_manager, 'observe'):
                        return obs_manager.observe(self.agent)




            obs = np.zeros(275)


            pos = self.agent.position
            obs[0:2] = pos[:2]  # x, y position
            obs[2] = self.agent.speed  # speed
            obs[3] = self.agent.heading_theta  # heading

            if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
                return self._extend_observation_with_cognitive_params(obs, {})
            else:

                return obs

        except Exception as e:


            if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
                signature = self._get_network_signature()
                base_dim = int(signature.get('base_obs_dim', 275))
                return self._extend_observation_with_cognitive_params(np.zeros(base_dim), {})
            else:
                return np.zeros(275)

    def _force_cleanup_before_reset(self):

        if not hasattr(self, 'engine') or not self.engine:
            return

        try:



            for manager_name, manager in self.engine.managers.items():
                if hasattr(manager, 'spawned_objects') and manager.spawned_objects:


                    manager.clear_objects(list(manager.spawned_objects.keys()), force_destroy=True)
                    manager.spawned_objects.clear()


            if hasattr(self.engine, '_spawned_objects') and self.engine._spawned_objects:

                self.engine.clear_objects(list(self.engine._spawned_objects.keys()), force_destroy=True)



        except Exception as e:
            print(f"Cleanup before environment reset failed: {e}")


    def reset(self, theta, seed=None):


        self._force_cleanup_before_reset()


        if hasattr(self, 'vehicle_manager'):
            self.vehicle_manager.cleanup_all_vehicles()

        obs = super().reset(seed=seed)


        if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
            obs = self._extend_observation_with_cognitive_params(obs,theta)

        self._step_count = 0
        self._real_start_time = self._real_time_module.time()
        self._last_step_time = self._real_start_time


        self.time_synchronizer.reset_simulation_time()


        self._physics_step_count = 0


        self._last_observation = obs


        self._display_decision_repeat_info()


        if hasattr(self, 'config') and 'initial_timestamp' in self.config:
            initial_timestamp = self.config['initial_timestamp']

            self.time_synchronizer.initialize_trajectory_start_time(custom_start_timestamp=initial_timestamp)
        else:

            self.time_synchronizer.initialize_trajectory_start_time()


        self.navigation_manager.set_custom_destination()


        self.navigation_manager.debug_navigation_info()


        if hasattr(self, 'config') and 'initial_state' in self.config:
            initial_state = self.config['initial_state']

            #       f"vel=({initial_state['vx']:.1f}, {initial_state['vy']:.1f})")
            self._initialize_main_vehicle_from_state(initial_state)
        else:

            self._initialize_main_vehicle_state()


        try:
            cam = getattr(self.engine, "main_camera", None)
            if cam:
                cam.track(self.agent)
                cam.set_bird_view_pos(self.agent.position)
        except Exception as e:
            print(f"Camera failed to track the ego vehicle: {e}")

        return obs

    def _display_decision_repeat_info(self):
        if not hasattr(self, '_decision_repeat_displayed'):
            self._decision_repeat_displayed = True

    def _initialize_main_vehicle_state(self):
        if self.main_vehicle_trajectory and len(self.main_vehicle_trajectory) > 0:
            initial_state = self.main_vehicle_trajectory[0]

            self.agent.set_position([initial_state["x"], initial_state["y"]])
            self.agent.set_heading_theta(initial_state["heading"])

            direction = [np.cos(initial_state["heading"]), np.sin(initial_state["heading"])]
            self.agent.set_velocity(direction, initial_state["speed"])


            self.navigation_manager.fix_lane_detection()

    def _initialize_main_vehicle_from_state(self, initial_state):

        px, py = initial_state['px'], initial_state['py']
        vx, vy = initial_state['vx'], initial_state['vy']


        speed = np.sqrt(vx**2 + vy**2)
        heading = np.arctan2(vy, vx) if speed > 0.1 else 0.0


        self.agent.set_position([px, py])
        self.agent.set_heading_theta(heading)


        if speed > 0.1:
            direction = [np.cos(heading), np.sin(heading)]
            self.agent.set_velocity(direction, speed)
        else:

            self.agent.set_velocity([1, 0], 0)


        #       f"heading={heading:.2f} rad, speed={speed:.1f} m/s")


        self.navigation_manager.fix_lane_detection()

    def step(self, action=None, theta=None):


        self._handle_realtime_control()


        decision_repeat = self.engine.global_config.get('decision_repeat', 1)
        effective_time_step = self.physics_world_step_size * decision_repeat
        self._simulation_time += effective_time_step


        if action is None:
            current_obs = self._last_observation if self._last_observation is not None else self.get_current_observation()
            ppo_action = self.ppo_controller.get_action(current_obs, deterministic=True)
            act = ppo_action
        else:
            act = action

            ppo_action = action


        obs, reward, terminated, truncated, info = super().step(act)


        if hasattr(self, 'use_cognitive_modules') and self.use_cognitive_modules:
            obs = self._extend_observation_with_cognitive_params(obs,theta)


        self._last_observation = obs


        self._record_observation_if_enabled(ppo_action, obs, reward, info)


        self.vehicle_manager.cleanup_finished_trajectories()

        self._step_count += 1


        if self._step_count % 10 == 0:
            self.vehicle_manager.print_speed_comparison()


        done = self._check_termination_conditions(info)


        self._add_diagnostic_info(info, terminated, truncated, done, ppo_action)

        return obs, reward, done, info

    def _handle_realtime_control(self):
        if self.enable_realtime and self._last_step_time is not None:
            current_time = self._real_time_module.time()
            target_step_duration = 1.0 / self.target_fps
            elapsed_since_last_step = current_time - self._last_step_time

            if elapsed_since_last_step < target_step_duration:
                sleep_duration = target_step_duration - elapsed_since_last_step
                self._real_time_module.sleep(sleep_duration)

            self._last_step_time = self._real_time_module.time()

    def _record_observation_if_enabled(self, ppo_action, obs, reward, info):
        if self.observation_recorder:
            action_info = {"source": "ppo_model", "success": True}
            self.observation_recorder.record_step(
                env=self,
                action=ppo_action,
                action_info=action_info,
                obs=obs,
                reward=reward,
                info=info,
                step_count=self._step_count
            )

    def _check_termination_conditions(self, info):
        crash_flag = info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False) or info.get("crash_building", False)
        out_of_road_flag = info.get("out_of_road", False)
        arrive_dest_flag = info.get("arrive_dest", False)
        horizon_reached_flag = (self._step_count >= self.max_step)

        should_end = False
        if crash_flag and getattr(self, "end_on_crash", True):
            should_end = True
        if out_of_road_flag and getattr(self, "end_on_out_of_road", False):
            should_end = True
        if arrive_dest_flag and getattr(self, "end_on_arrive_dest", False):
            should_end = True
        if horizon_reached_flag and getattr(self, "end_on_horizon", False):
            should_end = True

        return bool(should_end)

    def _add_diagnostic_info(self, info, terminated, truncated, done, ppo_action):
        info["termination_overridden"] = (terminated or truncated) and (not done)
        info["crash_flag"] = info.get("crash", False)
        info["out_of_road_flag"] = info.get("out_of_road", False)
        info["arrive_dest_flag"] = info.get("arrive_dest", False)
        info["horizon_reached_flag"] = (self._step_count >= self.max_step)
        info["simulation_time"] = self._simulation_time
        info["enhanced_background_vehicles"] = True


        info["Control Mode"] = "PPO Model (Enhanced)"
        info["action_source"] = "ppo_model"
        info["ppo_action"] = ppo_action

    def render(self, *args, **kwargs):

        render_text = kwargs.get("text", {})


        real_elapsed_time = self._real_time_module.time() - self._real_start_time if self._real_start_time else 0.0
        time_ratio = self._simulation_time / real_elapsed_time if real_elapsed_time > 0 else 0.0


        distance_to_dest = self._calculate_distance_to_destination()


        ppo_info = ""
        if hasattr(self.ppo_controller, 'checkpoint_info'):
            iteration = self.ppo_controller.checkpoint_info.get('iteration', 'Unknown')
            ppo_info = f" (Iter: {iteration})"

        render_text.update({
            "Control Mode": f"PPO Model Enhanced{ppo_info}",
            "Step": f"{self._step_count}/{self.max_step}",
            "Simulation Time": f"{self._simulation_time:.1f}s",
            "Real Time": f"{real_elapsed_time:.1f}s",
            "Time Ratio": f"{time_ratio:.1f}x",
            "Physics Step": f"{self.physics_world_step_size:.3f}s",
            "Background Update": "0.02s (Enhanced)",
            "Realtime Mode": "ON" if self.enable_realtime else "OFF",
            "Target FPS": f"{self.target_fps:.0f}",
            "Main Car Position": f"({self.agent.position[0]:.1f}, {self.agent.position[1]:.1f})",
            "Main Car Speed": f"{self.agent.speed:.1f} m/s",
            "Distance to Destination": distance_to_dest,
            "Background Vehicles": f"{self.vehicle_manager.get_vehicle_count()}" + ("" if self.enable_background_vehicles else " (Disabled)"),
        })
        kwargs["text"] = render_text
        return super().render(*args, **kwargs)

    def _calculate_distance_to_destination(self):
        distance_to_dest = "N/A"
        if hasattr(self, 'custom_destination'):
            dest = self.custom_destination
            agent_pos = self.agent.position
            distance_to_dest = f"{np.sqrt((agent_pos[0] - dest[0])**2 + (agent_pos[1] - dest[1])**2):.1f}m"
        return distance_to_dest

    def close(self):


        if self.observation_recorder:
            self.observation_recorder.finalize_recording()


        if hasattr(self, 'vehicle_manager'):
            self.vehicle_manager.cleanup_all_vehicles()

        super().close()


    def _get_trajectory_state_at_time(self, trajectory, sim_time):
        return self.time_synchronizer.get_trajectory_state_at_time(trajectory, sim_time)

    def _is_trajectory_finished(self, trajectory, sim_time):
        return self.time_synchronizer.is_trajectory_finished(trajectory, sim_time)

    def set_custom_config(self, custom_config):


        if 'use_cognitive_modules' in custom_config:
            self.use_cognitive_modules = custom_config['use_cognitive_modules']


        if hasattr(self, 'ppo_controller') and self.ppo_controller is not None:

            if 'use_cognitive_modules' in custom_config:
                if hasattr(self.ppo_controller, 'set_cognitive_config'):
                    self.ppo_controller.set_cognitive_config(custom_config)



        if not hasattr(self, '_custom_config'):
            self._custom_config = {}
        self._custom_config.update(custom_config)

    def _extend_observation_with_cognitive_params(self, obs, theta):

        theta = theta or {}
        signature = self._get_network_signature()
        base_dim = int(signature.get('base_obs_dim', 275))
        raw_dim = int(signature.get('raw_obs_dim') or signature.get('obs_dim') or base_dim)
        param_dim = int(signature.get('cognitive_param_dim', 0) or 0)
        mask_dim = int(signature.get('cognitive_mask_dim', 0) or 0)
        modulation = signature.get('cognitive_modulation', 'none')


        if isinstance(obs, (tuple, list)):

            if len(obs) > 0:
                obs = obs[0]
            else:
                raise ValueError("Observation is empty; cannot build input for the checkpoint signature")


        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)


        if obs.ndim > 1:
            obs = obs.flatten()

        if len(obs) not in {base_dim, raw_dim}:
            raise ValueError(
                "Observation dimensions do not match the checkpoint signature; cannot build input: "
                f"expected_one_of={[base_dim, raw_dim]}, actual={len(obs)}"
            )

        base_obs = obs[:base_dim].astype(np.float32, copy=False)
        if modulation == 'none' or param_dim == 0:
            if raw_dim != base_dim:
                raise ValueError(f"Signature mismatch: modulation={modulation}, raw_dim={raw_dim}, base_dim={base_dim}")
            return base_obs

        params = (
            self._build_signature_cognitive_params(param_dim, theta)
            if self.use_cognitive_modules
            else np.zeros(param_dim, dtype=np.float32)
        )

        if modulation == 'concat':
            masks = (
                np.ones(mask_dim, dtype=np.float32)
                if self.use_cognitive_modules
                else np.zeros(mask_dim, dtype=np.float32)
            )
            result = np.concatenate([base_obs, params, masks], axis=0).astype(np.float32)
            if result.shape[0] != raw_dim:
                raise ValueError(f"Concat input dimension does not match the checkpoint signature: expected={raw_dim}, actual={result.shape[0]}")
            return result

        raise ValueError(f"Unsupported cognitive modulation mode: {modulation}")

    def _get_network_signature(self):
        if hasattr(self, 'ppo_controller') and self.ppo_controller is not None:
            signature = getattr(self.ppo_controller, 'network_signature', None)
            if signature:
                return signature
            network = getattr(self.ppo_controller, 'network', None)
            if network is not None:
                return {
                    'obs_dim': getattr(network, 'obs_dim', None),
                    'raw_obs_dim': getattr(network, 'raw_obs_dim', getattr(network, 'obs_dim', None)),
                    'base_obs_dim': getattr(network, 'base_obs_dim', None),
                    'cognitive_param_dim': getattr(network, 'cognitive_param_dim', 0),
                    'cognitive_mask_dim': getattr(network, 'cognitive_mask_dim', 0),
                    'cognitive_modulation': getattr(network, 'cognitive_modulation', 'none'),
                }
        return {
            'obs_dim': 275,
            'raw_obs_dim': 275,
            'base_obs_dim': 275,
            'cognitive_param_dim': 0,
            'cognitive_mask_dim': 0,
            'cognitive_modulation': 'none',
        }

    def _build_signature_cognitive_params(self, dim, theta):
        controller = getattr(self, 'ppo_controller', None)
        values = [
            theta.get('bias_inverse_tta_coef', getattr(controller, '_current_bias_coef', 1.5)),
            theta.get('perception_sigma0', getattr(controller, '_current_sigma0', 0.1)),
            theta.get('perception_sigma_max', getattr(controller, '_current_sigma_max', 0.8)),
            float(theta.get('delay_steps', getattr(controller, '_current_delay', 2.0))),
        ]
        if dim > len(values):
            raise ValueError(
                "Checkpoint cognitive-parameter dimension exceeds what the enhanced replay environment can construct: "
                f"cognitive_param_dim={dim}, supported={len(values)}"
            )
        vector = np.zeros(dim, dtype=np.float32)
        limit = min(dim, len(values))
        vector[:limit] = np.asarray(values[:limit], dtype=np.float32)
        return vector
