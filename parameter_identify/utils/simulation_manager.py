#!/usr/bin/env python3
"""Simulation management for environment control and trajectory rollouts."""


import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy
import sys
import os
import argparse
import logging

from common.cognitive_input import validate_perception_sigmas
from common.random_seed import SeedDomain, derive_seed, seed_global_generators

logging.getLogger('cognitive_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_bias_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_perception_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_delay_module').setLevel(logging.WARNING)


from parameter_identify.utils.likelihood_calculator import TrajectoryPoint
from parameter_identify.utils.trajectory_loader import TrajectoryLoader

from parameter_identify.sim_module.cognitive_module_manager import CognitiveModuleManager
from parameter_identify.sim_module.action_processor import ActionProcessor


class RolloutStateError(RuntimeError):
    """Raised when real vehicle state is unavailable; never fabricate trajectory points."""

    def __init__(self, step, scenario, reason, cause=None):
        self.step = step
        self.scenario = scenario or 'unknown scenario'
        message = f"rollout aborted at step {step} for {self.scenario}: {reason}"
        if cause is not None:
            message = f"{message} (caused by {type(cause).__name__}: {cause})"
        super().__init__(message)


class SimulationManager:
    """Manage the simulation environment and generate trajectories."""

    @staticmethod
    def _validate_perception_sigmas(theta: Dict[str, float]) -> Tuple[float, float]:
        """Reject perception parameters outside the paper domain."""
        return validate_perception_sigmas(
            theta.get('perception_sigma0', 0.1),
            theta.get('perception_sigma_max', 0.8),
        )

    def __init__(self,
                 env_config: Dict[str, Any],
                 ppo_model_path: str,
                 use_cognitive_modules: bool = False,
                 device: str = 'cpu',
                 cognitive_args: Optional[argparse.Namespace] = None):
        """Initialize the simulation manager.

        Args:
            env_config: MetaDrive environment configuration.
            ppo_model_path: Path to the PPO checkpoint used for actions.
            use_cognitive_modules: Whether to attach cognitive modules.
            device: Compute device for policy inference.
            cognitive_args: Optional cognitive-module configuration.
        """
        self.env_config = env_config
        self.ppo_model_path = ppo_model_path
        self.use_cognitive_modules = use_cognitive_modules
        self.device = device

        self.env = None
        self.ppo_model = None
        self.action_processor = None

        self._cognitive_args_template = None
        self._checkpoint_obs_dim = None

        if use_cognitive_modules:
            if cognitive_args:
                self._cognitive_args_template = copy.deepcopy(cognitive_args)
                self.cognitive_manager = CognitiveModuleManager(
                    args=copy.deepcopy(cognitive_args)
                )
            else:
                import argparse
                default_cognitive_args = argparse.Namespace()
                default_cognitive_args.use_cognitive_modules = True
                default_cognitive_args.use_cognitive_bias = True
                default_cognitive_args.use_cognitive_delay = True
                default_cognitive_args.use_cognitive_perception = True
                default_cognitive_args.bias_inverse_tta_coef = 1.5
                default_cognitive_args.bias_tta_threshold = 0.1
                default_cognitive_args.bias_visual_distance = 300.0
                default_cognitive_args.perception_sigma0 = 0.1
                default_cognitive_args.perception_sigma_max = 0.8
                default_cognitive_args.perception_use_kf = True
                default_cognitive_args.perception_kf_dt = 0.1
                default_cognitive_args.perception_kf_q_scale = 100.0
                default_cognitive_args.delay_steps = 2
                default_cognitive_args.enable_radar_beam_viz = False
                self._cognitive_args_template = copy.deepcopy(default_cognitive_args)
                self.cognitive_manager = CognitiveModuleManager(
                    args=copy.deepcopy(default_cognitive_args)
                )
        else:
            self.cognitive_manager = None
            self._cognitive_args_template = None

        self.original_data = None
        self.original_timestamps = None

    def load_environment(self):
        """Lazily construct the simulation environment."""
        if self.env is None:
            try:
                from metadrive.engine.engine_utils import close_engine, engine_initialized
                if engine_initialized():
                    close_engine()
            except Exception as e:
                print(f"Failed to clean up the existing engine instance: {e}")

            from parameter_identify.utils.enhanced_trajectory_replay_env import EnhancedTrajectoryReplayEnv

            trajectory_data = getattr(self, 'trajectory_dict', {})
            config = self.env_config.copy()
            # Identification owns the full rollout horizon, so ordinary driving
            # terminal events must not truncate the observed trajectory.
            config['force_destroy'] = True
            config['end_on_out_of_road'] = False
            config['end_on_crash'] = False
            config['end_on_arrive_dest'] = False
            config['end_on_horizon'] = False

            custom_config = {}
            if self.use_cognitive_modules:
                custom_config['use_cognitive_modules'] = True
                custom_config['perception_sigma0'] = 0.1
                custom_config['perception_sigma_max'] = 0.8
                custom_config['delay_steps'] = 2
            else:
                custom_config['use_cognitive_modules'] = False

            if 'ppo_checkpoint_path' not in config:
                config['ppo_checkpoint_path'] = self.ppo_model_path

            self.custom_config = custom_config

            self.env = EnhancedTrajectoryReplayEnv(trajectory_data, config)
            if hasattr(self.env, 'set_custom_config') and custom_config:
                self.env.set_custom_config(custom_config)

    def load_ppo_model(self):
        """Lazily load the PPO checkpoint and create the action processor."""
        if self.ppo_model is None:
            from parameter_identify.sim_module.checkpoint_loader import CheckpointLoader

            checkpoint_loader = CheckpointLoader(self.ppo_model_path, self.device)
            use_cognitive_modules_for_network = self.use_cognitive_modules
            modulation_override = self.env_config.get('cognitive_modulation_override', None)
            network = checkpoint_loader.load_checkpoint(
                use_cognitive_modules_for_network,
                cognitive_modulation_override=modulation_override
            )

            actual_network_obs_dim = getattr(network, 'obs_dim', None)

            if actual_network_obs_dim is not None:
                self._checkpoint_obs_dim = actual_network_obs_dim

            if actual_network_obs_dim is None or int(actual_network_obs_dim) <= 0:
                raise ValueError(f"invalid network observation dimension: {actual_network_obs_dim}")

            self.action_processor = ActionProcessor(
                network=network,
                cognitive_manager=self.cognitive_manager,
                device=self.device
            )

            self.network_signature = {
                'obs_dim': getattr(network, 'obs_dim', None),
                'raw_obs_dim': getattr(network, 'raw_obs_dim', getattr(network, 'obs_dim', None)),
                'base_obs_dim': getattr(network, 'base_obs_dim', None),
                'cognitive_param_dim': getattr(network, 'cognitive_param_dim', 0),
                'cognitive_mask_dim': getattr(network, 'cognitive_mask_dim', 0),
                'cognitive_modulation': getattr(network, 'cognitive_modulation', 'none'),
            }
            if self.cognitive_manager is not None and hasattr(self.cognitive_manager, 'set_network_signature'):
                self.cognitive_manager.set_network_signature(self.network_signature)

            print("Inferred network configuration:")
            print(f"   Input dimension: {self.network_signature['obs_dim']} (base {self.network_signature['base_obs_dim']})")
            print(f"   Cognitive modulation: {self.network_signature['cognitive_modulation']}")
            print(f"   Cognitive parameter dimension: {self.network_signature['cognitive_param_dim']}")
            print(f"   Cognitive mask dimension: {self.network_signature['cognitive_mask_dim']}")

            self.ppo_model = network

            return self.ppo_model

        return self.ppo_model

    def _load_ppo_checkpoint(self, checkpoint_path: str):
        """Load a PPO checkpoint. Retained for backward compatibility."""
        import torch

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            return None

        except Exception as e:
            return None

    def _force_cleanup_environment(self):
        """Force-clean environment objects before a reset."""
        if not self.env or not hasattr(self.env, 'engine'):
            return

        try:
            print("Force-cleaning environment objects...")
            if self.env.engine is None:
                print(" Engine not initialized; skipping cleanup")
                return

            for manager_name, manager in self.env.engine.managers.items():
                if hasattr(manager, 'spawned_objects') and manager.spawned_objects:
                    print(f"  Clearing {len(manager.spawned_objects)} objects from manager {manager_name}")
                    manager.clear_objects(list(manager.spawned_objects.keys()), force_destroy=True)
                    manager.spawned_objects.clear()

            if hasattr(self.env.engine, '_spawned_objects') and self.env.engine._spawned_objects:
                print(f"  Clearing {len(self.env.engine._spawned_objects)} engine-level objects")
                self.env.engine.clear_objects(list(self.env.engine._spawned_objects.keys()), force_destroy=True)

            if hasattr(self.env.engine, 'physics_world'):
                physics_world = self.env.engine.physics_world
                for world in [physics_world.dynamic_world, physics_world.static_world]:
                    bodies = world.getRigidBodies() + world.getSoftBodies() + world.getGhosts() + world.getVehicles() + world.getCharacters()
                    for body in bodies:
                        if body.getName() not in ["detector_mask", "debug"]:
                            try:
                                world.removeRigidBody(body)
                            except:
                                pass

            print("Environment object cleanup complete")

        except Exception as e:
            print(f"Error while cleaning the environment: {e}")

    def close(self):
        """Close the owned simulation environment exactly once."""
        env = self.env
        if env is None:
            return
        try:
            env.close()
        finally:
            self.env = None

    def reset_to_state(self, initial_state: Dict[str, float],
                      timestamp: Optional[float] = None,
                      theta: Optional[Dict[str, float]] = None,
                      seed: Optional[int] = None) -> np.ndarray:
        """Reset the environment to an observed initial state.

        Existing engine objects are force-cleaned before reset so a reused
        MetaDrive engine cannot leak actors into the next identification window.

        Args:
            initial_state: Observed ego position and velocity components.
            timestamp: Optional source timestamp for replay alignment.
            theta: Cognitive parameters applied during reset.
            seed: Optional deterministic rollout seed.

        Returns:
            The observation returned by the environment reset.
        """
        self.load_environment()
        self._force_cleanup_environment()
        reset_config = copy.deepcopy(self.env_config)
        if timestamp is not None:
            reset_config['initial_timestamp'] = timestamp
            reset_config['start_from_timestamp'] = True

        reset_config['initial_state'] = initial_state
        if hasattr(self.env, 'config'):
            self.env.config.update(reset_config)
        environment_seed = None
        if seed is not None:
            start_seed = int(self.env.start_index)
            num_scenarios = int(self.env.num_scenarios)
            scenario_offset = derive_seed(seed, SeedDomain.ENVIRONMENT) % num_scenarios
            environment_seed = start_seed + scenario_offset
        obs = self.env.reset(theta, seed=environment_seed)

        return obs

    def rollout(self,
                theta: Dict[str, float],
                initial_state: Dict[str, float],
                num_steps: int,
                start_timestamp: Optional[float] = None,
                deterministic: bool = True,
                return_collision_info: bool = False,
                if_prediction: bool = False,
                seed: Optional[int] = None) -> Union[List[TrajectoryPoint], Tuple[List[TrajectoryPoint], Dict]]:
        """Run one closed-loop simulation rollout.

        Args:
            theta: Cognitive parameter values for this rollout.
            initial_state: Observed ego state at the window start.
            num_steps: Number of policy steps to simulate.
            start_timestamp: Optional replay timestamp for the initial state.
            deterministic: Whether policy actions use their deterministic mode.
            return_collision_info: Return collision metadata with the trajectory.
            if_prediction: Mark the rollout as future prediction rather than fitting.
            seed: Optional deterministic seed for all rollout random streams.

        Returns:
            The simulated trajectory, optionally paired with collision metadata.
        """

        if seed is not None:
            seed_global_generators(seed)
            policy_seed = derive_seed(seed, SeedDomain.POLICY)
            perception_seed = derive_seed(seed, SeedDomain.PERCEPTION)
            try:
                import torch
                torch.manual_seed(policy_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(policy_seed)
            except ImportError:
                pass
            import random as py_random
            np.random.seed(perception_seed)
            py_random.seed(perception_seed)
            perception_module = getattr(
                getattr(self, "cognitive_manager", None),
                "cognitive_perception_module",
                None,
            )
            if perception_module is not None:
                perception_module.noise_config["random_seed"] = perception_seed
                noise_lidar = getattr(perception_module, "noise_lidar", None)
                if noise_lidar is not None:
                    noise_lidar.random_seed = perception_seed
                    noise_lidar.rng = np.random.default_rng(perception_seed)

        self._update_cognitive_parameters(theta)
        obs = self.reset_to_state(initial_state, start_timestamp, theta, seed=seed)
        self.load_ppo_model()
        if self.use_cognitive_modules and self.cognitive_manager:
            self.cognitive_manager.attach_to_env(self.env)
            self.cognitive_manager.reset_modules()
            print("Cognitive modules attached to the environment and reset")

        trajectory = []
        collision_info = {
            'collision_detected': False,
            'collision_step': None,
            'collision_time': None,
            'collision_type': None,
            'out_of_road_detected': False,
            'out_of_road_step': None,
            'out_of_road_time': None,
            'termination_reason': None,
            'all_step_info': []
        } if return_collision_info else None

        vehicle = getattr(self.env, 'vehicle', None)
        if vehicle is None:
            raise RolloutStateError(
                step=0,
                scenario=getattr(self, 'scenario_path', None),
                reason="environment returned no vehicle after reset; "
                       "refusing to fabricate the initial trajectory point",
            )
        try:
            position = vehicle.position
            velocity = vehicle.velocity
            heading = getattr(vehicle, 'heading', 0.0)
        except Exception as exc:
            raise RolloutStateError(
                step=0,
                scenario=getattr(self, 'scenario_path', None),
                reason="failed to read the vehicle state after reset",
                cause=exc,
            ) from exc

        trajectory.append(TrajectoryPoint(
            px=position[0],
            py=position[1],
            vx=velocity[0],
            vy=velocity[1],
            timestamp=start_timestamp or 0.0,
            yaw=heading
        ))

        for step in range(num_steps):
            action = self.action_processor.get_action(
                observation=obs,
                deterministic=deterministic,
                env=self.env,
                step_count=step,
                theta=theta
            )

            obs, reward, done, info = self.env.step(action, theta)
            original_reward = reward

            if return_collision_info and collision_info is not None:
                current_time = (start_timestamp or 0.0) + (step + 1) * 0.1

                crash_flag = info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False) or info.get("crash_building", False)
                out_of_road_flag = bool(
                    info.get("out_of_road", False)
                    or info.get("out_of_road_flag", False)
                )

                step_info = {
                    'step': step + 1,
                    'timestamp': current_time,
                    'crash_flag': crash_flag,
                    'out_of_road_flag': out_of_road_flag,
                    'done': done,
                    'info': info.copy()
                }
                collision_info['all_step_info'].append(step_info)

                if crash_flag and not collision_info['collision_detected']:
                    collision_info['collision_detected'] = True
                    collision_info['collision_step'] = step + 1
                    collision_info['collision_time'] = current_time

                    if info.get("crash_vehicle", False):
                        collision_info['collision_type'] = 'vehicle'
                    elif info.get("crash_object", False):
                        collision_info['collision_type'] = 'object'
                    elif info.get("crash_building", False):
                        collision_info['collision_type'] = 'building'
                    elif info.get("crash_sidewalk", False):
                        collision_info['collision_type'] = 'sidewalk'
                    else:
                        collision_info['collision_type'] = 'unknown'

                if out_of_road_flag and not collision_info['out_of_road_detected']:
                    collision_info['out_of_road_detected'] = True
                    collision_info['out_of_road_step'] = step + 1
                    collision_info['out_of_road_time'] = current_time

                if done:
                    if crash_flag:
                        collision_info['termination_reason'] = 'collision'
                    elif out_of_road_flag:
                        collision_info['termination_reason'] = 'out_of_road'
                    elif info.get("arrive_dest", False):
                        collision_info['termination_reason'] = 'arrive_destination'
                    elif step + 1 >= num_steps:
                        collision_info['termination_reason'] = 'max_steps'
                    else:
                        collision_info['termination_reason'] = 'unknown'
            if self.use_cognitive_modules and self.cognitive_manager:
                reward, bias_applied, bias_info = self.cognitive_manager.process_reward(reward, self.env, info)
                if bias_applied and step < 5:
                    print(f"  Step {step}: cognitive bias {original_reward:.3f} -> {reward:.3f}")

            current_time = (start_timestamp or 0.0) + (step + 1) * 0.1
            vehicle = getattr(self.env, 'vehicle', None)
            if vehicle is None:
                if done:
                    break
                raise RolloutStateError(
                    step=step,
                    scenario=getattr(self, 'scenario_path', None),
                    reason="vehicle disappeared before termination; "
                           "refusing to fabricate trajectory points",
                )
            try:
                position = vehicle.position
                velocity = vehicle.velocity
                heading = getattr(vehicle, 'heading', 0.0)
            except Exception as exc:
                raise RolloutStateError(
                    step=step,
                    scenario=getattr(self, 'scenario_path', None),
                    reason="failed to read the vehicle state",
                    cause=exc,
                ) from exc

            trajectory.append(TrajectoryPoint(
                px=position[0],
                py=position[1],
                vx=velocity[0],
                vy=velocity[1],
                timestamp=current_time,
                yaw=heading
            ))

            if done:
                break

        if self.use_cognitive_modules and self.cognitive_manager:
            try:
                self.cognitive_manager.detach_from_env()
                print("Cognitive modules detached from the environment")
            except Exception as e:
                print(f"Failed to detach cognitive modules: {e}")

        if return_collision_info:
            return trajectory, collision_info
        else:
            return trajectory

    def _update_cognitive_parameters(self, theta: Dict[str, float]):
        """Update cognitive-module parameters from one particle."""
        if not self.use_cognitive_modules:
            return

        if self._cognitive_args_template is not None:
            cognitive_args = copy.deepcopy(self._cognitive_args_template)
        else:
            cognitive_args = argparse.Namespace()
            cognitive_args.use_cognitive_modules = True
            cognitive_args.use_cognitive_bias = True
            cognitive_args.use_cognitive_delay = True
            cognitive_args.use_cognitive_perception = True

        perception_sigma0, perception_sigma_max = self._validate_perception_sigmas(theta)
        cognitive_args.bias_inverse_tta_coef = theta.get('bias_inverse_tta_coef', 1.5)
        cognitive_args.perception_sigma0 = perception_sigma0
        cognitive_args.perception_sigma_max = perception_sigma_max
        cognitive_args.delay_steps = int(theta.get('delay_steps', 2))

        cognitive_args.bias_tta_threshold = getattr(cognitive_args, 'bias_tta_threshold', 0.1)
        cognitive_args.bias_visual_distance = getattr(cognitive_args, 'bias_visual_distance', 300.0)
        cognitive_args.perception_use_kf = getattr(cognitive_args, 'perception_use_kf', True)
        cognitive_args.perception_kf_dt = getattr(cognitive_args, 'perception_kf_dt', 0.1)
        cognitive_args.perception_kf_q_scale = getattr(cognitive_args, 'perception_kf_q_scale', 100.0)
        self.cognitive_manager = CognitiveModuleManager(
            args=copy.deepcopy(cognitive_args)
        )
        if hasattr(self, 'network_signature') and hasattr(self.cognitive_manager, 'set_network_signature'):
            self.cognitive_manager.set_network_signature(self.network_signature)

        if self.ppo_model is not None:
            current_obs_dim = getattr(self.ppo_model, 'obs_dim', None)
            expected_obs_dim = None
            if hasattr(self, 'network_signature'):
                expected_obs_dim = self.network_signature.get('obs_dim')
            if expected_obs_dim is None:
                expected_obs_dim = current_obs_dim

            if current_obs_dim != expected_obs_dim:
                raise RuntimeError(
                    "network observation dimension does not match the checkpoint signature: "
                    f"current={current_obs_dim}, expected={expected_obs_dim}"
                )
            else:
                if self.action_processor:
                    self.action_processor.update_cognitive_manager(self.cognitive_manager)

    def parallel_rollout(self,
                        theta_list: List[Dict[str, float]],
                        initial_state: Dict[str, float],
                        num_steps: int,
                        start_timestamp: Optional[float] = None,
                        num_workers: int = 2,
                        rollout_seeds: Optional[List[int]] = None) -> List[List[TrajectoryPoint]]:
        """Run multiple rollouts in parallel."""
        from multiprocessing import Pool

        if rollout_seeds is None:
            rollout_seeds = [None] * len(theta_list)
        if len(rollout_seeds) != len(theta_list):
            raise ValueError("rollout_seeds must match theta_list length")

        tasks = []
        for theta, rollout_seed in zip(theta_list, rollout_seeds):
            tasks.append((theta, initial_state, num_steps, start_timestamp, rollout_seed))

        with Pool(num_workers) as pool:
            trajectories = pool.starmap(self._rollout_worker, tasks)

        return trajectories

    def _rollout_worker(self, theta: Dict[str, float],
                       initial_state: Dict[str, float],
                       num_steps: int,
                       start_timestamp: Optional[float],
                       rollout_seed: Optional[int]) -> List[TrajectoryPoint]:
        """Worker entry point for parallel rollout."""
        try:
            from metadrive.engine.engine_utils import close_engine, engine_initialized
            if engine_initialized():
                close_engine()
        except Exception as e:
            pass

        if self.use_cognitive_modules and hasattr(self, 'cognitive_manager') and self.cognitive_manager:
            import argparse
            cognitive_args = argparse.Namespace()

            original_args = self.cognitive_manager.args
            for attr in dir(original_args):
                if not attr.startswith('_'):
                    setattr(cognitive_args, attr, getattr(original_args, attr))

        else:
            cognitive_args = None

        worker_manager = self.__class__(
            env_config=self.env_config,
            ppo_model_path=self.ppo_model_path,
            use_cognitive_modules=self.use_cognitive_modules,
            device=self.device,
            cognitive_args=cognitive_args
        )

        parent_trajectory_dict = getattr(self, 'trajectory_dict', None)
        if parent_trajectory_dict is not None:
            worker_manager.trajectory_dict = parent_trajectory_dict
        parent_scenario_path = getattr(self, 'scenario_path', None)
        if parent_scenario_path is not None:
            worker_manager.scenario_path = parent_scenario_path

        return worker_manager.rollout(
            theta=theta,
            initial_state=initial_state,
            num_steps=num_steps,
            start_timestamp=start_timestamp,
            deterministic=True,
            seed=rollout_seed,
        )

    def load_original_data(self, data_path: str):
        """Load raw trajectory data and cache the ego trajectory."""
        loader = TrajectoryLoader(verbose=False)
        self.scenario_path = data_path
        self.full_data = loader.read_raw_csv(data_path)

        trajectory_dict = loader.to_trajectory_dict_original(self.full_data)

        vehicle_info = loader.get_vehicle_info(trajectory_dict)

        self.trajectory_dict = trajectory_dict

        main_vehicle_data = self.full_data[self.full_data['vehicle_id'] == -1]
        self.original_data = main_vehicle_data
        self.original_timestamps = self.original_data['timestamp'].values

    def get_original_trajectory(self, start_idx: int, num_steps: int) -> List[TrajectoryPoint]:
        """Extract one trajectory segment from the cached raw data."""
        if self.original_data is None:
            raise ValueError("raw trajectory data has not been loaded")

        trajectory = []

        for i in range(start_idx, min(start_idx + num_steps, len(self.original_data))):
            row = self.original_data.iloc[i]

            point = TrajectoryPoint(
                px=row['position_x'],
                py=row['position_y'],
                vx=row['speed_x'],
                vy=row['speed_y'],
                timestamp=row['timestamp'],
                yaw=None
            )
            trajectory.append(point)

        return trajectory
