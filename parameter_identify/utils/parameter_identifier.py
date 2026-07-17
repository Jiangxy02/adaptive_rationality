#!/usr/bin/env python3
"""Particle-filter-based cognitive parameter identification."""


import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
import logging
from datetime import datetime
# The non-interactive backend avoids Tk conflicts in multiprocessing workers.
import matplotlib
matplotlib.use('Agg')  # Must be selected before importing pyplot.
import matplotlib.pyplot as plt
from tqdm import tqdm

from common.random_seed import SeedDomain, derive_seed
# Keep the identification CLI readable by suppressing verbose cognitive-module logs.
logging.getLogger('cognitive_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_bias_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_perception_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_delay_module').setLevel(logging.WARNING)

from metadrive.obs import observation_base
from parameter_identify.utils.particle_manager import (
    DEFAULT_EVOLUTION_NOISE_RATIO,
    ParticleManager,
    Particle,
    validate_evolution_noise_ratio,
)
from parameter_identify.utils.adaptive_particle_manager import AdaptiveParticleManager
from parameter_identify.utils.likelihood_calculator import LikelihoodCalculator, TrajectoryPoint
from parameter_identify.utils.visualization import create_trajectory_visualizer, TrajectoryVisualizer
from parameter_identify.utils.parameter_time_sync import ParameterIdentificationTimeSynchronizer
from parameter_identify.utils.trajectory_visualization import TrajectoryVisualizationManager
from parameter_identify.utils.getmetrics import GetMetrics
from parameter_identify.utils.save_results import SaveIntermediateResults, DataSave
from parameter_identify.utils.dataloader import DataLoad

from parameter_identify.utils.simulation_manager import SimulationManager
COMPLEX_SIM_AVAILABLE = True


def independent_prediction_worker(queue, env_config, ppo_model_path,
                                use_cognitive_modules, best_theta,
                                initial_state, num_steps, start_timestamp):
    """Prediction worker that runs in a separate process."""
    if use_cognitive_modules:
        import argparse
        cognitive_args = argparse.Namespace()
        cognitive_args.use_cognitive_modules = True
        cognitive_args.use_cognitive_bias = True
        cognitive_args.use_cognitive_delay = True
        cognitive_args.use_cognitive_perception = True
        cognitive_args.bias_tta_threshold = 0.1
        cognitive_args.bias_visual_distance = 300.0
        cognitive_args.perception_use_kf = True
        cognitive_args.perception_kf_dt = 0.1
        cognitive_args.perception_kf_q_scale = 100.0
        cognitive_args.enable_radar_beam_viz = False
        for param_name, param_value in best_theta.items():
            setattr(cognitive_args, param_name, param_value)
    else:
        cognitive_args = None

    from parameter_identify.utils.simulation_manager import SimulationManager
    prediction_manager = SimulationManager(
        env_config=env_config,
        ppo_model_path=ppo_model_path,
        use_cognitive_modules=use_cognitive_modules,
        cognitive_args=cognitive_args
    )

    predicted_trajectory, collision_info = prediction_manager.rollout(
        theta=best_theta,
        initial_state=initial_state,
        num_steps=num_steps,
        start_timestamp=start_timestamp,
        deterministic=False,
        return_collision_info=True,
        if_prediction=True
    )

    if hasattr(prediction_manager, 'env') and prediction_manager.env is not None:
        prediction_manager.env.close()

    result_data = {
        'trajectory': predicted_trajectory,
        'collision_info': collision_info
    }
    queue.put(('success', result_data))


def direct_prediction_function(env_config, ppo_model_path,
                              use_cognitive_modules, theta,
                              initial_state, num_steps, start_timestamp,
                              trajectory_dict=None,
                              stochastic: bool = True,
                              sample_seed: Optional[int] = None):
    """Direct prediction helper used instead of the multiprocessing variant."""

    policy_seed = None
    perception_seed = None

    if sample_seed is not None:
        sample_seed = int(sample_seed)
        policy_seed = derive_seed(sample_seed, SeedDomain.POLICY)
        perception_seed = derive_seed(sample_seed, SeedDomain.PERCEPTION)

    if use_cognitive_modules:
        import argparse
        cognitive_args = argparse.Namespace()
        cognitive_args.use_cognitive_modules = True
        cognitive_args.use_cognitive_bias = True
        cognitive_args.use_cognitive_delay = True
        cognitive_args.use_cognitive_perception = True
        cognitive_args.bias_tta_threshold = 0.1
        cognitive_args.bias_visual_distance = 300.0
        cognitive_args.perception_use_kf = True
        cognitive_args.perception_kf_dt = 0.1
        cognitive_args.perception_kf_q_scale = 100.0
        cognitive_args.perception_random_seed = perception_seed
        cognitive_args.enable_radar_beam_viz = False
        for param_name, param_value in theta.items():
            setattr(cognitive_args, param_name, param_value)
    else:
        cognitive_args = None

    from parameter_identify.utils.simulation_manager import SimulationManager
    prediction_manager = SimulationManager(
        env_config=env_config,
        ppo_model_path=ppo_model_path,
        use_cognitive_modules=use_cognitive_modules,
        cognitive_args=cognitive_args
    )

    if trajectory_dict is not None:
        prediction_manager.trajectory_dict = trajectory_dict
        print(f"Prediction manager received trajectory data for {len(trajectory_dict)} vehicles")
    else:
        print("No trajectory data provided; the prediction environment will not include background vehicles")

    try:
        prediction_manager.load_environment()
        if hasattr(prediction_manager.env, 'vehicle_manager'):
            vehicle_count = len(prediction_manager.env.vehicle_manager.all_vehicle_data)
            print(f"Prediction environment check: background_vehicle_count={vehicle_count}")
            if vehicle_count == 0:
                print("Warning: the prediction environment contains no background vehicles")
        else:
            print("Warning: the prediction environment has no vehicle_manager")
    except Exception as e:
        print(f"Prediction environment validation failed: {e}")

    predicted_trajectory, collision_info = prediction_manager.rollout(
        theta=theta,
        initial_state=initial_state,
        num_steps=num_steps,
        start_timestamp=start_timestamp,
        deterministic=not stochastic,
        return_collision_info=True,
        if_prediction=True,
        seed=sample_seed,
    )

    if hasattr(prediction_manager, 'env') and prediction_manager.env is not None:
        prediction_manager.env.close()

    result_data = {
        'trajectory': predicted_trajectory,
        'collision_info': collision_info,
        'policy_seed': policy_seed,
        'perception_seed': perception_seed,
        'sample_seed': sample_seed
    }
    return ('success', result_data)



class ParameterIdentifier:
    """Identify cognitive parameters with a particle filter."""

    def __init__(self,
                 env_config: Dict[str, Any],
                 ppo_model_path: str,
                 num_particles: int = 128,
                 window_size: int = 20,
                 step_interval: int = 1,
                 horizon: int = 0,
                 sigma_diag: Optional[List[float]] = None,
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 use_cognitive_modules: bool = False,
                 num_workers: int = 4,
                 output_dir: str = './results',
                 enable_trajectory_visualization: bool = False,
                 visualization_interval: int = 1,
                 enable_prediction: bool = False,
                 pred_win: int = 10,
                 trajectory_duration: Optional[float] = None,
                 enable_comprehensive_visualization: bool = True,
                 use_geometric_mean_likelihood: bool = False,
                 predict_with_all_particles: bool = False,
                 mc_samples: int = 1,
                 mc_seed_mode: str = "sequence",
                 mc_use_all_particles: bool = False,
                 enable_multi_traj_metrics: bool = False,
                 metric_k: int = 5,
                 miss_epsilon: float = 2.0,
                 collision_info_csv_path: Optional[str] = None,
                 data_path: Optional[str] = None,
                 seed: Optional[int] = None):

        """Initialize the particle-filter identification pipeline.

        Args:
            env_config: Simulation environment configuration.
            ppo_model_path: Path to the PPO checkpoint.
            num_particles: Number of particles in the posterior approximation.
            window_size: Sliding observation-window size in samples.
            step_interval: Number of samples between identification windows.
            horizon: Look-ahead horizon in samples.
            sigma_diag: Diagonal of the observation-noise covariance.
            param_bounds: Lower and upper bounds for each identified parameter.
            use_cognitive_modules: Whether cognitive effects are active in rollouts.
            num_workers: Number of parallel rollout workers.
            output_dir: Directory for identification artifacts.
            enable_trajectory_visualization: Whether to render trajectory comparisons.
            visualization_interval: Identification-window interval between renders.
            enable_prediction: Whether to run future-trajectory prediction.
            pred_win: Future-prediction window size.
            trajectory_duration: Optional duration limit, in seconds.
            enable_comprehensive_visualization: Whether to render combined outputs.
            use_geometric_mean_likelihood: Whether to temper likelihoods by length.
            predict_with_all_particles: Use all particles instead of only the MAP one.
        """
        self.num_particles = num_particles
        self.window_size = window_size
        self.step_interval = step_interval
        self.horizon = horizon
        self.num_workers = num_workers
        self.root_seed = int(env_config.get("seed", 0) if seed is None else seed)
        self.use_geometric_mean_likelihood = use_geometric_mean_likelihood
        self.use_cognitive_modules = use_cognitive_modules
        self.output_dir = output_dir

        self.enable_prediction = enable_prediction
        self.pred_win = pred_win
        self.predict_with_all_particles = predict_with_all_particles
        self.trajectory_duration = trajectory_duration
        self.enable_multi_traj_metrics = enable_multi_traj_metrics
        self.metric_k = int(metric_k)
        self.miss_epsilon = float(miss_epsilon)

        self.mc_samples = max(1, int(mc_samples))
        self.mc_seed_mode = mc_seed_mode
        self.mc_use_all_particles = mc_use_all_particles
        self._mc_seed_rng = np.random.default_rng(
            derive_seed(self.root_seed, SeedDomain.MC_RANDOM)
        )

        self.enable_comprehensive_visualization = enable_comprehensive_visualization

        self.trajectory_visualizer = create_trajectory_visualizer(
            output_dir=output_dir,
            enable_visualization=enable_trajectory_visualization,
            visualization_interval=visualization_interval
        )

        self.visualization_manager = TrajectoryVisualizationManager(output_dir)

        self.time_synchronizer = ParameterIdentificationTimeSynchronizer(
            simulation_time_step=0.1
        )

        os.makedirs(output_dir, exist_ok=True)

        self.particle_manager = AdaptiveParticleManager(
            num_particles=num_particles,
            param_bounds=param_bounds,
            evolution_noise=DEFAULT_EVOLUTION_NOISE_RATIO,
            target_diversity_ratio=0.6,
            min_variance_ratio=0.2,
            seed=derive_seed(self.root_seed, SeedDomain.PARTICLE)
        )

        self.likelihood_calculator = LikelihoodCalculator(
            sigma_diag=sigma_diag,
            use_yaw=False
        )

        self.getmetrics = GetMetrics(
            sigma_diag=sigma_diag,
            use_geometric_mean_likelihood=use_geometric_mean_likelihood,
            collision_info_csv_path=collision_info_csv_path
        )

        self.data_path = data_path

        if use_cognitive_modules:
            import argparse
            cognitive_args = argparse.Namespace()
            cognitive_args.use_cognitive_modules = True
            cognitive_args.use_cognitive_bias = True
            cognitive_args.use_cognitive_delay = True
            cognitive_args.use_cognitive_perception = True
            cognitive_args.bias_inverse_tta_coef = 1.5
            cognitive_args.bias_tta_threshold = 0.1
            cognitive_args.bias_visual_distance = 300.0
            cognitive_args.perception_sigma0 = 0.1
            cognitive_args.perception_sigma_max = 0.8
            cognitive_args.perception_use_kf = True
            cognitive_args.perception_kf_dt = 0.1
            cognitive_args.perception_kf_q_scale = 100.0
            cognitive_args.delay_steps = 2
            cognitive_args.enable_radar_beam_viz = False
        else:
            cognitive_args = None

        self.simulation_manager = SimulationManager(
            env_config=env_config,
            ppo_model_path=ppo_model_path,
            use_cognitive_modules=use_cognitive_modules,
            cognitive_args=cognitive_args
        )

        self.estimation_history = []
        self.particle_history = []

        self.prediction_history = []
        self.original_trajectory_segments = []

        self.nll_history = []
        self.rmse_history = []

        self.data_save = DataSave(output_dir=output_dir)

    def identify_parameters(self,
                           observed_trajectory: List[TrajectoryPoint],
                           initialization_method: str = 'lhs',
                           resampling_threshold: float = 0.3,
                           evolution_noise: float = DEFAULT_EVOLUTION_NOISE_RATIO,
                           save_interval: int = 10) -> Dict[str, Any]:
        """Run parameter identification over the observed trajectory."""
        evolution_noise = validate_evolution_noise_ratio(evolution_noise)

        print(f"Initializing {self.num_particles} particles...")
        particles = self.particle_manager.initialize_particles_with_diversity_tracking(initialization_method)
        timing_info = self.time_synchronizer.analyze_trajectory_timing(observed_trajectory)
        print("Trajectory timing characteristics:")
        print(f"   Observation frequency: {timing_info['frequency']:.1f} Hz")
        print(f"   Mean interval: {timing_info['avg_interval']:.4f} s")
        print(f"   Total duration: {timing_info['total_duration']:.2f} s")

        window_duration = self.window_size * self.time_synchronizer.simulation_time_step
        step_duration = self.step_interval * self.time_synchronizer.simulation_time_step

        time_aligned_windows = self.time_synchronizer.create_time_aligned_windows(
            observed_trajectory,
            window_duration=window_duration,
            step_duration=step_duration
        )

        if not time_aligned_windows:
            raise ValueError("unable to create valid time-aligned windows")

        total_windows = len(time_aligned_windows)
        print(f"Starting time-synchronized parameter identification with {total_windows} windows")
        print(f"   Window duration: {window_duration:.2f} s")
        print(f"   Sliding step: {step_duration:.2f} s (identify every {self.step_interval} steps)")

        previous_mean = None

        for k, window_trajectory in enumerate(tqdm(time_aligned_windows, desc="Parameter identification progress")):
            obs_window = self.time_synchronizer.create_simulation_aligned_trajectory(
                window_trajectory,
                num_steps=self.window_size
            )

            window_duration_actual = window_trajectory[-1].timestamp - window_trajectory[0].timestamp
            print(f"\nProcessing window {k + 1}/{total_windows}")
            print(f"   Time range: {window_trajectory[0].timestamp:.3f} - {window_trajectory[-1].timestamp:.3f} s")
            print(f"   Window duration: {window_duration_actual:.3f} s")
            print(f"   Original points: {len(window_trajectory)} -> aligned points: {len(obs_window)}")

            if len(obs_window) < self.window_size:
                print(f"Window {k + 1} has too few aligned points; skipping")
                continue

            if evolution_noise > 0:
                particles = self.particle_manager.adaptive_evolve_particles(
                    particles,
                    window_idx=k,
                    previous_mean=previous_mean,
                    noise_scale=evolution_noise
                )
            log_likelihoods, predicted_trajectories = self._compute_particle_likelihoods(
                particles, obs_window, k, k + 1
            )

            particles = self.particle_manager.update_weights(particles, log_likelihoods)

            current_stats = self.particle_manager.get_posterior_statistics(particles)

            window_nll, window_rmse = self.getmetrics._perform_evaluation_metrics(
                particles,
                predicted_trajectories,
                obs_window,
                observed_trajectory,
                k,
                total_windows,
            )
            self.nll_history.extend(window_nll)
            self.rmse_history.extend(window_rmse)

            if not hasattr(self, 'particles_history'):
                self.particles_history = []

            particles_before_resample = [
                {
                    'particle_idx': i,
                    'weight': float(p.weight),
                    'log_weight': float(p.log_weight),
                    'theta': {k: float(v) for k, v in p.theta.items()}
                }
                for i, p in enumerate(particles)
            ]

            ess = self.particle_manager.compute_ess(particles)

            resampled = False
            if ess < resampling_threshold * self.num_particles:
                print(f"\nESS is low ({ess:.1f}/{self.num_particles}); running hybrid resampling...")
                particles = self.particle_manager.hybrid_resample(
                    particles,
                    previous_mean=previous_mean,
                    preserve_diversity_ratio=0.3
                )
                resampled = True

                theta_values = np.array([[p.theta[key] for key in sorted(p.theta.keys())]
                                       for p in particles[:5]])
                print("Post-resampling diversity among the first five particles:")
                for i, theta_array in enumerate(theta_values):
                    print(f"   Particle {i+1}: {theta_array}")

            previous_mean = current_stats['mean'].copy()

            stats = current_stats
            stats['window_idx'] = k
            stats['window_start_time'] = window_trajectory[0].timestamp
            stats['window_end_time'] = window_trajectory[-1].timestamp
            stats['window_duration'] = window_duration_actual
            stats['original_points'] = len(window_trajectory)
            stats['aligned_points'] = len(obs_window)

            particles_info = {
                'window_idx': k,
                'window_start_time': window_trajectory[0].timestamp,
                'window_end_time': window_trajectory[-1].timestamp,
                'resampled': resampled,
                'particles': particles_before_resample
            }
            self.particles_history.append(particles_info)

            self.estimation_history.append(stats)

            if self.enable_prediction:
                prediction_data = self._perform_future_prediction_independent(
                    particles, obs_window, window_trajectory,
                    observed_trajectory, k, total_windows
                )
                if_collision = prediction_data.get('collision_info', {}).get('collision_detected', False)

                if (self.enable_comprehensive_visualization and
                    self.trajectory_visualizer.enable_visualization and
                    prediction_data):
                    current_timestamp = obs_window[-1].timestamp
                    background_vehicles_data = self.visualization_manager.get_background_vehicles_data(
                        self.simulation_manager, current_timestamp
                    )

                    self.visualization_manager.create_comprehensive_visualization(
                        obs_window=obs_window,
                        predicted_trajectories=predicted_trajectories,
                        log_likelihoods=log_likelihoods,
                        prediction_data=prediction_data,
                        window_idx=k,
                        window_start=k,
                        current_timestamp=current_timestamp,
                        if_collision=if_collision,
                        trajectory_visualizer=self.trajectory_visualizer,
                        background_vehicles_data=background_vehicles_data,
                        pred_win=self.pred_win,
                        predict_with_all_particles=self.predict_with_all_particles,
                        simulation_manager=self.simulation_manager
                    )

            if (k + 1) % save_interval == 0:
                self.data_save._save_intermediate_results(k + 1, self.estimation_history)


        final_stats = self.particle_manager.get_posterior_statistics(particles)

        self.data_save._generate_report(final_stats,
                                       num_particles=self.num_particles,
                                       window_size=self.window_size,
                                       step_interval=self.step_interval,
                                       horizon=self.horizon)

        if self.enable_prediction and self.prediction_history:
            print(f"Generating prediction-summary visualization for {len(self.prediction_history)} prediction times...")
            self.visualization_manager.create_prediction_summary_visualization(
                self.prediction_history, self.simulation_manager, self.pred_win
            )

            self.visualization_manager.create_prediction_statistics_report(
                self.prediction_history, self.pred_win
            )

            if self.enable_multi_traj_metrics:
                try:
                    self.visualization_manager.export_multi_traj_metrics_to_csv(
                        self.prediction_history, self.pred_win
                    )
                except Exception as e:
                    print(f"Failed to export multi-trajectory metrics CSV: {e}")
                    import traceback
                    traceback.print_exc()
        elif self.enable_multi_traj_metrics:
            print("Multi-trajectory metrics are enabled, but prediction is disabled or no prediction history is available")
            print("   Hint: enable --enable_prediction to compute and export multi-trajectory metrics")

        print(f"Debug info - NLL history entries: {len(self.nll_history) if self.nll_history else 0}")
        print(f"Debug info - RMSE history entries: {len(self.rmse_history) if self.rmse_history else 0}")
        print(f"Debug info - cognitive modules enabled: {self.use_cognitive_modules}")

        if self.nll_history or self.rmse_history:
            print("Generating NLL and RMSE plots...")
            print(f"   NLL points: {len(self.nll_history) if self.nll_history else 0}")
            print(f"   RMSE points: {len(self.rmse_history) if self.rmse_history else 0}")

            nll_rmse_path = self.trajectory_visualizer.visualize_nll_and_rmse(
                self.nll_history, self.rmse_history
            )

            if nll_rmse_path:
                print(f"NLL and RMSE plots saved to: {nll_rmse_path}")
        else:
            print("No NLL or RMSE data is available for plotting")

        return {
            'final_estimate': final_stats,
            'history': self.estimation_history,
            'final_particles': particles,
            'particles_history': getattr(self, 'particles_history', []),
            'prediction_history': self.prediction_history if self.enable_prediction else None,
            'nll_history': self.nll_history if self.nll_history else None,
            'rmse_history': self.rmse_history if self.rmse_history else None
        }

    def _compute_particle_likelihoods(self,
                                    particles: List[Particle],
                                    obs_window: List[TrajectoryPoint],
                                    window_start: int,
                                    window_idx: int) -> Tuple[np.ndarray, List[List[TrajectoryPoint]]]:
        """Compute likelihoods for every particle, in parallel when configured."""
        initial_state = {
            'px': obs_window[0].px,
            'py': obs_window[0].py,
            'vx': obs_window[0].vx,
            'vy': obs_window[0].vy
        }

        start_timestamp = obs_window[0].timestamp

        theta_list = [p.theta for p in particles]
        rollout_seeds = [
            self._generate_rollout_seed(window_start, particle_index)
            for particle_index in range(len(theta_list))
        ]

        if self.num_workers > 1:
            predicted_trajectories = self.simulation_manager.parallel_rollout(
                theta_list=theta_list,
                initial_state=initial_state,
                num_steps=self.window_size + self.horizon,
                start_timestamp=start_timestamp,
                num_workers=self.num_workers,
                rollout_seeds=rollout_seeds,
            )
        else:
            predicted_trajectories = []
            for particle_index, theta in enumerate(theta_list):
                traj = self.simulation_manager.rollout(
                    theta=theta,
                    initial_state=initial_state,
                    num_steps=self.window_size + self.horizon,
                    start_timestamp=start_timestamp,
                    seed=rollout_seeds[particle_index],
                )
                predicted_trajectories.append(traj)

        log_likelihoods = np.zeros(len(particles))

        for i, pred_traj in enumerate(predicted_trajectories):
            pred_window = pred_traj[:self.window_size]

            if len(pred_window) < len(obs_window):
                scenario = getattr(self, 'data_path', None) or 'unknown scenario'
                raise RuntimeError(
                    f"particle {i} returned {len(pred_window)} usable prediction "
                    f"steps for a {len(obs_window)}-step observation window "
                    f"(window {window_idx}, start index {window_start}, "
                    f"scenario {scenario}); refusing to pad fabricated states "
                    "into the likelihood"
                )

            likelihood_result = self.likelihood_calculator.compute_trajectory_likelihood(
                obs_window, pred_window, use_geometric_mean=self.use_geometric_mean_likelihood
            )

            log_likelihoods[i] = -likelihood_result['nll']

        if self.trajectory_visualizer.should_visualize():
            self.trajectory_visualizer.visualize_trajectories(
                observed_trajectory=obs_window,
                predicted_trajectories=predicted_trajectories,
                log_likelihoods=log_likelihoods,
                window_idx=window_idx,
                window_start=window_start
            )
        return log_likelihoods, predicted_trajectories


    def _generate_rollout_seed(self, window_index: int, particle_index: int) -> int:
        """Derive a task seed independent of process scheduling and worker count."""
        return derive_seed(
            self.root_seed,
            SeedDomain.WINDOW,
            window_index,
            SeedDomain.WORKER,
            particle_index,
            SeedDomain.PARTICLE,
            particle_index,
        )

    def _generate_mc_seed(
        self, window_index: int, particle_index: int, sample_index: int
    ) -> int:
        """Generate the random seed used for one Monte Carlo sample."""
        context = (
            SeedDomain.WINDOW,
            window_index,
            SeedDomain.PARTICLE,
            particle_index,
            SeedDomain.SAMPLE,
            sample_index,
        )
        if self.mc_seed_mode == 'sequence':
            return derive_seed(self.root_seed, *context)

        random_draw = int(self._mc_seed_rng.integers(1, 2 ** 31 - 1))
        return derive_seed(self.root_seed, SeedDomain.MC_RANDOM, random_draw, *context)

    def _perform_future_prediction_independent(self,
                                 particles: List[Particle],
                                 obs_window: List[TrajectoryPoint],
                                 window_trajectory: List[TrajectoryPoint],
                                 observed_trajectory: List[TrajectoryPoint],
                                 window_idx: int,
                                 total_windows: int):
        """Predict future trajectories in an isolated environment."""
        import os
        import tempfile
        import pickle

        prediction_start_state = {
            'px': obs_window[-1].px,
            'py': obs_window[-1].py,
            'vx': obs_window[-1].vx,
            'vy': obs_window[-1].vy
        }

        prediction_start_timestamp = obs_window[-1].timestamp

        original_future_segment = self.getmetrics._get_original_trajectory_segment(
            observed_trajectory, prediction_start_timestamp, self.pred_win
        )

        weights = [p.weight for p in particles]
        if not weights:
            print("No particles are available for Monte Carlo prediction")
            prediction_data = {
                'window_idx': window_idx,
                'prediction_start_time': prediction_start_timestamp,
                'prediction_start_state': prediction_start_state.copy(),
                'best_theta': {},
                'predicted_trajectory': [],
                'collision_info': None,
                'original_trajectory_segment': original_future_segment,
                'best_particle_weight': None,
                'window_end_time': window_trajectory[-1].timestamp if window_trajectory else None,
                'mc_trajectories': [],
                'policy_seed': None,
                'perception_seed': None,
                'sample_seed': None
            }
        else:
            best_particle_idx = int(np.argmax(weights))
            best_particle = particles[best_particle_idx]
            use_all_particles = self.predict_with_all_particles or self.mc_use_all_particles
            target_indices = list(range(len(particles))) if use_all_particles else [best_particle_idx]

            if use_all_particles:
                print(f"Running all-particle prediction mode with {len(target_indices)} particles and {self.mc_samples} Monte Carlo samples")
            else:
                print(f"Running Monte Carlo prediction with the best particle for {self.mc_samples} samples")

            all_predicted_trajectories = []
            all_collision_infos = []
            all_thetas = []
            all_weights = []
            mc_trajectories = []

            valid_sample_records = []
            for rank, particle_idx in enumerate(target_indices, start=1):
                particle = particles[particle_idx]
                particle_samples = []
                best_sample_record = None

                for sample_idx in range(self.mc_samples):
                    print('sample_idx:', sample_idx)
                    sample_seed = self._generate_mc_seed(
                        window_idx, particle_idx, sample_idx
                    )
                    status, result_data = direct_prediction_function(
                        self.simulation_manager.env_config,
                        self.simulation_manager.ppo_model_path,
                        self.simulation_manager.use_cognitive_modules,
                        particle.theta,
                        prediction_start_state,
                        self.pred_win,
                        prediction_start_timestamp,
                        getattr(self.simulation_manager, 'trajectory_dict', {}),
                        stochastic=True,
                        sample_seed=sample_seed
                    )

                    if status == 'success':
                        trajectory = result_data.get('trajectory', [])
                        collision_info = result_data.get('collision_info')
                        policy_seed = result_data.get('policy_seed')
                        perception_seed = result_data.get('perception_seed')
                        resolved_sample_seed = result_data.get('sample_seed', sample_seed)
                        print(
                            f"Particle {rank}/{len(target_indices)} Monte Carlo sample {sample_idx + 1}/{self.mc_samples} succeeded with {len(trajectory)} steps"
                        )
                    else:
                        trajectory = []
                        collision_info = None
                        policy_seed = None
                        perception_seed = None
                        resolved_sample_seed = sample_seed
                        print(
                            f"Particle {rank}/{len(target_indices)} Monte Carlo sample {sample_idx + 1} failed: {result_data}"
                        )

                    sample_record = {
                        'sample_index': sample_idx,
                        'status': status,
                        'sample_seed': resolved_sample_seed,
                        'policy_seed': policy_seed,
                        'perception_seed': perception_seed,
                        'trajectory': trajectory,
                        'collision_info': collision_info
                    }
                    if status != 'success':
                        sample_record['error'] = result_data

                    particle_samples.append(sample_record)

                    if status == 'success' and best_sample_record is None:
                        best_sample_record = sample_record
                        valid_sample_records.append(sample_record)
                    elif status == 'success':
                        valid_sample_records.append(sample_record)

                if best_sample_record is None and particle_samples:
                    best_sample_record = particle_samples[-1]

                best_sample_index = (
                    particle_samples.index(best_sample_record)
                    if best_sample_record in particle_samples
                    else None
                )
                best_summary = {
                    'trajectory': best_sample_record.get('trajectory', []) if best_sample_record else [],
                    'collision_info': best_sample_record.get('collision_info') if best_sample_record else None,
                    'policy_seed': best_sample_record.get('policy_seed') if best_sample_record else None,
                    'perception_seed': best_sample_record.get('perception_seed') if best_sample_record else None,
                    'sample_seed': best_sample_record.get('sample_seed') if best_sample_record else None
                }

                mc_entry = {
                    'particle_index': particle_idx,
                    'particle_rank': rank - 1,
                    'particle_weight': particle.weight,
                    'theta': particle.theta.copy(),
                    'samples': particle_samples,
                    'best_sample_index': best_sample_index,
                    'best_sample': best_summary
                }
                mc_trajectories.append(mc_entry)

                if use_all_particles:
                    all_predicted_trajectories.append(best_summary['trajectory'])
                    all_collision_infos.append(best_summary['collision_info'])
                    all_thetas.append(particle.theta.copy())
                    all_weights.append(particle.weight)
            mc_stats = None
            cluster_labels = None
            if valid_sample_records:
                sample_positions = []
                for rec in valid_sample_records:
                    traj_points = rec.get('trajectory', [])
                    coords = np.array([[pt.px, pt.py] for pt in traj_points], dtype=np.float32)
                    if coords.size == 0:
                        continue
                    sample_positions.append(coords)

                if sample_positions:
                    max_len = max(len(coords) for coords in sample_positions)
                    pos_tensor = np.full((len(sample_positions), max_len, 2), np.nan, dtype=np.float32)
                    for idx, coords in enumerate(sample_positions):
                        used_len = min(len(coords), max_len)
                        pos_tensor[idx, :used_len, :] = coords[:used_len]

                    mean_xy = np.nanmean(pos_tensor, axis=0)
                    std_xy = np.nanstd(pos_tensor, axis=0)

                    mc_stats = {
                        'positions': pos_tensor.tolist(),
                        'mean_xy': mean_xy.tolist(),
                        'std_xy': std_xy.tolist(),
                        'effective_samples': len(sample_positions),
                        'time_step': 0.1
                    }
            for entry in mc_trajectories:
                best_idx = entry.get('best_sample_index')
                if best_idx is not None and 0 <= best_idx < len(entry['samples']):
                    best_record = entry['samples'][best_idx]
                    entry['best_sample']['cluster_label'] = best_record.get('cluster_label')

            best_mc_entry = (
                next(
                    (entry for entry in mc_trajectories if entry['particle_index'] == best_particle_idx),
                    mc_trajectories[0] if mc_trajectories else None
                )
            )
            best_sample = best_mc_entry['best_sample'] if best_mc_entry else {
                'trajectory': [],
                'collision_info': None,
                'policy_seed': None,
                'perception_seed': None,
                'sample_seed': None,
                'cluster_label': None
            }

            prediction_data = {
                'window_idx': window_idx,
                'prediction_start_time': prediction_start_timestamp,
                'prediction_start_state': prediction_start_state.copy(),
                'best_particle_index': best_particle_idx,
                'best_theta': best_particle.theta.copy(),
                'predicted_trajectory': best_sample.get('trajectory', []),
                'collision_info': best_sample.get('collision_info'),
                'original_trajectory_segment': original_future_segment,
                'best_particle_weight': best_particle.weight,
                'window_end_time': window_trajectory[-1].timestamp,
                'mc_trajectories': mc_trajectories,
                'policy_seed': best_sample.get('policy_seed'),
                'perception_seed': best_sample.get('perception_seed'),
                'sample_seed': best_sample.get('sample_seed'),
                'cluster_label': best_sample.get('cluster_label'),
                'mc_stats': mc_stats,
                'mc_cluster_labels': cluster_labels
            }

            if use_all_particles:
                prediction_data.update({
                    'all_predicted_trajectories': all_predicted_trajectories,
                    'all_collision_infos': all_collision_infos,
                    'all_thetas': all_thetas,
                    'all_weights': all_weights
                })

            total_success = sum(
                1 for entry in mc_trajectories for sample in entry['samples'] if sample['status'] == 'success'
            )
            total_samples = len(mc_trajectories) * self.mc_samples if mc_trajectories else 0
            if total_samples > 0:
                print(f"Monte Carlo sampling complete: successful samples {total_success}/{total_samples}")

        if getattr(self, 'enable_multi_traj_metrics', False):
            try:
                candidate_trajs: List[List[TrajectoryPoint]] = []
                source = None

                if prediction_data.get('all_predicted_trajectories') and prediction_data.get('all_weights'):
                    trajs = prediction_data['all_predicted_trajectories']
                    weights = prediction_data['all_weights']
                    if len(trajs) == len(weights) and len(weights) > 0:
                        sorted_idx = np.argsort(np.array(weights, dtype=np.float32))[::-1]
                    else:
                        sorted_idx = np.arange(len(trajs))
                    for idx in sorted_idx.tolist():
                        if idx < 0 or idx >= len(trajs):
                            continue
                        t = trajs[idx]
                        if t and len(t) >= 2:
                            candidate_trajs.append(t)
                    source = 'all_particles_topk'

                if not candidate_trajs and prediction_data.get('mc_trajectories'):
                    mc_entries = prediction_data['mc_trajectories']
                    best_particle_index = prediction_data.get('best_particle_index')
                    entry = mc_entries[0] if mc_entries else None
                    if best_particle_index is not None:
                        for e in mc_entries:
                            if e.get('particle_index') == best_particle_index:
                                entry = e
                                break
                    if entry is None:
                        entry = mc_entries[0]
                    samples = entry.get('samples', [])
                    for s in samples:
                        if s.get('status') != 'success':
                            continue
                        t = s.get('trajectory', [])
                        if t and len(t) >= 2:
                            candidate_trajs.append(t)
                    source = 'mc_best_particle'

                if not candidate_trajs and prediction_data.get('predicted_trajectory'):
                    candidate_trajs = [prediction_data['predicted_trajectory']]
                    source = 'single_best'

                data_filename = getattr(self, 'data_path', None)
                sim_collision_info = prediction_data.get('collision_info')
                gt_likelihood_info = prediction_data.get('gt_likelihood_info')
                mc_stats = prediction_data.get('mc_stats')

                if not gt_likelihood_info or not gt_likelihood_info.get('valid'):
                    if mc_stats and isinstance(mc_stats, dict):
                        mean_xy = mc_stats.get('mean_xy')
                        std_xy = mc_stats.get('std_xy')
                        original_trajectory_segment = prediction_data.get('original_trajectory_segment')

                        if mean_xy and std_xy and original_trajectory_segment:
                            try:
                                mean_xy_arr = np.asarray(mean_xy, dtype=np.float32)
                                std_xy_arr = np.asarray(std_xy, dtype=np.float32)
                                if hasattr(self, 'trajectory_visualizer'):
                                    gt_likelihood_info = self.trajectory_visualizer.compute_gt_log_likelihood(
                                        mean_xy_arr, std_xy_arr, original_trajectory_segment
                                    )
                                    prediction_data['gt_likelihood_info'] = gt_likelihood_info
                            except Exception as e:
                                pass

                metrics = self.getmetrics.compute_minade_minfde_missrate_at_k(
                    predicted_trajectories=candidate_trajs,
                    gt_trajectory=prediction_data.get('original_trajectory_segment'),
                    k=getattr(self, 'metric_k', 5),
                    miss_epsilon=getattr(self, 'miss_epsilon', 2.0),
                    data_filename=data_filename,
                    simulation_collision_info=sim_collision_info,
                    gt_likelihood_info=gt_likelihood_info,
                    mc_stats=mc_stats
                )
                if metrics is not None:
                    metrics['source'] = source or 'unknown'
                    prediction_data['multi_traj_metrics'] = metrics
            except Exception as e:
                print(f"Failed to compute multi-trajectory prediction metrics: {e}")

        if prediction_data.get('collision_info'):
            ppo_collision = prediction_data['collision_info'].get('collision_detected', False)
            linear_collision = False

            print("\nCollision-detection comparison:")
            print(f"   PPO prediction: {'collision' if ppo_collision else 'safe'}")
            print(f"   Linear-motion prediction: {'collision' if linear_collision else 'safe'}")

            if ppo_collision and linear_collision:
                ppo_time = prediction_data['collision_info']['collision_time'] - prediction_start_timestamp
                linear_time = 0
                print(f"   Both methods predict collision; time difference: {abs(ppo_time - linear_time):.3f}s")
            elif ppo_collision != linear_collision:
                if ppo_collision:
                    print("   PPO predicts a collision while linear motion stays safe; PPO may be capturing evasive behavior")
                else:
                    print("   Linear motion predicts a collision while PPO stays safe; PPO may be modeling avoidance")
            else:
                print("   Both methods agree")

        self.prediction_history.append(prediction_data)

        print(
            f"Independent prediction for window {window_idx + 1}/{total_windows} completed; "
            f"predicted {self.pred_win} future steps from time {prediction_start_timestamp:.3f}s"
        )
        return prediction_data


    def _get_background_vehicles_data(self,
                                     current_timestamp: float) -> Optional[List[Dict]]:
        """Get one nearest-position sample per background vehicle at one timestamp."""
        if not hasattr(self.simulation_manager, 'full_data') or self.simulation_manager.full_data is None:
            return None

        full_data = self.simulation_manager.full_data
        background_data = full_data[full_data['vehicle_id'] != -1]

        if background_data.empty:
            return None

        background_vehicles = []
        for vehicle_id in background_data['vehicle_id'].unique():
            vehicle_data = background_data[background_data['vehicle_id'] == vehicle_id]
            time_diffs = abs(vehicle_data['timestamp'] - current_timestamp)
            closest_idx = time_diffs.idxmin()
            closest_row = vehicle_data.loc[closest_idx]
            if time_diffs.loc[closest_idx] <= 0.5:
                background_vehicles.append({
                    'vehicle_id': int(vehicle_id),
                    'positions': [(
                        closest_row['position_x'],
                        closest_row['position_y'],
                        closest_row['timestamp']
                    )]
                })

        return background_vehicles if background_vehicles else None


    def _perform_linear_collision_detection(self, obs_window, pred_win, trajectory_dict):
        """Run a simplified linear-motion collision check for comparison."""
        import math

        last_point = obs_window[-1]
        ego_px, ego_py = last_point.px, last_point.py
        ego_vx, ego_vy = last_point.vx, last_point.vy
        start_timestamp = last_point.timestamp

        dt = 0.1
        collision_radius = 1.5

        collision_info = {
            'collision_detected': False,
            'collision_step': None,
            'collision_time': None,
            'collision_vehicle_id': None,
            'ego_trajectory': [],
            'collision_details': []
        }

        print("Starting simplified linear-motion collision detection:")
        print(f"   Initial position: ({ego_px:.2f}, {ego_py:.2f})")
        print(f"   Initial velocity: ({ego_vx:.2f}, {ego_vy:.2f}) m/s")
        print(f"   Prediction steps: {pred_win}")
        print(f"   Collision radius: {collision_radius}m")

        for step in range(pred_win):
            current_time = start_timestamp + step * dt
            ego_x = ego_px + ego_vx * step * dt
            ego_y = ego_py + ego_vy * step * dt

            collision_info['ego_trajectory'].append({
                'step': step,
                'time': current_time,
                'px': ego_x,
                'py': ego_y
            })

            step_collision_info = {
                'step': step,
                'time': current_time,
                'ego_pos': (ego_x, ego_y),
                'background_vehicles': [],
                'min_distance': float('inf'),
                'collision_detected': False
            }

            if trajectory_dict:
                for vehicle_id, vehicle_trajectory in trajectory_dict.items():
                    if vehicle_id == -1:
                        continue
                    bg_vehicle_state = self._get_background_vehicle_state_at_time(
                        vehicle_trajectory, current_time
                    )

                    if bg_vehicle_state is not None:
                        bg_x = bg_vehicle_state.get('px', bg_vehicle_state.get('x', 0))
                        bg_y = bg_vehicle_state.get('py', bg_vehicle_state.get('y', 0))
                        distance = math.sqrt((ego_x - bg_x)**2 + (ego_y - bg_y)**2)

                        step_collision_info['background_vehicles'].append({
                            'vehicle_id': vehicle_id,
                            'pos': (bg_x, bg_y),
                            'distance': distance
                        })

                        step_collision_info['min_distance'] = min(
                            step_collision_info['min_distance'], distance
                        )

                        if distance < collision_radius:
                            step_collision_info['collision_detected'] = True

                            if not collision_info['collision_detected']:
                                collision_info['collision_detected'] = True
                                collision_info['collision_step'] = step
                                collision_info['collision_time'] = current_time
                                collision_info['collision_vehicle_id'] = vehicle_id

                                print(f"Simplified collision detection: collision detected at step {step}")
                                print(f"   Collision time: {current_time:.3f}s")
                                print(f"   Ego position: ({ego_x:.2f}, {ego_y:.2f})")
                                print(f"   Background vehicle {vehicle_id} position: ({bg_x:.2f}, {bg_y:.2f})")
                                print(f"   Collision distance: {distance:.2f}m")

            collision_info['collision_details'].append(step_collision_info)
            if collision_info['collision_detected']:
                break

        if collision_info['collision_detected']:
            time_to_collision = collision_info['collision_time'] - start_timestamp
            print("Simplified collision detection complete: collision detected")
            print(f"   Collision step: {collision_info['collision_step']}")
            print(f"   Time to collision from prediction start: {time_to_collision:.3f}s")
            print(f"   Colliding vehicle ID: {collision_info['collision_vehicle_id']}")
        else:
            print("Simplified collision detection complete: no collision")
            print(f"   Predicted trajectory steps: {len(collision_info['ego_trajectory'])}")
            if collision_info['collision_details']:
                min_distance = min(detail['min_distance'] for detail in collision_info['collision_details']
                                 if detail['min_distance'] != float('inf'))
                if min_distance != float('inf'):
                    print(f"   Minimum distance: {min_distance:.2f}m")

        return collision_info

    def _get_background_vehicle_state_at_time(self, vehicle_trajectory, target_time):
        """Get the background-vehicle state closest to one timestamp."""
        if not vehicle_trajectory:
            return None

        if isinstance(vehicle_trajectory, list) and len(vehicle_trajectory) > 0:
            if isinstance(vehicle_trajectory[0], dict):
                time_field = None
                if 'timestamp' in vehicle_trajectory[0]:
                    time_field = 'timestamp'
                elif 'original_timestamp' in vehicle_trajectory[0]:
                    time_field = 'original_timestamp'
                elif 'time' in vehicle_trajectory[0]:
                    time_field = 'time'

                if time_field:
                    closest_point = None
                    min_time_diff = float('inf')

                    for point in vehicle_trajectory:
                        time_diff = abs(point[time_field] - target_time)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_point = point

                    return closest_point
        return None
