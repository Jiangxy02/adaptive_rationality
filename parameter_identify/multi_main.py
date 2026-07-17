#!/usr/bin/env python3


import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cli import add_boolean_argument
from common.headless import apply_headless_guard
from common.random_seed import seed_global_generators
apply_headless_guard()

import logging
import pandas as pd
import numpy as np
import csv
from datetime import datetime
from typing import Dict, Any, List, Tuple

# argparse bool helper
def str2bool(value):
    if isinstance(value, bool):
        return value
    value_str = str(value).lower()
    if value_str in {"yes", "true", "t", "1", "y"}:
        return True
    if value_str in {"no", "false", "f", "0", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def evolution_noise_ratio(value):
    """Parse the normalized evolution-noise ratio for command-line callers."""
    try:
        return validate_evolution_noise_ratio(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


try:
    from  parameter_identify.utils.fix_overflow_error import apply_all_fixes
    apply_all_fixes()
except ImportError:
    print("Overflow-fix module not found; continuing...")


logging.getLogger('cognitive_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_bias_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_perception_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_delay_module').setLevel(logging.WARNING)


from  parameter_identify.utils.parameter_identifier import ParameterIdentifier
from parameter_identify.utils.particle_manager import (
    DEFAULT_EVOLUTION_NOISE_RATIO,
    perception_sigma_domain_has_positive_probability,
    validate_evolution_noise_ratio,
)
from parameter_identify.utils.trajectory_loader import TrajectoryLoader
from parameter_identify.utils.util import (
    load_config,
    save_config,
    create_env_config,
    generate_experiment_name,
    export_results,
    validate_trajectory_data
)
from parameter_identify.utils.output_transaction import experiment_transaction
from parameter_identify.utils.likelihood_calculator import (
    LikelihoodCalculator,
    validate_sigma_value,
)


def likelihood_sigma(value):
    """Parse one positive, finite observation-noise standard deviation."""
    try:
        return validate_sigma_value(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _validate_perception_sigma_domain(args) -> None:
    """Validate non-empty paper domain sigma0 <= sigma_max."""
    if args.perception_sigma0_min < 0.0 or args.perception_sigma_max_min < 0.0:
        raise ValueError("perception sigma bounds must be non-negative")
    if args.perception_sigma0_min > args.perception_sigma0_max:
        raise ValueError("perception_sigma0_min must be <= perception_sigma0_max")
    if args.perception_sigma_max_min > args.perception_sigma_max_max:
        raise ValueError("perception_sigma_max_min must be <= perception_sigma_max_max")
    if not perception_sigma_domain_has_positive_probability(
        (args.perception_sigma0_min, args.perception_sigma0_max),
        (args.perception_sigma_max_min, args.perception_sigma_max_max),
    ):
        raise ValueError(
            "perception sigma domain has no positive-probability sample "
            "satisfying sigma0 <= sigma_max"
        )


def _ensure_checkpoint_file_exists(checkpoint_path: str) -> None:
    """Fail before creating outputs when the requested PPO checkpoint is missing."""
    if not checkpoint_path:
        print("\nError: --ppo_model_path is required.")
        print("Run `python ppo_train/scripts/train.py` first, then pass the generated `checkpoints/latest_model.pt` path.")
        sys.exit(2)

    if Path(checkpoint_path).expanduser().is_file():
        return

    print("\nError: the PPO checkpoint file does not exist.")
    print(f"Current --ppo_model_path: {checkpoint_path}")
    print("Suggested fix: provide your own checkpoint path, or run `python ppo_train/scripts/train.py` first to produce a checkpoint.")
    print("After training, you can usually use `checkpoints/latest_model.pt` from the output directory as --ppo_model_path.")
    sys.exit(2)


class ScenarioDataAnalyzer:

    def __init__(self, data_dir: str, max_files: int = 200):
        self.data_dir = data_dir
        self.max_files = max_files
        self.scenario_files = []
        self.analysis_results = []

    def find_scenario_files(self) -> List[str]:
        print(f"Searching for trajectory CSV files under {self.data_dir}...")

        valid_scenario_files = []
        all_csv_files = sorted(Path(self.data_dir).glob("*.csv"))
        total_files = len(all_csv_files)

        print(f"Found {total_files} CSV files in total")

        print("Validating file contents...")
        for i, csv_file in enumerate(all_csv_files):
            if i % 50 == 0:
                print(f"   Validation progress: {i}/{total_files}")

            if self._validate_file_data(csv_file):
                valid_scenario_files.append(str(csv_file))
            else:
                print(f"File format or data did not meet requirements; skipping: {os.path.basename(csv_file)}")

        print(f"Files that passed validation: {len(valid_scenario_files)}/{total_files}")

        if not valid_scenario_files:
            print("No valid trajectory CSV files were found")
            return []

        print(
            f"Scenarios will be attempted from the sorted candidate pool until {self.max_files} are processed successfully"
        )
        self.scenario_files = valid_scenario_files
        return valid_scenario_files

    def _validate_file_data(self, csv_file: str, min_duration: float = 3.0, min_points: int = 30) -> bool:

        return TrajectoryLoader().validate_scene_csv(csv_file, min_duration, min_points)

    def detect_linear_collision_times(
        self,
        observed_trajectory,
        trajectory_dict: Dict[int, List[Dict]],
        pred_win: int = 50,
    ) -> Dict[str, Any]:
        collision_detections = []
        background_vehicle_ids = sorted(
            int(vehicle_id)
            for vehicle_id in (trajectory_dict or {})
            if int(vehicle_id) != -1
        )
        if not background_vehicle_ids:
            return {
                'status': 'not_computable',
                'reason': 'no_background_vehicles',
                'vehicle_ids': [],
                'detections': [],
            }

        background_state_index = {}
        for vehicle_id, vehicle_trajectory in trajectory_dict.items():
            vehicle_id = int(vehicle_id)
            if vehicle_id == -1:
                continue
            for state in vehicle_trajectory:
                state_timestamp = state.get('timestamp', state.get('original_timestamp'))
                if state_timestamp is None:
                    continue
                timestamp_key = round(float(state_timestamp), 6)
                background_state_index.setdefault(timestamp_key, {}).setdefault(
                    vehicle_id, state
                )

        participating_vehicle_ids = set()


        for start_idx in range(0, len(observed_trajectory) - 5, 5):
            obs_window = observed_trajectory[start_idx:start_idx + 5]
            if len(obs_window) < 5:
                continue


            collision_info = self._perform_simple_linear_collision_detection(
                obs_window, background_state_index, pred_win
            )
            participating_vehicle_ids.update(collision_info['vehicle_ids'])

            if collision_info['collision_detected']:
                collision_detections.append({
                    'detection_start_time': obs_window[-1].timestamp,
                    'predicted_collision_time': collision_info['collision_time'],
                    'time_to_collision': collision_info['collision_time'] - obs_window[-1].timestamp,
                    'collision_step': collision_info['collision_step'],
                    'collision_vehicle_id': collision_info['collision_vehicle_id'],
                    'vehicle_ids': collision_info['vehicle_ids'],
                })

        if not participating_vehicle_ids:
            return {
                'status': 'not_computable',
                'reason': 'no_background_vehicle_state_at_detection_timestamps',
                'vehicle_ids': [],
                'detections': [],
            }

        return {
            'status': 'computed',
            'reason': '',
            'vehicle_ids': sorted(participating_vehicle_ids),
            'detections': collision_detections,
        }

    def _perform_simple_linear_collision_detection(
        self,
        obs_window,
        background_state_index: Dict[float, Dict[int, Dict]],
        pred_win: int,
    ) -> Dict:
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
            'vehicle_ids': [],
        }


        background_vehicles = []
        states_at_start = background_state_index.get(round(float(start_timestamp), 6), {})
        for vehicle_id, state in states_at_start.items():
            required_fields = ('x', 'y', 'speed_x', 'speed_y')
            missing_fields = [field for field in required_fields if field not in state]
            if missing_fields:
                raise ValueError(
                    f"Background vehicle {vehicle_id} is missing constant-velocity extrapolation fields at timestamp {start_timestamp}: {missing_fields}"
                )
            background_vehicles.append({
                'vehicle_id': vehicle_id,
                'px': float(state['x']),
                'py': float(state['y']),
                'vx': float(state['speed_x']),
                'vy': float(state['speed_y']),
            })

        collision_info['vehicle_ids'] = sorted(
            vehicle['vehicle_id'] for vehicle in background_vehicles
        )


        for step in range(pred_win):
            current_time = start_timestamp + step * dt


            ego_x = ego_px + ego_vx * step * dt
            ego_y = ego_py + ego_vy * step * dt


            for bg_vehicle in background_vehicles:
                bg_x = bg_vehicle['px'] + bg_vehicle['vx'] * step * dt
                bg_y = bg_vehicle['py'] + bg_vehicle['vy'] * step * dt
                distance = math.sqrt((ego_x - bg_x)**2 + (ego_y - bg_y)**2)

                if distance < collision_radius:
                    collision_info['collision_detected'] = True
                    collision_info['collision_step'] = step
                    collision_info['collision_time'] = current_time
                    collision_info['collision_vehicle_id'] = bg_vehicle['vehicle_id']
                    return collision_info

        return collision_info


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multi-scenario trajectory cognitive-parameter identification and analysis tool"
    )


    parser.add_argument(
        '--data_dir',
        type=str,
        default='examples/trajectory_scenario',
        help='Trajectory CSV directory; each file must exactly match the public six-column example format'
    )

    parser.add_argument(
        '--max_scenario_files',
        type=int,
        default=200,
        help='Target number of scenario CSV files to process successfully (failed files do not count, default: 200)'
    )

    parser.add_argument(
        '--trajectory_duration',
        type=float,
        default=10,
        help='Trajectory duration in seconds (default: 10)'
    )

    parser.add_argument(
        '--ppo_model_path',
        type=str,
        default=None,
        help='Path to a trained PPO model checkpoint (required when running identification)'
    )


    parser.add_argument(
        '--num_particles',
        type=int,
        default=5,
        help='Number of particles (default: 5)'
    )

    parser.add_argument(
        '--window_size',
        type=int,
        default=5,
        help='Sliding-window size (default: 5)'
    )

    parser.add_argument(
        '--step_interval',
        type=int,
        default=1,
        help='Step interval between rolling windows (default: 1)'
    )

    parser.add_argument(
        '--evolution_noise',
        type=evolution_noise_ratio,
        default=DEFAULT_EVOLUTION_NOISE_RATIO,
        help=(
            'Dimensionless ratio of continuous parameter-evolution noise std over the full parameter range '
            f'(default: {DEFAULT_EVOLUTION_NOISE_RATIO})'
        )
    )


    add_boolean_argument(
        parser,
        '--enable_prediction',
        default=True,
        help_text='Enable the prediction module (enabled by default)',
    )

    parser.add_argument(
        '--pred_win',
        type=int,
        default=20,
        help='Prediction-window size (default: 20)'
    )

    parser.add_argument(
        '--predict_with_all_particles',
        action='store_true',
        default=False,
        help='Whether to use all particles for prediction'
    )

    add_boolean_argument(
        parser,
        '--enable_multi_traj_metrics',
        default=True,
        help_text='Enable future multi-trajectory metrics (MinADE@K/MinFDE@K/MissRate@K, enabled by default)',
    )

    parser.add_argument(
        '--metric_k',
        type=int,
        default=10,
        help='K for multi-trajectory metrics (computed on the top-K candidate trajectories, default: 10)'
    )

    parser.add_argument(
        '--miss_epsilon',
        type=float,
        default=2.0,
        help='MissRate@K threshold epsilon in meters (default: 2.0)'
    )


    parser.add_argument(
        '--num_workers',
        type=int,
        default=10,
        help='Number of parallel worker processes (default: 10)'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=0,
        help='Lookahead horizon in steps (default: 0)'
    )


    parser.add_argument(
        '--sigma_px',
        type=likelihood_sigma,
        default=0.1,
        help='Standard deviation for x position (default: 0.1 m)'
    )

    parser.add_argument(
        '--sigma_py',
        type=likelihood_sigma,
        default=0.1,
        help='Standard deviation for y position (default: 0.1 m)'
    )

    parser.add_argument(
        '--sigma_vx',
        type=likelihood_sigma,
        default=0.5,
        help='Standard deviation for x velocity (default: 0.5 m/s)'
    )

    parser.add_argument(
        '--sigma_vy',
        type=likelihood_sigma,
        default=0.5,
        help='Standard deviation for y velocity (default: 0.5 m/s)'
    )


    parser.add_argument(
        '--perception_sigma0_min',
        type=float,
        default=0.0,
        help='Minimum perception-noise sigma0'
    )

    parser.add_argument(
        '--perception_sigma0_max',
        type=float,
        default=1.0,
        help='Maximum perception-noise sigma0'
    )

    parser.add_argument(
        '--perception_sigma_max_min',
        type=float,
        default=0.0,
        help='Minimum perception-noise sigma_max'
    )

    parser.add_argument(
        '--perception_sigma_max_max',
        type=float,
        default=5.0,
        help='Maximum perception-noise sigma_max'
    )

    parser.add_argument(
        '--bias_coef_min',
        type=float,
        default=0.5,
        help='Minimum cognitive-bias coefficient'
    )

    parser.add_argument(
        '--bias_coef_max',
        type=float,
        default=10.0,
        help='Maximum cognitive-bias coefficient'
    )

    parser.add_argument(
        '--delay_steps_min',
        type=int,
        default=0,
        help='Minimum delay steps'
    )

    parser.add_argument(
        '--delay_steps_max',
        type=int,
        default=20,
        help='Maximum delay steps'
    )

    add_boolean_argument(
        parser,
        '--use_cognitive_modules',
        default=True,
        help_text='Enable cognitive effects (enabled by default; disabling still keeps the network at 283 dims)',
    )


    parser.add_argument(
        '--cognitive_modulation',
        type=str,
        default='auto',
        choices=['auto', 'none', 'concat'],
        help='Cognitive modulation mode (auto uses checkpoint config, default: auto)'
    )


    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/multi_results_without_cog',
        help='Output directory'
    )

    parser.add_argument(
        '--enable_trajectory_visualization',
        action='store_true',
        default=False,
        help='Whether to enable trajectory visualization'
    )


    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Whether to print verbose output'
    )

    args = parser.parse_args()
    _validate_perception_sigma_domain(args)
    return args


def setup_environment(args) -> Dict[str, Any]:
    env_config = create_env_config(
        data_path=None,
        enable_render=False,
        ppo_checkpoint_path=args.ppo_model_path,
        use_cognitive_modules=args.use_cognitive_modules,
        seed=args.seed
    )
    return env_config


def setup_param_bounds(args) -> Dict[str, tuple]:
    _validate_perception_sigma_domain(args)
    param_bounds = {
        'perception_sigma0': (args.perception_sigma0_min, args.perception_sigma0_max),
        'perception_sigma_max': (args.perception_sigma_max_min, args.perception_sigma_max_max),
        'bias_inverse_tta_coef': (args.bias_coef_min, args.bias_coef_max),
        'delay_steps': (args.delay_steps_min, args.delay_steps_max)
    }
    return param_bounds


def _failed_scenario_result(scenario_file, exc):
    return {
        'csv_name': os.path.basename(scenario_file),
        'simulation_collision_times': [],
        'linear_collision_times': [],
        'linear_collision_status': 'not_computed',
        'linear_collision_reason': 'identification failed',
        'linear_collision_vehicle_ids': [],
        'identified_parameters': {},
        'multi_traj_metrics': None,
        'error_message': f"{type(exc).__name__}: {exc}",
        'status': 'failed',
    }


def _scenario_output_name(sequence_number, scenario_file):
    safe_stem = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(scenario_file).stem).strip('._')
    return f"{int(sequence_number):04d}_{safe_stem or 'scenario'}"


def run_scenario_with_isolation(env_config, scenario_file, analyzer, args,
    scenario_output_dir, identifier_kwargs):
    """Run one scenario in an owned output transaction and return a summary row."""
    result = None
    identifier = None
    try:
        with experiment_transaction(
            scenario_output_dir, scenario=scenario_file
        ) as working_dir:
            try:
                identifier = ParameterIdentifier(
                    output_dir=working_dir,
                    **identifier_kwargs,
                )
                result = process_single_scenario_file(
                    env_config, scenario_file, identifier, analyzer, args
                )
                if result.get('status') != 'completed':
                    raise RuntimeError(
                        result.get('error_message') or 'scenario did not complete'
                    )
            finally:
                manager = getattr(identifier, 'simulation_manager', None)
                if manager is not None:
                    manager.close()
        return result
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(
            f"Scenario {os.path.basename(scenario_file)} failed, "
            f"and the failure snapshot was kept at {scenario_output_dir}.failed: "
            f"{type(exc).__name__}: {exc}"
        )
        if result is None:
            return _failed_scenario_result(scenario_file, exc)
        result['status'] = 'failed'
        if not result.get('error_message'):
            result['error_message'] = f"{type(exc).__name__}: {exc}"
        return result


def process_single_scenario_file(
    env_config,
    file_path: str,
    identifier: ParameterIdentifier,
    analyzer: ScenarioDataAnalyzer,
    args
) -> Dict:

    print(f"\n{'='*80}")
    print(f"Processing trajectory scenario: {os.path.basename(file_path)}")
    print(f"{'='*80}")

    result = {
        'csv_name': os.path.basename(file_path),
        'simulation_collision_times': [],
        'linear_collision_times': [],
        'linear_collision_status': 'pending',
        'linear_collision_reason': '',
        'linear_collision_vehicle_ids': [],
        'identified_parameters': {},
        'multi_traj_metrics': None,
        'error_message': None,
        'status': 'pending'
    }


    print("Loading observation data...")
    from parameter_identify.utils.dataloader import DataLoad
    dataload = DataLoad()

    observed_trajectory, trajectory_dict, timestamps, full_data, main_vehicle_data = dataload.load_trajectory_data(
        file_path,
        trajectory_duration=args.trajectory_duration
    )


    if args.use_cognitive_modules:
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


    from parameter_identify.utils.simulation_manager import SimulationManager
    identifier.simulation_manager = SimulationManager(
        env_config=env_config,
        ppo_model_path=args.ppo_model_path,
        use_cognitive_modules=args.use_cognitive_modules,
        cognitive_args=cognitive_args
    )


    identifier.simulation_manager.trajectory_dict = trajectory_dict
    identifier.simulation_manager.full_data = full_data
    identifier.simulation_manager.original_data = main_vehicle_data
    identifier.simulation_manager.original_timestamps = timestamps
    identifier.simulation_manager.scenario_path = file_path


    identifier.data_path = file_path


    try:
        _ = identifier.simulation_manager.load_ppo_model()
        net_sig = getattr(identifier.simulation_manager, 'network_signature', {})
        if net_sig:
            print("\nLoaded PPO network (from common):")
            print(f"   Input dimension: {net_sig.get('obs_dim')} (base {net_sig.get('base_obs_dim')})")
            print(f"   Cognitive modulation: {net_sig.get('cognitive_modulation')}")
            print(f"   Cognitive-parameter dimension: {net_sig.get('cognitive_param_dim')}")
    except Exception as e:
        print(f"Failed to preload the PPO network: {e}")

    if not observed_trajectory or len(observed_trajectory) < 10:
        result['error_message'] = "Insufficient observation data"
        result['status'] = 'failed'
        return result

    print(f"Successfully loaded {len(observed_trajectory)} data points")


    print("Starting parameter identification...")
    identification_results = identifier.identify_parameters(
        observed_trajectory=observed_trajectory,
        initialization_method='lhs',
        resampling_threshold=0.5,
        evolution_noise=args.evolution_noise,
        save_interval=50
    )

    if identification_results and 'final_estimate' in identification_results:
        result['identified_parameters'] = identification_results['final_estimate']['mean'].copy()


    print("Extracting simulation collision timestamps...")
    simulation_collision_times = []


    if hasattr(identifier, 'prediction_history') and identifier.prediction_history:
        for pred_data in identifier.prediction_history:
            if pred_data.get('collision_info') and pred_data['collision_info'].get('collision_detected'):
                collision_time = pred_data['collision_info'].get('collision_time')
                detection_start_time = pred_data.get('prediction_start_time')

                if collision_time and detection_start_time:
                    simulation_collision_times.append({
                        'detection_start_time': detection_start_time,
                        'predicted_collision_time': collision_time,
                        'time_to_collision': collision_time - detection_start_time
                    })

    result['simulation_collision_times'] = simulation_collision_times
    print(f"Simulation detected {len(simulation_collision_times)} predicted collisions")


    print("Running constant-velocity linear-motion collision detection...")
    linear_collision_result = analyzer.detect_linear_collision_times(
        observed_trajectory,
        trajectory_dict,
        pred_win=args.pred_win
    )
    result['linear_collision_times'] = linear_collision_result['detections']
    result['linear_collision_status'] = linear_collision_result['status']
    result['linear_collision_reason'] = linear_collision_result['reason']
    result['linear_collision_vehicle_ids'] = linear_collision_result['vehicle_ids']
    if linear_collision_result['status'] == 'computed':
        print(
            f"Linear-motion detection found {len(linear_collision_result['detections'])} predicted collisions, "
            f"with vehicle IDs: {linear_collision_result['vehicle_ids']}"
        )
    else:
        print(f"Linear-motion collision baseline is not computable: {linear_collision_result['reason']}")

    if args.enable_multi_traj_metrics and hasattr(identifier, 'prediction_history') and identifier.prediction_history:
        print("Extracting multi-trajectory metrics...")
        metrics_list = []
        for pred_data in identifier.prediction_history:
            metrics = pred_data.get('multi_traj_metrics')
            if metrics:
                metrics_list.append(metrics)

        if metrics_list:

            avg_metrics = {
                'mean_minADE': np.mean([m.get('minADE', float('nan')) for m in metrics_list if m.get('minADE') is not None]),
                'mean_minFDE': np.mean([m.get('minFDE', float('nan')) for m in metrics_list if m.get('minFDE') is not None]),
                'mean_miss_rate': np.mean([m.get('miss', float('nan')) for m in metrics_list if m.get('miss') is not None]),
                'mean_spread_pair': np.mean([m.get('spread_pair', float('nan')) for m in metrics_list if m.get('spread_pair') is not None and np.isfinite(m.get('spread_pair', float('nan')))]),
                'mean_avg_log_likelihood': np.mean([m.get('avg_log_likelihood', float('nan')) for m in metrics_list if m.get('avg_log_likelihood') is not None and np.isfinite(m.get('avg_log_likelihood', float('nan')))]),
                'mean_log_likelihood': np.mean([m.get('log_likelihood', float('nan')) for m in metrics_list if m.get('log_likelihood') is not None and np.isfinite(m.get('log_likelihood', float('nan')))]),
                'mean_total_variance': np.mean([m.get('total_variance', float('nan')) for m in metrics_list if m.get('total_variance') is not None and np.isfinite(m.get('total_variance', float('nan')))]),
                'total_windows': len(metrics_list),
                'k': metrics_list[0].get('k', args.metric_k) if metrics_list else args.metric_k,
                'miss_epsilon': metrics_list[0].get('epsilon', args.miss_epsilon) if metrics_list else args.miss_epsilon
            }
            result['multi_traj_metrics'] = avg_metrics
            spread_str = f"{avg_metrics['mean_spread_pair']:.4f}" if not np.isnan(avg_metrics['mean_spread_pair']) else "N/A"
            avgll_str = f"{avg_metrics['mean_avg_log_likelihood']:.4f}" if not np.isnan(avg_metrics['mean_avg_log_likelihood']) else "N/A"
            var_str = f"{avg_metrics['mean_total_variance']:.4f}" if not np.isnan(avg_metrics['mean_total_variance']) else "N/A"
            print(f"Multi-trajectory metrics: MinADE@K={avg_metrics['mean_minADE']:.4f}m, "
                  f"MinFDE@K={avg_metrics['mean_minFDE']:.4f}m, "
                  f"MissRate@K={avg_metrics['mean_miss_rate']:.4f}, "
                  f"Spread_pair={spread_str}m, "
                  f"AvgLL={avgll_str}, "
                  f"TotalVar={var_str}")
        else:
            print("No multi-trajectory metric data was found")

    result['status'] = 'completed'

    print(f"Finished processing {os.path.basename(file_path)}")


    return result


def save_metrics_summary(results: List[Dict], output_dir: str, csv_filepath: str = None):

    if csv_filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filepath = os.path.join(output_dir, f"multi_traj_metrics_summary_{timestamp}.csv")


    metrics_data = []
    for result in results:
        if result['status'] != 'completed':
            continue

        metrics = result.get('multi_traj_metrics')
        if not metrics:
            continue

        row = {
            'csv_name': result['csv_name'],
            'mean_minADE_at_K': metrics.get('mean_minADE', float('nan')),
            'mean_minFDE_at_K': metrics.get('mean_minFDE', float('nan')),
            'mean_miss_rate_at_K': metrics.get('mean_miss_rate', float('nan')),
            'mean_spread_pair': metrics.get('mean_spread_pair', float('nan')),
            'mean_avg_log_likelihood': metrics.get('mean_avg_log_likelihood', float('nan')),
            'mean_log_likelihood': metrics.get('mean_log_likelihood', float('nan')),
            'mean_total_variance': metrics.get('mean_total_variance', float('nan')),
            'total_windows': metrics.get('total_windows', 0),
            'k': metrics.get('k', 10),
            'miss_epsilon': metrics.get('miss_epsilon', 2.0)
        }
        metrics_data.append(row)

    if not metrics_data:

        df = pd.DataFrame(columns=[
            'csv_name', 'mean_minADE_at_K', 'mean_minFDE_at_K',
            'mean_miss_rate_at_K', 'mean_spread_pair',
            'mean_avg_log_likelihood', 'mean_log_likelihood', 'mean_total_variance',
            'total_windows', 'k', 'miss_epsilon'
        ])
        df.to_csv(csv_filepath, index=False, float_format='%.6f')
        return csv_filepath


    df = pd.DataFrame(metrics_data)


    if len(df) > 0:
        avg_row = {
            'csv_name': 'AVERAGE_ALL_SCENARIOS',
            'mean_minADE_at_K': df['mean_minADE_at_K'].mean(),
            'mean_minFDE_at_K': df['mean_minFDE_at_K'].mean(),
            'mean_miss_rate_at_K': df['mean_miss_rate_at_K'].mean(),
            'mean_spread_pair': df['mean_spread_pair'].mean(),
            'total_windows': df['total_windows'].sum(),
            'k': df['k'].iloc[0] if len(df) > 0 else 10,
            'miss_epsilon': df['miss_epsilon'].iloc[0] if len(df) > 0 else 2.0
        }
        avg_df = pd.DataFrame([avg_row])


        df = pd.concat([df, avg_df], ignore_index=True)


    df.to_csv(csv_filepath, index=False, float_format='%.6f')

    return csv_filepath


def save_analysis_results(results: List[Dict], output_dir: str):


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"scenario_analysis_results_{timestamp}.csv"
    csv_filepath = os.path.join(output_dir, csv_filename)

    print(f"\nSaving analysis results to: {csv_filepath}")


    csv_data = []

    for result in results:

        expected_result_fields = {
            'csv_name', 'simulation_collision_times',
            'linear_collision_times', 'linear_collision_status',
            'linear_collision_reason', 'linear_collision_vehicle_ids',
            'identified_parameters', 'multi_traj_metrics',
            'error_message', 'status'
        }
        unexpected_result_fields = set(result.keys()) - expected_result_fields
        if unexpected_result_fields:
            print(f"Result dictionary contains unexpected fields: {unexpected_result_fields} in file {result.get('csv_name', 'unknown')}")
            print(f"   All result fields: {list(result.keys())}")


        base_info = {
            'csv_name': result['csv_name'],
            'status': result['status'],
            'error_message': result.get('error_message', ''),
            'simulation_collision_count': len(result.get('simulation_collision_times', [])),
            'linear_collision_count': len(result.get('linear_collision_times', [])),
            'linear_collision_status': result.get('linear_collision_status', ''),
            'linear_collision_reason': result.get('linear_collision_reason', ''),
            'linear_collision_vehicle_ids': ';'.join(
                str(vehicle_id)
                for vehicle_id in result.get('linear_collision_vehicle_ids', [])
            ),
        }


        if result.get('simulation_collision_times'):
            sim_times = []
            for sim_collision in result['simulation_collision_times']:
                sim_times.append(f"{sim_collision['detection_start_time']:.3f}->{sim_collision['predicted_collision_time']:.3f}")
            base_info['simulation_collision_times'] = ';'.join(sim_times)
        else:
            base_info['simulation_collision_times'] = ''


        if result.get('linear_collision_times'):
            linear_times = []
            for linear_collision in result['linear_collision_times']:
                linear_times.append(
                    f"{linear_collision['detection_start_time']:.3f}->"
                    f"{linear_collision['predicted_collision_time']:.3f} "
                    f"V{linear_collision['collision_vehicle_id']}"
                )
            base_info['linear_collision_times'] = ';'.join(linear_times)
        else:
            base_info['linear_collision_times'] = ''


        if result.get('identified_parameters'):
            params = result['identified_parameters']
            base_info.update({
                'perception_sigma0': params.get('perception_sigma0', ''),
                'perception_sigma_max': params.get('perception_sigma_max', ''),
                'bias_inverse_tta_coef': params.get('bias_inverse_tta_coef', ''),
                'delay_steps': params.get('delay_steps', '')
            })
        else:
            base_info.update({
                'perception_sigma0': '',
                'perception_sigma_max': '',
                'bias_inverse_tta_coef': '',
                'delay_steps': ''
            })


        unexpected_fields = set(base_info.keys()) - {
            'csv_name', 'status', 'error_message',
            'simulation_collision_count', 'linear_collision_count',
            'simulation_collision_times', 'linear_collision_times',
            'linear_collision_status', 'linear_collision_reason', 'linear_collision_vehicle_ids',
            'perception_sigma0', 'perception_sigma_max', 'bias_inverse_tta_coef', 'delay_steps'
        }
        if unexpected_fields:
            print(f"Unexpected fields found: {unexpected_fields} in file {result['csv_name']}")
            print(f"   All base_info fields: {list(base_info.keys())}")

        csv_data.append(base_info)


    if csv_data:

        all_possible_fields = set()
        for record in csv_data:
            all_possible_fields.update(record.keys())

        fieldnames = sorted(list(all_possible_fields))
        print(f"Detected fields: {fieldnames}")

        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)

        print(f"Results were saved to {csv_filepath}")


        successful_count = sum(1 for r in results if r['status'] == 'completed')
        failed_count = len(results) - successful_count

        print("\nProcessing summary:")
        print(f"   Successfully processed: {successful_count}/{len(results)} files")
        print(f"   Failed: {failed_count}/{len(results)} files")

        if successful_count > 0:

            total_sim_collisions = sum(len(r.get('simulation_collision_times', [])) for r in results if r['status'] == 'completed')
            total_linear_collisions = sum(len(r.get('linear_collision_times', [])) for r in results if r['status'] == 'completed')

            print(f"   Simulation-detected collisions: {total_sim_collisions}")
            print(f"   Linear-motion-detected collisions: {total_linear_collisions}")

    return csv_filepath


def main():


    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    import os
    os.environ['METADRIVE_MULTIPROCESS_MODE'] = '1'


    args = parse_arguments()
    _ensure_checkpoint_file_exists(args.ppo_model_path)


    print("=" * 80)
    print("Multi-Scenario Trajectory Cognitive-Parameter Identification and Analysis Tool")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Maximum files to process successfully: {args.max_scenario_files}")
    print(f"Trajectory duration: {args.trajectory_duration if args.trajectory_duration else 'all'} s")
    print(f"PPO model: {args.ppo_model_path}")
    print(f"Particle count: {args.num_particles}")
    print(f"Window size: {args.window_size}")
    print(f"Worker count: {args.num_workers}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    seed_global_generators(args.seed)


    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        print(f"\nError: the data directory does not exist or is not a directory: {args.data_dir}")
        raise SystemExit(2)

    print("\nInitializing the scenario data analyzer...")
    analyzer = ScenarioDataAnalyzer(str(data_dir), args.max_scenario_files)


    scenario_files = analyzer.find_scenario_files()
    if not scenario_files:
        print("Error: no trajectory CSV matching the public six-column format was found")
        raise SystemExit(2)


    experiment_name = f"batch_scenario_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)


    args_dict = vars(args)
    save_config(args_dict, os.path.join(output_dir, 'config.json'))

    print(f"\nWill process {len(scenario_files)} trajectory scenario files")


    env_config = setup_environment(args)

    if getattr(args, 'cognitive_modulation', 'auto') != 'auto':
        env_config['cognitive_modulation_override'] = args.cognitive_modulation
    param_bounds = setup_param_bounds(args)


    sigma_diag = [args.sigma_px, args.sigma_py, args.sigma_vx, args.sigma_vy]


    analysis_results = []
    successful_count = 0
    processed_count = 0
    scenarios_dir = os.path.join(output_dir, 'scenarios')
    os.makedirs(scenarios_dir, exist_ok=True)

    print("\nStarting batch processing...")
    print(f"Target: process {args.max_scenario_files} files successfully")

    for scenario_file in scenario_files:

        if successful_count >= args.max_scenario_files:
            print(f"\nSuccessfully processed {successful_count} files; reached the target count, stopping")
            break

        processed_count += 1
        print(f"\nProgress: attempted {processed_count}, succeeded {successful_count}/{args.max_scenario_files}")

        identifier_kwargs = dict(
            env_config=env_config,
            ppo_model_path=args.ppo_model_path,
            num_particles=args.num_particles,
            window_size=args.window_size,
            step_interval=args.step_interval,
            horizon=args.horizon,
            sigma_diag=sigma_diag,
            param_bounds=param_bounds,
            use_cognitive_modules=args.use_cognitive_modules,
            num_workers=args.num_workers,
            enable_trajectory_visualization=args.enable_trajectory_visualization,
            visualization_interval=100,
            enable_prediction=args.enable_prediction,
            pred_win=args.pred_win,
            trajectory_duration=args.trajectory_duration,
            enable_comprehensive_visualization=False,
            predict_with_all_particles=args.predict_with_all_particles,
            use_geometric_mean_likelihood=True,
            enable_multi_traj_metrics=getattr(args, 'enable_multi_traj_metrics', True),
            metric_k=getattr(args, 'metric_k', 10),
            miss_epsilon=getattr(args, 'miss_epsilon', 2.0),
            collision_info_csv_path=None,
            data_path=scenario_file,
            seed=args.seed,
        )
        scenario_output_dir = os.path.join(
            scenarios_dir,
            _scenario_output_name(processed_count, scenario_file),
        )


        file_result = run_scenario_with_isolation(
            env_config,
            scenario_file,
            analyzer,
            args,
            scenario_output_dir,
            identifier_kwargs,
        )
        analysis_results.append(file_result)


        if file_result['status'] == 'completed':
            successful_count += 1
            print(f"File processed successfully. Current success count: {successful_count}/{args.max_scenario_files}")
        else:
            print(f"File processing failed: {file_result.get('error_message', 'unknown error')}")


        temp_csv_path = save_analysis_results(analysis_results, output_dir)
        print(f"Intermediate results saved: {temp_csv_path}")


        if args.enable_multi_traj_metrics:

            summary_csv_path = os.path.join(output_dir, 'multi_traj_metrics_summary.csv')
            summary_path = save_metrics_summary(analysis_results, output_dir, summary_csv_path)
            print(f"Multi-trajectory metric summary updated: {summary_path}")


        if processed_count >= len(scenario_files):
            print(f"\nProcessed all available trajectory scenarios ({len(scenario_files)} files)")
            print(f"   Successfully processed: {successful_count}/{args.max_scenario_files}")
            if successful_count < args.max_scenario_files:
                print("   Warning: the target success count was not reached; check data quality or parameter settings")
            break


    print("\nSaving final analysis results...")
    final_csv_path = save_analysis_results(analysis_results, output_dir)


    if args.enable_multi_traj_metrics:
        summary_csv_path = os.path.join(output_dir, 'multi_traj_metrics_summary.csv')
        summary_path = save_metrics_summary(analysis_results, output_dir, summary_csv_path)
        print(f"Final multi-trajectory metric summary saved: {summary_path}")

    print("\nMulti-scenario trajectory analysis completed.")
    print("Processing summary:")
    print(f"   Target successful files: {args.max_scenario_files}")
    print(f"   Files attempted: {processed_count}")
    print(f"   Files processed successfully: {successful_count}")
    print(f"   Failed files: {processed_count - successful_count}")
    print(f"   Success rate: {successful_count/processed_count*100:.1f}%" if processed_count > 0 else "   Success rate: 0%")
    print(f"Results saved under: {output_dir}")
    print(f"Detailed results: {final_csv_path}")

    if successful_count == 0:
        print("\nError: no scenario succeeded during batch processing; returning a non-zero exit code.")
        raise SystemExit(1)
    if successful_count >= args.max_scenario_files:
        print("\nTarget reached successfully. Program finished.")
    else:
        print("\nThe target was not fully reached, but all available files were processed. Program finished.")


if __name__ == "__main__":
    main()
