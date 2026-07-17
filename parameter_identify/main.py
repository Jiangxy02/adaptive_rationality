#!/usr/bin/env python3


import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.headless import apply_headless_guard
from common.random_seed import seed_global_generators
apply_headless_guard()

import argparse
import logging
from typing import Dict, Any


# Apply the MetaDrive overflow fix.

from parameter_identify.utils.fix_overflow_error import apply_all_fixes
apply_all_fixes()

# Reduce cognitive-module log noise.
logging.getLogger('cognitive_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_bias_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_perception_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_delay_module').setLevel(logging.WARNING)


from parameter_identify.utils.parameter_identifier import ParameterIdentifier
from parameter_identify.utils.dataloader import DataLoad
from parameter_identify.utils.trajectory_loader import TrajectoryLoader

from parameter_identify.utils.util import (
    load_config,
    save_config,
    create_env_config,
    generate_experiment_name,
    export_results,
    validate_trajectory_data
)
import pandas as pd

from parameter_identify.utils.output_transaction import experiment_transaction
from parameter_identify.utils.parser import parse_arguments


def _validate_perception_sigma_domain(args) -> None:
    """Validate non-empty paper domain sigma0 <= sigma_max."""
    if args.perception_sigma0_min < 0.0 or args.perception_sigma_max_min < 0.0:
        raise ValueError("perception sigma bounds must be non-negative")
    if args.perception_sigma0_min > args.perception_sigma0_max:
        raise ValueError("perception_sigma0_min must be <= perception_sigma0_max")
    if args.perception_sigma_max_min > args.perception_sigma_max_max:
        raise ValueError("perception_sigma_max_min must be <= perception_sigma_max_max")
    if args.perception_sigma0_min > args.perception_sigma_max_max:
        raise ValueError("empty perception sigma domain: require sigma0 <= sigma_max")


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


def _load_observation_data(data_path: str, trajectory_duration):
    """Load the user-selected trajectory without substituting another dataset."""
    try:
        return DataLoad().load_trajectory_data(
            data_path,
            trajectory_duration=trajectory_duration,
        )
    except Exception as exc:
        print("\nError: failed to load the requested trajectory data.", file=sys.stderr)
        print(f"Current --data_path: {data_path}", file=sys.stderr)
        print(f"Data error: {exc}", file=sys.stderr)
        print(
            "Check the file path and CSV format. The columns must exactly match the public example: "
            "timestamp, vehicle_id, speed_x, speed_y, position_x, position_y.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def setup_environment(args) -> Dict[str, Any]:
    if args.env_config:
        # Load config from file.
        env_config = load_config(args.env_config)
    else:
        # Create the default config.
        env_config = create_env_config(
            data_path=args.data_path,
            enable_render=args.enable_render,
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


def get_initial_timestamp_from_csv(csv_path: str) -> float:

    return TrajectoryLoader().get_initial_timestamp(csv_path)


def export_gt_likelihood_csv(
    prediction_history: list,
    output_dir: str,
    initial_timestamp: float,
    csv_filename: str = 'gt_likelihood_timeseries.csv'
):

    import numpy as np

    if not prediction_history:
        print("No prediction history is available for export")
        return

    csv_rows = []

    for pred_data in prediction_history:
        gt_likelihood_info = pred_data.get('gt_likelihood_info')
        if not gt_likelihood_info or not gt_likelihood_info.get('valid'):
            continue

        window_idx = pred_data.get('window_idx', -1)
        prediction_start_time = pred_data.get('prediction_start_time', 0.0)
        window_end_time = pred_data.get('window_end_time', 0.0)

        # Convert to relative time.
        relative_start_time = prediction_start_time - initial_timestamp
        relative_end_time = window_end_time - initial_timestamp if window_end_time else relative_start_time

        row = {
            'window_idx': window_idx,
            'relative_timestamp': relative_start_time,
            'absolute_timestamp': prediction_start_time,
            'window_end_time': relative_end_time,
            'log_likelihood': gt_likelihood_info.get('log_likelihood'),
            'avg_log_likelihood': gt_likelihood_info.get('avg_log_likelihood'),
            'nll': gt_likelihood_info.get('nll'),
            'avg_nll': gt_likelihood_info.get('avg_nll'),
            'num_steps': gt_likelihood_info.get('num_steps'),
            'num_pred_steps': gt_likelihood_info.get('num_pred_steps'),
            'num_gt_steps': gt_likelihood_info.get('num_gt_steps'),
            'mean_error_x': gt_likelihood_info.get('mean_error_x'),
            'mean_error_y': gt_likelihood_info.get('mean_error_y'),
            'mean_mahalanobis': gt_likelihood_info.get('mean_mahalanobis'),
            'std_mahalanobis': gt_likelihood_info.get('std_mahalanobis'),
        }

        csv_rows.append(row)

    if not csv_rows:
        print("No valid log-likelihood data is available for export")
        return

    # Convert to a DataFrame and save it.
    df = pd.DataFrame(csv_rows)
    df = df.sort_values('relative_timestamp')

    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False, float_format='%.6f')

    # Compute summary statistics.
    valid_ll = [r['log_likelihood'] for r in csv_rows if r['log_likelihood'] is not None]
    valid_nll = [r['avg_nll'] for r in csv_rows if r['avg_nll'] is not None]

    print(f"Saved GT log-likelihood CSV: {csv_path}")
    print(f"   Total records: {len(csv_rows)}")
    if valid_ll:
        print(f"   Mean total log likelihood: {np.mean(valid_ll):.4f}")
        print(f"   Mean per-step NLL: {np.mean(valid_nll):.4f}")
        print(f"   Time range: {df['relative_timestamp'].min():.3f} - {df['relative_timestamp'].max():.3f} s")


def export_parameter_timeseries_csv(
    results: Dict[str, Any],
    output_dir: str,
    initial_timestamp: float,
    csv_filename: str = 'parameter_timeseries.csv'
):

    if 'history' not in results or not results['history']:
        print("No parameter-identification history is available for export")
        return

    estimation_history = results['history']

    # Prepare CSV data.
    csv_rows = []

    for hist_item in estimation_history:
        # Read window timing.
        window_start_time = hist_item.get('window_start_time', 0.0)
        window_end_time = hist_item.get('window_end_time', 0.0)

        # Convert to time relative to the initial timestamp.
        relative_start_time = window_start_time - initial_timestamp
        relative_end_time = window_end_time - initial_timestamp

        # Use the window end time as the representative timestamp for the frame.
        relative_timestamp = relative_end_time

        # Read parameter values.
        mean_values = hist_item.get('mean', {})
        std_values = hist_item.get('std', {})
        map_values = hist_item.get('map', {})
        ci_95_values = hist_item.get('ci_95', {})

        # Build one CSV row.
        row = {
            'window_idx': hist_item.get('window_idx', -1),
            'relative_timestamp': relative_timestamp,
            'absolute_timestamp': window_end_time,
            'window_start_time': relative_start_time,
            'window_end_time': relative_end_time,
        }

        # Add mean, std, MAP, and confidence interval for each parameter.
        for param_name in mean_values.keys():
            row[f'{param_name}_mean'] = mean_values[param_name]
            row[f'{param_name}_std'] = std_values.get(param_name, 0.0)
            row[f'{param_name}_map'] = map_values.get(param_name, 0.0)

            # Confidence interval.
            if param_name in ci_95_values:
                ci = ci_95_values[param_name]
                row[f'{param_name}_ci_lower'] = ci[0] if isinstance(ci, (list, tuple)) else 0.0
                row[f'{param_name}_ci_upper'] = ci[1] if isinstance(ci, (list, tuple)) else 0.0
            else:
                row[f'{param_name}_ci_lower'] = 0.0
                row[f'{param_name}_ci_upper'] = 0.0

        # Add other statistics.
        if 'ess' in hist_item:
            row['ess'] = hist_item['ess']
        if 'window_duration' in hist_item:
            row['window_duration'] = hist_item['window_duration']

        # Add weight statistics.
        if 'weight_mean' in hist_item:
            row['weight_mean'] = hist_item['weight_mean']
        if 'weight_max' in hist_item:
            row['weight_max'] = hist_item['weight_max']
        if 'weight_min' in hist_item:
            row['weight_min'] = hist_item['weight_min']
        if 'weight_std' in hist_item:
            row['weight_std'] = hist_item['weight_std']
        if 'weight_median' in hist_item:
            row['weight_median'] = hist_item['weight_median']
        if 'weight_entropy' in hist_item:
            row['weight_entropy'] = hist_item['weight_entropy']
        if 'max_weight_particle_idx' in hist_item:
            row['max_weight_particle_idx'] = hist_item['max_weight_particle_idx']

        csv_rows.append(row)

    # Convert to a DataFrame and save it.
    df = pd.DataFrame(csv_rows)

    # Sort by timestamp.
    df = df.sort_values('relative_timestamp')

    # Save as CSV.
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False, float_format='%.6f')

    print(f"Saved parameter time-series CSV: {csv_path}")
    print(f"   Total records: {len(csv_rows)}")
    print(f"   Time range: {df['relative_timestamp'].min():.3f} - {df['relative_timestamp'].max():.3f} s")
    print(f"   Parameters: {list(mean_values.keys())}")


def export_particle_weights_csv(
    particles_history: list,
    output_dir: str,
    initial_timestamp: float,
    csv_filename: str = 'particle_weights.csv'
):

    import numpy as np

    if not particles_history:
        print("No particle-history data is available for export")
        return

    csv_rows = []

    for window_info in particles_history:
        window_idx = window_info.get('window_idx', -1)
        window_start_time = window_info.get('window_start_time', 0.0)
        window_end_time = window_info.get('window_end_time', 0.0)
        particles = window_info.get('particles', [])

        # Convert to relative time.
        relative_start_time = window_start_time - initial_timestamp
        relative_end_time = window_end_time - initial_timestamp

        for particle in particles:
            particle_idx = particle.get('particle_idx', -1)
            weight = particle.get('weight', 0.0)
            log_weight = particle.get('log_weight', 0.0)
            theta = particle.get('theta', {})

            row = {
                'window_idx': window_idx,
                'particle_idx': particle_idx,
                'relative_timestamp': relative_end_time,
                'absolute_timestamp': window_end_time,
                'window_start_time': relative_start_time,
                'window_end_time': relative_end_time,
                'weight': float(weight),
                'log_weight': float(log_weight),
            }

            # Add each parameter value.
            for param_name, param_value in theta.items():
                row[f'param_{param_name}'] = float(param_value)

            csv_rows.append(row)

    if not csv_rows:
        print("No valid particle data is available for export")
        return

    # Convert to a DataFrame and save it.
    df = pd.DataFrame(csv_rows)
    df = df.sort_values(['window_idx', 'particle_idx'])

    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False, float_format='%.6f')

    print(f"Saved particle-weight CSV: {csv_path}")
    print(f"   Total records: {len(csv_rows)}")
    print(f"   Window count: {len(particles_history)}")
    print(f"   Particles per window: {len(particles_history[0]['particles']) if particles_history else 0}")

    # Compute summary statistics.
    if csv_rows:
        weights = [r['weight'] for r in csv_rows]
        print("   Weight statistics:")
        print(f"     Mean: {np.mean(weights):.6f}")
        print(f"     Max: {np.max(weights):.6f}")
        print(f"     Min: {np.min(weights):.6f}")
        print(f"     Std: {np.std(weights):.6f}")


def _identify_parameters_with_cleanup(identifier, **kwargs):
    """Run identification and deterministically release its simulation engine."""
    try:
        return identifier.identify_parameters(**kwargs)
    finally:
        identifier.simulation_manager.close()


def _run_experiment(args, output_dir, observed_trajectory, trajectory_dict,
                    timestamps, full_data, main_vehicle_data, actual_csv_path):
    # Save the parameter config.
    args_dict = vars(args)
    save_config(args_dict, os.path.join(output_dir, 'config.json'))

    # Build the environment config.
    env_config = setup_environment(args)
    # Write the cognitive modulation override into the environment config for downstream readers.
    if getattr(args, 'cognitive_modulation', 'auto') != 'auto':
        env_config['cognitive_modulation_override'] = args.cognitive_modulation

    # Set parameter bounds.
    param_bounds = setup_param_bounds(args)

    # Set the likelihood covariance.
    sigma_diag = [args.sigma_px, args.sigma_py, args.sigma_vx, args.sigma_vy]
    print("\nCreating the parameter identifier...")
    identifier = ParameterIdentifier(
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
        output_dir=output_dir,
        enable_trajectory_visualization=args.enable_trajectory_visualization,
        visualization_interval=args.visualization_interval,
        enable_prediction=args.enable_prediction,
        pred_win=args.pred_win,
        trajectory_duration=args.trajectory_duration,
        enable_comprehensive_visualization=args.enable_comprehensive_visualization,
        predict_with_all_particles=args.predict_with_all_particles,
        use_geometric_mean_likelihood=args.use_geometric_mean_likelihood,
        mc_samples=args.mc_samples,
        mc_seed_mode=args.mc_seed_mode,
        mc_use_all_particles=args.mc_use_all_particles,
        enable_multi_traj_metrics=getattr(args, 'enable_multi_traj_metrics', False),
        metric_k=getattr(args, 'metric_k', 5),
        miss_epsilon=getattr(args, 'miss_epsilon', 2.0),
        collision_info_csv_path=None,
        data_path=args.data_path,
        seed=args.seed,

    )

    # 3. Create the cognitive-parameter config.
    if args.use_cognitive_modules:
        import argparse
        cognitive_args = argparse.Namespace()
        cognitive_args.use_cognitive_modules = True
        cognitive_args.use_cognitive_bias = args.use_cognitive_bias
        cognitive_args.use_cognitive_delay = args.use_cognitive_delay
        cognitive_args.use_cognitive_perception = args.use_cognitive_perception
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

    # Create the SimulationManager and inject trajectory data.
    from parameter_identify.utils.simulation_manager import SimulationManager
    identifier.simulation_manager = SimulationManager(
        env_config=env_config,
        ppo_model_path=args.ppo_model_path,
        use_cognitive_modules=args.use_cognitive_modules,
        cognitive_args=cognitive_args
    )

    # Attach trajectory data to the SimulationManager.
    identifier.simulation_manager.trajectory_dict = trajectory_dict
    identifier.simulation_manager.full_data = full_data
    identifier.simulation_manager.original_data = main_vehicle_data
    identifier.simulation_manager.original_timestamps = timestamps
    identifier.simulation_manager.scenario_path = actual_csv_path

    # Preload the PPO network and report its cognitive-modulation signature.
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

    print(f"Successfully loaded {len(observed_trajectory)} data points")
    # Validate data.
    if not validate_trajectory_data([{
        'px': p.px, 'py': p.py, 'vx': p.vx, 'vy': p.vy
    } for p in observed_trajectory]):
        print("Data validation failed")
        exit()
        return

    print("\nStarting parameter identification...")
    results = _identify_parameters_with_cleanup(
        identifier,
        observed_trajectory=observed_trajectory,
        initialization_method=args.initialization_method,
        resampling_threshold=args.resampling_threshold,
        evolution_noise=args.evolution_noise,
        save_interval=args.save_interval
    )

    print("\nPlotting result figures...")
    identifier.data_save.plot_results()

    print("\nExporting results...")
    export_results(results, output_dir, args.export_formats)

    print("\nExporting the parameter time-series CSV...")
    initial_timestamp = get_initial_timestamp_from_csv(actual_csv_path)

    # Export failures must propagate to the transaction boundary.
    export_parameter_timeseries_csv(
        results=results,
        output_dir=output_dir,
        initial_timestamp=initial_timestamp,
        csv_filename='parameter_timeseries.csv'
    )
    print("Parameter time-series CSV saved")

    print("\nExporting the GT log-likelihood time-series CSV...")
    if hasattr(identifier, 'prediction_history') and identifier.prediction_history:
        export_gt_likelihood_csv(
            prediction_history=identifier.prediction_history,
            output_dir=output_dir,
            initial_timestamp=initial_timestamp,
            csv_filename='gt_likelihood_timeseries.csv'
        )
    else:
        print("No prediction history is available; skipping GT log-likelihood export")

    print("\nExporting the per-particle weight CSV...")
    if 'particles_history' in results and results['particles_history']:
        export_particle_weights_csv(
            particles_history=results['particles_history'],
            output_dir=output_dir,
            initial_timestamp=initial_timestamp,
            csv_filename='particle_weights.csv'
        )
    else:
        print("No particle-history data is available; skipping particle-weight export")



def main():

    # 1. Configure multiprocessing to avoid CUDA and MetaDrive engine conflicts.
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    import os
    os.environ['METADRIVE_MULTIPROCESS_MODE'] = '1'


    # 2. Parse arguments.
    args = parse_arguments()
    _ensure_checkpoint_file_exists(args.ppo_model_path)

    # Validate the requested input before creating output directories or saving config.json.
    print("\nLoading observation data...")
    observed_trajectory, trajectory_dict, timestamps, full_data, main_vehicle_data = (
        _load_observation_data(args.data_path, args.trajectory_duration)
    )
    actual_csv_path = args.data_path

    # Print the resolved run configuration.
    print("=" * 60)
    print("Cognitive Parameter Identification Tool")
    print("=" * 60)
    print(f"Data file: {args.data_path}")
    print(f"Trajectory duration: {args.trajectory_duration if args.trajectory_duration else 'all'} s")
    print(f"PPO model: {args.ppo_model_path}")
    print(f"Particle count: {args.num_particles}")
    print(f"Window size: {args.window_size}")
    print(f"Step interval: {args.step_interval}")
    print(f"Worker count: {args.num_workers}")
    print(f"Prediction module: {'enabled' if args.enable_prediction else 'disabled'}")
    if args.enable_prediction:
        print(f"Prediction window: {args.pred_win}")
        print(f"Prediction mode: {'all particles' if args.predict_with_all_particles else 'best particle'}")
    print(f"Cognitive modules: {'enabled' if args.use_cognitive_modules else 'disabled'}")
    print(f"Comprehensive visualization: {'enabled' if args.enable_comprehensive_visualization else 'disabled'}")
    print(f"Output root: {args.output_dir}")
    print("=" * 60)

    seed_global_generators(args.seed)

    # Create the output directory under a transaction boundary.
    experiment_name = generate_experiment_name()
    final_output_dir = os.path.join(args.output_dir, experiment_name)
    with experiment_transaction(final_output_dir, scenario=args.data_path) as output_dir:
        _run_experiment(
            args=args,
            output_dir=output_dir,
            observed_trajectory=observed_trajectory,
            trajectory_dict=trajectory_dict,
            timestamps=timestamps,
            full_data=full_data,
            main_vehicle_data=main_vehicle_data,
            actual_csv_path=actual_csv_path,
        )

    print(f"\nAll results were saved to: {final_output_dir}")

    print("\nProgram finished successfully")


if __name__ == "__main__":
    main()
