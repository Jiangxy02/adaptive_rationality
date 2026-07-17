
import argparse
from pathlib import Path

from common.cli import add_boolean_argument
from parameter_identify.utils.particle_manager import (
    DEFAULT_EVOLUTION_NOISE_RATIO,
    perception_sigma_domain_has_positive_probability,
    validate_evolution_noise_ratio,
)
from parameter_identify.utils.likelihood_calculator import validate_sigma_value


def str2bool(value):
    """argparse-friendly bool parser."""
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


def likelihood_sigma(value):
    """Parse one positive, finite observation-noise standard deviation."""
    try:
        return validate_sigma_value(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def trajectory_csv_path(value):
    """Reject non-CSV trajectory inputs before any file is opened."""
    if Path(value).suffix.lower() != '.csv':
        raise argparse.ArgumentTypeError(
            "trajectory data must be a CSV matching "
            "examples/trajectory_scenario/example_trajectory_scene.csv"
        )
    return value


def _validate_perception_sigma_bounds(parser, args):
    """Validate the paper-domain perception sigma search bounds."""
    if args.perception_sigma0_min < 0.0 or args.perception_sigma_max_min < 0.0:
        parser.error("perception sigma bounds must be non-negative")
    if args.perception_sigma0_min > args.perception_sigma0_max:
        parser.error("perception_sigma0_min must be <= perception_sigma0_max")
    if args.perception_sigma_max_min > args.perception_sigma_max_max:
        parser.error("perception_sigma_max_min must be <= perception_sigma_max_max")
    if not perception_sigma_domain_has_positive_probability(
        (args.perception_sigma0_min, args.perception_sigma0_max),
        (args.perception_sigma_max_min, args.perception_sigma_max_max),
    ):
        parser.error(
            "perception sigma domain has no positive-probability sample "
            "satisfying sigma0 <= sigma_max"
        )


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Particle-filter-based cognitive parameter identification tool"
    )

    parser.add_argument(
        '--data_path',
        type=trajectory_csv_path,
        default='examples/trajectory_scenario/example_trajectory_scene.csv',
        help=(
            'Trajectory CSV path; must match the six-column public example format exactly'
        )
    )

    parser.add_argument(
        '--trajectory_duration',
        type=float,
        default=11,
        help='Trajectory duration in seconds (default: 11)'
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
        default=40,
        help='Number of particles (default: 40)'
    )

    parser.add_argument(
        '--window_size',
        type=int,
        default=10,
        help='Sliding window size (default: 10)'
    )

    parser.add_argument(
        '--step_interval',
        type=int,
        default=1,
        help='Sliding-window step interval in simulation steps (default: 1)'
    )

    add_boolean_argument(
        parser,
        '--enable_prediction',
        default=True,
        help_text='Enable future-trajectory prediction from the best particle (enabled by default)',
    )

    parser.add_argument(
        '--pred_win',
        type=int,
        default=30,
        help='Prediction window size in future steps (default: 30)'
    )

    parser.add_argument(
        '--predict_with_all_particles',
        action='store_true',
        default=False,
        help='Run prediction rollouts for all particles instead of only the best particle'
    )

    parser.add_argument(
        '--mc_samples',
        type=int,
        default=10,
        help='Monte Carlo trajectories per particle during prediction (default: 10, best particle only)'
    )
    parser.add_argument(
        '--mc_seed_mode',
        type=str,
        default='sequence',
        choices=['sequence', 'random'],
        help='Monte Carlo seed mode: sequence increments deterministically, random draws fresh seeds'
    )
    parser.add_argument(
        '--mc_use_all_particles',
        action='store_true',
        default=False,
        help='Run Monte Carlo prediction for all particles (default: False, best particle only)'
    )

    add_boolean_argument(
        parser,
        '--enable_multi_traj_metrics',
        default=True,
        help_text='Enable multi-trajectory metrics (MinADE@K/MinFDE@K/MissRate@K, enabled by default)',
    )
    parser.add_argument(
        '--metric_k',
        type=int,
        default=10,
        help='Top-K value for multi-trajectory metrics (default: 10)'
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
        help='Look-ahead horizon in steps (default: 0)'
    )

    parser.add_argument(
        '--cognitive_modulation',
        type=str,
        default='auto',
        choices=['auto', 'none', 'concat'],
        help='Cognitive modulation mode (auto uses the checkpoint configuration)'
    )

    parser.add_argument(
        '--initialization_method',
        type=str,
        default='lhs',
        choices=['uniform', 'gaussian', 'lhs'],
        help='Particle initialization method (default: lhs)'
    )

    parser.add_argument(
        '--resampling_threshold',
        type=float,
        default=0.5,
        help='Resampling threshold as an ESS ratio (default: 0.5)'
    )

    parser.add_argument(
        '--evolution_noise',
        type=evolution_noise_ratio,
        default=DEFAULT_EVOLUTION_NOISE_RATIO,
        help=(
            'Normalized evolution-noise standard deviation as a fraction of the '
            f'full continuous-parameter range (default: {DEFAULT_EVOLUTION_NOISE_RATIO})'
        )
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
        '--robust_loss',
        type=str,
        default=None,
        choices=['huber', 'tukey', None],
        help='Robust loss function type'
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
        default=10,
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
        default=10,
        help='Maximum delay steps'
    )

    add_boolean_argument(
        parser,
        '--use_cognitive_modules',
        default=True,
        help_text='Enable cognitive effects (enabled by default; the network remains 283-dimensional when disabled)',
    )

    parser.add_argument(
        '--use_cognitive_bias',
        type=str2bool,
        default=True,
        help='Enable the cognitive-bias module (default: True)'
    )

    parser.add_argument(
        '--use_cognitive_delay',
        type=str2bool,
        default=True,
        help='Enable the cognitive-delay module (default: True)'
    )

    parser.add_argument(
        '--use_cognitive_perception',
        type=str2bool,
        default=True,
        help='Enable the cognitive-perception module (default: True)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Output directory (default: ./results)'
    )


    parser.add_argument(
        '--save_interval',
        type=int,
        default=10,
        help='Interval for saving intermediate results (default: 10)'
    )

    parser.add_argument(
        '--export_formats',
        nargs='+',
        default=['json', 'pkl', 'csv'],
        choices=['json', 'pkl', 'csv'],
        help='Export format list'
    )

    add_boolean_argument(
        parser,
        '--enable_trajectory_visualization',
        default=True,
        help_text='Enable trajectory visualization (enabled by default)',
    )

    parser.add_argument(
        '--visualization_interval',
        type=int,
        default=10,
        help='Visualization interval in rolling iterations (default: 10)'
    )

    add_boolean_argument(
        parser,
        '--enable_comprehensive_visualization',
        default=True,
        help_text='Enable comprehensive trajectory visualization including identification and prediction (enabled by default)',
    )

    add_boolean_argument(
        parser,
        '--use_geometric_mean_likelihood',
        default=True,
        help_text='Enable geometric-mean likelihood tempering (enabled by default)',
    )

    parser.add_argument(
        '--env_config',
        type=str,
        default=None,
        help='Environment config file path (JSON or YAML)'
    )

    parser.add_argument(
        '--enable_render',
        action='store_true',
        default=False,
        help='Enable rendering'
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
        help='Show verbose output'
    )

    args = parser.parse_args()
    _validate_perception_sigma_bounds(parser, args)
    return args
