
import argparse
from pathlib import Path

from common.cli import add_boolean_argument
from ppo_train.config.defaults import (
    BATCH_SIZE_DEFAULT,
    BIAS_ADAPTATION_RATE_DEFAULT,
    BIAS_ADAPTIVE_DEFAULT,
    BIAS_INVERSE_TTA_COEF_DEFAULT,
    BIAS_INVERSE_TTA_COEF_DENSITY_DEFAULT,
    BIAS_INVERSE_TTA_COEF_RANGE_DEFAULT,
    BIAS_TTA_THRESHOLD_DEFAULT,
    BIAS_VISUAL_ANGLE_DEFAULT,
    BIAS_VISUAL_DISTANCE_DEFAULT,
    BIAS_VISUAL_STRENGTH_DEFAULT,
    CHECKPOINT_FREQ_DEFAULT,
    CLIP_RANGE_DEFAULT,
    COGNITIVE_MODULATION_DEFAULT,
    COGNITIVE_PARAM_UPDATE_STEPS_DEFAULT,
    COGNITIVE_SAMPLER_TYPE_DEFAULT,
    COGNITIVE_VISUALIZATION_DEFAULT,
    CRASH_PENALTY_DEFAULT,
    CRASH_SIDEWALK_PENALTY_DEFAULT,
    CURRICULUM_ALPHA_DEFAULT,
    CURRICULUM_MODE_DEFAULT,
    DELAY_SMOOTHING_DEFAULT,
    DELAY_SMOOTHING_FACTOR_DEFAULT,
    DELAY_STEPS_DEFAULT,
    DELAY_STEPS_DENSITY_DEFAULT,
    DELAY_STEPS_RANGE_DEFAULT,
    DEVICE_DEFAULT,
    DRIVING_REWARD_DEFAULT,
    ENABLE_RADAR_BEAM_VIZ_DEFAULT,
    ENTROPY_COEF_DEFAULT,
    ENTROPY_COEF_END_DEFAULT,
    ENTROPY_COEF_START_DEFAULT,
    ENTROPY_DECAY_END_RATIO_DEFAULT,
    EVAL_FREQ_DEFAULT,
    GAE_LAMBDA_DEFAULT,
    GAMMA_DEFAULT,
    GATE_COLL_THRESHOLD_DEFAULT,
    GATE_SUCC_THRESHOLD_DEFAULT,
    LOG_FREQ_DEFAULT,
    LR_DEFAULT,
    LR_MIN_DEFAULT,
    LR_SCHEDULE_DEFAULT,
    MAX_GRAD_NORM_DEFAULT,
    N_ENVS_DEFAULT,
    N_EPOCHS_DEFAULT,
    N_STEPS_DEFAULT,
    OUT_OF_ROAD_PENALTY_DEFAULT,
    PERCEPTION_KF_DT_DEFAULT,
    PERCEPTION_KF_Q_SCALE_DEFAULT,
    PERCEPTION_SIGMA0_DEFAULT,
    PERCEPTION_SIGMA0_DENSITY_DEFAULT,
    PERCEPTION_SIGMA0_RANGE_DEFAULT,
    PERCEPTION_SIGMA_MAX_DEFAULT,
    PERCEPTION_SIGMA_MAX_DENSITY_DEFAULT,
    PERCEPTION_SIGMA_MAX_RANGE_DEFAULT,
    PERCEPTION_USE_KF_DEFAULT,
    RESUME_FROM_DEFAULT,
    SEED_DEFAULT,
    SPEED_CONTROL_ENABLE_BEHAVIOR_GUIDANCE_DEFAULT,
    SPEED_CONTROL_ENABLE_SOFT_WALL_DEFAULT,
    SPEED_CONTROL_ENABLE_TRACKING_DEFAULT,
    SPEED_CONTROL_K_DEFAULT,
    SPEED_CONTROL_KAPPA_DEFAULT,
    SPEED_CONTROL_MU_DEFAULT,
    SPEED_CONTROL_NU_DEFAULT,
    SPEED_CONTROL_V_REF_DEFAULT,
    SPEED_CONTROL_V_TOLERANCE_DEFAULT,
    SPEED_REWARD_DEFAULT,
    SUCCESS_REWARD_DEFAULT,
    TARGET_KL_DEFAULT,
    TOTAL_TIMESTEPS_DEFAULT,
    USE_COGNITIVE_BIAS_DEFAULT,
    USE_COGNITIVE_DELAY_DEFAULT,
    USE_COGNITIVE_MODULES_DEFAULT,
    USE_COGNITIVE_PARAMETER_SAMPLING_DEFAULT,
    USE_COGNITIVE_PERCEPTION_DEFAULT,
    USE_LATERAL_REWARD_DEFAULT,
    USE_SPEED_CONTROL_REWARD_DEFAULT,
    VF_COEF_DEFAULT,
    WARMUP_RATIO_DEFAULT,
)


RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEGACY_SAVE_DIR = RELEASE_ROOT / "ppo_train" / "cog_input" / "cog_save" / "radical_1"

def add_arguments():
    """Add command-line arguments"""
    parser = argparse.ArgumentParser(description="PPO Training")

    # ===== Key hyperparameters =====
    parser.add_argument("--lr", type=float, default=LR_DEFAULT,
                       help="Learning rate (default: 3e-4)")
    parser.add_argument("--lr_schedule", type=str, default=LR_SCHEDULE_DEFAULT,
                        choices=["constant", "linear", "cosine", "stage"],
                        help="Learning-rate schedule (default: linear)")
    parser.add_argument("--lr_min", type=float, default=LR_MIN_DEFAULT,
                        help="Minimum learning rate (lower bound for linear/cosine schedules)")
    parser.add_argument("--warmup_ratio", type=float, default=WARMUP_RATIO_DEFAULT,
                        help="Warmup ratio of the total training steps (default: 0.05)")

    parser.add_argument("--n_steps", type=int, default=N_STEPS_DEFAULT,
                       help="Rollout steps (default: 5376)")
    parser.add_argument("--n_envs", type=int, default=N_ENVS_DEFAULT,
                       help="Number of parallel environments (default: 3)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT,
                       help="SGD batch size (default: 512)")
    parser.add_argument("--n_epochs", type=int, default=N_EPOCHS_DEFAULT,
                       help="Training epochs per update (default: 10)")
    parser.add_argument("--gamma", type=float, default=GAMMA_DEFAULT,
                       help="Discount factor (default: 0.99)")
    parser.add_argument("--gae_lambda", type=float, default=GAE_LAMBDA_DEFAULT,
                       help="GAE lambda (default: 0.95)")
    parser.add_argument("--clip_range", type=float, default=CLIP_RANGE_DEFAULT,
                       help="PPO clip range (default: 0.2)")
    parser.add_argument("--entropy_coef", type=float, default=ENTROPY_COEF_DEFAULT,
                       help="Entropy coefficient (default: 0.01)")

    # ===== Other hyperparameters =====
    parser.add_argument("--vf_coef", type=float, default=VF_COEF_DEFAULT,
                       help="Value-function loss coefficient (default: 0.5)")
    parser.add_argument("--max_grad_norm", type=float, default=MAX_GRAD_NORM_DEFAULT,
                       help="Gradient clipping threshold (default: 0.5)")
    parser.add_argument("--target_kl", type=float, default=TARGET_KL_DEFAULT,
                       help="Target KL divergence (early stop, default: 0.03)")

    # ===== Entropy coefficient decay =====
    parser.add_argument("--entropy_coef_start", type=float, default=ENTROPY_COEF_START_DEFAULT,
                       help="Initial entropy coefficient (default: 0.015)")
    parser.add_argument("--entropy_coef_end", type=float, default=ENTROPY_COEF_END_DEFAULT,
                       help="Final entropy coefficient (default: 0.005)")
    parser.add_argument("--entropy_decay_end_ratio", type=float, default=ENTROPY_DECAY_END_RATIO_DEFAULT,
                       help="Training-progress ratio where entropy decay finishes (default: 0.8)")

    # ===== Reward configuration =====
    parser.add_argument("--success_reward", type=float, default=SUCCESS_REWARD_DEFAULT,
                       help="Success reward (default: 100.0)")
    parser.add_argument("--driving_reward", type=float, default=DRIVING_REWARD_DEFAULT,
                       help="Driving reward (default: 0.4)")
    parser.add_argument("--speed_reward", type=float, default=SPEED_REWARD_DEFAULT,
                       help="Speed reward (default: 0, reduced from 0.3)")
    add_boolean_argument(
        parser,
        "--use_lateral_reward",
        default=USE_LATERAL_REWARD_DEFAULT,
        help_text="Enable lateral lane-keeping reward so the vehicle stays on the navigation lane (default: enabled)",
    )
    parser.add_argument("--out_of_road_penalty", type=float, default=OUT_OF_ROAD_PENALTY_DEFAULT,
                       help="Off-road penalty (default: 8.0)")
    parser.add_argument("--crash_penalty", type=float, default=CRASH_PENALTY_DEFAULT,
                       help="Crash penalty (default: 8.0)")
    parser.add_argument("--crash_sidewalk_penalty", type=float, default=CRASH_SIDEWALK_PENALTY_DEFAULT,
                       help="Sidewalk-collision penalty (default: 8.0)")

    # ===== Training settings =====
    parser.add_argument("--total_timesteps", type=int, default=TOTAL_TIMESTEPS_DEFAULT,
                       help="Total training steps (default: 50,000,000)")
    parser.add_argument("--checkpoint_freq", type=int, default=CHECKPOINT_FREQ_DEFAULT,
                       help="Checkpoint save frequency (default: 10)")
    parser.add_argument("--eval_freq", type=int, default=EVAL_FREQ_DEFAULT,
                       help="Evaluation frequency (default: 2)")
    parser.add_argument("--log_freq", type=int, default=LOG_FREQ_DEFAULT,
                       help="Logging frequency (default: 1)")

    # ===== Curriculum-learning parameters =====
    parser.add_argument("--curriculum_mode", type=str, default=CURRICULUM_MODE_DEFAULT,
                        choices=["progress", "gate"],
                        help="Curriculum progression mode: progress = by training progress; gate = by evaluation thresholds (default: progress)")
    parser.add_argument("--curriculum_alpha", type=float, default=CURRICULUM_ALPHA_DEFAULT,
                        help="Piecewise interpolation exponent in progress mode (default: 1.5; larger is more conservative)")
    parser.add_argument("--gate_succ_threshold", type=float, default=GATE_SUCC_THRESHOLD_DEFAULT,
                        help="Gate mode: success-rate threshold required to advance (default: 0.70)")
    parser.add_argument("--gate_coll_threshold", type=float, default=GATE_COLL_THRESHOLD_DEFAULT,
                        help="Gate mode: maximum collision rate allowed to advance (default: 0.20)")

    # ===== Cognitive module parameters =====
    add_boolean_argument(
        parser,
        "--use_cognitive_modules",
        default=USE_COGNITIVE_MODULES_DEFAULT,
        help_text="Enable cognitive effects (default: enabled; the network remains 283-D when disabled)",
    )
    parser.add_argument("--cognitive_visualization", action="store_true", default=COGNITIVE_VISUALIZATION_DEFAULT,
                        help="Cognitive-module visualization output (default: disabled)")
    parser.add_argument(
        "--cognitive_modulation",
        type=str,
        default=COGNITIVE_MODULATION_DEFAULT,
        choices=["concat"],
        help="Concatenate cognitive parameters directly onto the base observation (the only supported public mode)",
    )

    # === Cognitive parameter sampler ===
    add_boolean_argument(
        parser,
        "--use_cognitive_parameter_sampling",
        default=USE_COGNITIVE_PARAMETER_SAMPLING_DEFAULT,
        help_text="Enable the cognitive parameter sampler (default: enabled)",
    )
    parser.add_argument("--cognitive_sampler_type", type=str, default=COGNITIVE_SAMPLER_TYPE_DEFAULT,
                        choices=["discrete", "continuous"],
                        help="Cognitive sampler type: continuous = paper-style continuous uniform sampling; discrete = discrete grid experiment (default: continuous)")
    parser.add_argument("--cognitive_param_update_steps", type=int, default=COGNITIVE_PARAM_UPDATE_STEPS_DEFAULT,
                        help="Cognitive parameter update frequency in simulation control steps (default: 5)")
    # Discrete sampler range parameters
    parser.add_argument("--bias_inverse_tta_coef_range", type=float, nargs=2, default=BIAS_INVERSE_TTA_COEF_RANGE_DEFAULT,
                        help="Sampling range for looming-aversion weight c [min, max] (default [0.0, 10.0])")
    parser.add_argument("--perception_sigma0_range", type=float, nargs=2, default=PERCEPTION_SIGMA0_RANGE_DEFAULT,
                        help="Sampling range for perception base noise sigma0 [min, max] (default [0.0, 1.0])")
    parser.add_argument("--perception_sigma_max_range", type=float, nargs=2, default=PERCEPTION_SIGMA_MAX_RANGE_DEFAULT,
                        help="Sampling range for perception max noise sigma_max [min, max] (default [0.0, 5.0])")
    parser.add_argument("--delay_steps_range", type=int, nargs=2, default=DELAY_STEPS_RANGE_DEFAULT,
                        help="Sampling range for action-delay steps [min, max] (default [0, 20])")
    # === Discrete sampler density parameters ===
    parser.add_argument("--perception_sigma_max_density", type=int, default=PERCEPTION_SIGMA_MAX_DENSITY_DEFAULT,
                        help="Number of discrete samples for perception max noise sigma_max (default: 4)")
    parser.add_argument("--perception_sigma0_density", type=int, default=PERCEPTION_SIGMA0_DENSITY_DEFAULT,
                        help="Number of discrete samples for perception noise sigma0 (default: 4)")
    parser.add_argument("--bias_inverse_tta_coef_density", type=int, default=BIAS_INVERSE_TTA_COEF_DENSITY_DEFAULT,
                        help="Number of discrete samples for looming-aversion weight c (discrete mode only, default: 4)")
    parser.add_argument("--delay_steps_density", type=int, default=DELAY_STEPS_DENSITY_DEFAULT,
                        help="Compatibility field; discrete delay always uses every integer in range, so this value does not change the support set")

    # Cognitive bias module parameters (risk aversion)
    add_boolean_argument(
        parser,
        "--use_cognitive_bias",
        default=USE_COGNITIVE_BIAS_DEFAULT,
        help_text="Enable the cognitive bias module (default: enabled)",
    )
    parser.add_argument("--bias_inverse_tta_coef", type=float, default=BIAS_INVERSE_TTA_COEF_DEFAULT,
                        help="Fixed looming-aversion weight c (default: 1.0)")
    parser.add_argument("--bias_tta_threshold", type=float, default=BIAS_TTA_THRESHOLD_DEFAULT,
                        help="TTA threshold (default: 1.0)")
    parser.add_argument("--bias_adaptive", action="store_true", default=BIAS_ADAPTIVE_DEFAULT,
                        help="Enable adaptive bias (default: disabled)")
    parser.add_argument("--bias_adaptation_rate", type=float, default=BIAS_ADAPTATION_RATE_DEFAULT,
                        help="Adaptation rate (default: 0.01)")
    parser.add_argument("--bias_visual_distance", type=float, default=BIAS_VISUAL_DISTANCE_DEFAULT,
                        help="Visual detection distance in meters (default: 50.0)")
    parser.add_argument("--bias_visual_angle", type=float, default=BIAS_VISUAL_ANGLE_DEFAULT,
                        help="Visual detection angle in degrees (default: 30.0)")
    parser.add_argument("--bias_visual_strength", type=float, default=BIAS_VISUAL_STRENGTH_DEFAULT,
                        help="Visual aversion strength (default: 0.5)")

    # Cognitive delay module parameters (action delay)
    add_boolean_argument(
        parser,
        "--use_cognitive_delay",
        default=USE_COGNITIVE_DELAY_DEFAULT,
        help_text="Enable the cognitive delay module (default: enabled)",
    )
    parser.add_argument("--delay_steps", type=int, default=DELAY_STEPS_DEFAULT,
                        help="Delay steps (default: 2)")
    parser.add_argument("--delay_smoothing", action="store_true", default=DELAY_SMOOTHING_DEFAULT,
                        help="Enable action smoothing (default: disabled)")
    parser.add_argument("--delay_smoothing_factor", type=float, default=DELAY_SMOOTHING_FACTOR_DEFAULT,
                        help="Smoothing factor (default: 0.3)")

    # Cognitive perception module parameters (observation noise)
    add_boolean_argument(
        parser,
        "--use_cognitive_perception",
        default=USE_COGNITIVE_PERCEPTION_DEFAULT,
        help_text="Enable the cognitive perception module (default: enabled)",
    )
    parser.add_argument("--perception_sigma0", type=float, default=PERCEPTION_SIGMA0_DEFAULT,
                        help="Base noise standard deviation in meters (default: 0.1)")
    parser.add_argument("--perception_sigma_max", type=float, default=PERCEPTION_SIGMA_MAX_DEFAULT,
                        help="Maximum perception noise in meters (default: 0.8)")
    add_boolean_argument(
        parser,
        "--perception_use_kf",
        default=PERCEPTION_USE_KF_DEFAULT,
        help_text="Enable the Kalman filter (default: enabled)",
    )
    parser.add_argument("--perception_kf_dt", type=float, default=PERCEPTION_KF_DT_DEFAULT,
                        help="Kalman-filter step size (default: 0.1)")
    parser.add_argument("--perception_kf_q_scale", type=float, default=PERCEPTION_KF_Q_SCALE_DEFAULT,
                        help="Kalman-filter process-noise scale (default: 100.0)")

    # ===== Cognitive visualization parameters =====
    parser.add_argument("--enable_radar_beam_viz", action="store_true", default=ENABLE_RADAR_BEAM_VIZ_DEFAULT,
                        help="Enable radar-beam visualization (default: disabled)")

    # ===== System settings =====
    parser.add_argument("--device", type=str, default=DEVICE_DEFAULT,
                       choices=["auto", "cpu", "cuda"],
                       help="Compute device (default: cuda)")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT,
                       help="Random seed (default: 101)")
    parser.add_argument("--save_dir", type=str, default=str(DEFAULT_LEGACY_SAVE_DIR),
                       help="Save directory (default: release/ppo_train/cog_input/cog_save/radical_1)")

    # ===== Training resume settings =====
    checkpoint_mode = parser.add_mutually_exclusive_group()
    checkpoint_mode.add_argument(
        "--resume_from",
        type=str,
        default=RESUME_FROM_DEFAULT,
        help="Strictly resume the full training state; the checkpoint structure and training config must match exactly",
    )
    checkpoint_mode.add_argument(
        "--warm_start_from",
        type=str,
        default=None,
        help="Transfer model weights only, allowing input-dimension extension/truncation; optimizer state, step counters, and runtime state restart from zero",
    )

    # ===== Speed-control reward parameters =====
    add_boolean_argument(
        parser,
        "--use_speed_control_reward",
        default=USE_SPEED_CONTROL_REWARD_DEFAULT,
        help_text="Enable the speed-control reward (default: disabled)",
    )

    # === Submodule enable flags ===
    add_boolean_argument(
        parser,
        "--speed_control_enable_tracking",
        default=SPEED_CONTROL_ENABLE_TRACKING_DEFAULT,
        help_text="Enable the speed-tracking submodule (default: enabled)",
    )
    add_boolean_argument(
        parser,
        "--speed_control_enable_soft_wall",
        default=SPEED_CONTROL_ENABLE_SOFT_WALL_DEFAULT,
        help_text="Enable the speed soft-wall submodule (default: enabled)",
    )
    add_boolean_argument(
        parser,
        "--speed_control_enable_behavior_guidance",
        default=SPEED_CONTROL_ENABLE_BEHAVIOR_GUIDANCE_DEFAULT,
        help_text="Enable the behavior-guidance submodule (default: enabled)",
    )

    # === Submodule parameters ===
    parser.add_argument("--speed_control_k", type=float, default=SPEED_CONTROL_K_DEFAULT,
                       help="Speed-tracking reward coefficient (default: 0.12)")
    parser.add_argument("--speed_control_kappa", type=float, default=SPEED_CONTROL_KAPPA_DEFAULT,
                       help="Overspeed soft-wall penalty coefficient (default: 0.15)")
    parser.add_argument("--speed_control_mu", type=float, default=SPEED_CONTROL_MU_DEFAULT,
                       help="Overspeed braking reward coefficient (default: 0.3)")
    parser.add_argument("--speed_control_nu", type=float, default=SPEED_CONTROL_NU_DEFAULT,
                       help="Overspeed acceleration penalty coefficient (default: 0.2)")
    parser.add_argument("--speed_control_v_tolerance", type=float, default=SPEED_CONTROL_V_TOLERANCE_DEFAULT,
                       help="Speed-tracking tolerance (default: 1.0 m/s)")
    parser.add_argument("--speed_control_v_ref", type=float, default=SPEED_CONTROL_V_REF_DEFAULT,
                       help="Target reference speed (default: 10.0 m/s)")

    return parser
