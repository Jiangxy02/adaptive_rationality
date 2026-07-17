"""Single source of truth for ppo_train default arguments and configuration constructor defaults."""

from common.cognitive_input import PERCEPTION_SIGMA_UNIT, validate_perception_sigmas


# Core PPO/LR defaults
LR_DEFAULT = 3e-4
LR_SCHEDULE_DEFAULT = "linear"
LR_MIN_DEFAULT = 3e-5
WARMUP_RATIO_DEFAULT = 0.05
N_STEPS_DEFAULT = 5376
N_ENVS_DEFAULT = 3
BATCH_SIZE_DEFAULT = 512
N_EPOCHS_DEFAULT = 10
GAMMA_DEFAULT = 0.99
GAE_LAMBDA_DEFAULT = 0.95
CLIP_RANGE_DEFAULT = 0.2
ENTROPY_COEF_DEFAULT = 0.01
VF_COEF_DEFAULT = 0.5
MAX_GRAD_NORM_DEFAULT = 0.5
TARGET_KL_DEFAULT = 0.03

# Entropy decay defaults
ENTROPY_COEF_START_DEFAULT = 0.015
ENTROPY_COEF_END_DEFAULT = 0.005
ENTROPY_DECAY_END_RATIO_DEFAULT = 0.8

# Reward defaults
SUCCESS_REWARD_DEFAULT = 100.0
DRIVING_REWARD_DEFAULT = 0.4
SPEED_REWARD_DEFAULT = 0
USE_LATERAL_REWARD_DEFAULT = True
OUT_OF_ROAD_PENALTY_DEFAULT = 8.0
CRASH_PENALTY_DEFAULT = 8.0
CRASH_SIDEWALK_PENALTY_DEFAULT = 8.0

# Training control defaults
TOTAL_TIMESTEPS_DEFAULT = 50000000
CHECKPOINT_FREQ_DEFAULT = 10
EVAL_FREQ_DEFAULT = 2
LOG_FREQ_DEFAULT = 1

# Curriculum learning defaults
CURRICULUM_MODE_DEFAULT = "progress"
CURRICULUM_ALPHA_DEFAULT = 1.5
GATE_SUCC_THRESHOLD_DEFAULT = 0.70
GATE_COLL_THRESHOLD_DEFAULT = 0.20

# Cognitive module core defaults
USE_COGNITIVE_MODULES_DEFAULT = True
COGNITIVE_VISUALIZATION_DEFAULT = False
COGNITIVE_MODULATION_DEFAULT = "concat"

# Cognitive parameter sampler defaults
USE_COGNITIVE_PARAMETER_SAMPLING_DEFAULT = True
COGNITIVE_SAMPLER_TYPE_DEFAULT = "continuous"
COGNITIVE_PARAM_UPDATE_STEPS_DEFAULT = 5
BIAS_INVERSE_TTA_COEF_RANGE_DEFAULT = [0.0, 10.0]
PERCEPTION_SIGMA0_RANGE_DEFAULT = [0.0, 1.0]
PERCEPTION_SIGMA_MAX_RANGE_DEFAULT = [0.0, 5.0]
DELAY_STEPS_RANGE_DEFAULT = [0, 20]
PERCEPTION_SIGMA_MAX_DENSITY_DEFAULT = 4
PERCEPTION_SIGMA0_DENSITY_DEFAULT = 4
BIAS_INVERSE_TTA_COEF_DENSITY_DEFAULT = 4
DELAY_STEPS_DENSITY_DEFAULT = 2

# Cognitive bias defaults
USE_COGNITIVE_BIAS_DEFAULT = True
BIAS_INVERSE_TTA_COEF_DEFAULT = 1.0
BIAS_TTA_THRESHOLD_DEFAULT = 1.0
BIAS_ADAPTIVE_DEFAULT = False
BIAS_ADAPTATION_RATE_DEFAULT = 0.01
BIAS_VISUAL_DISTANCE_DEFAULT = 50.0
BIAS_VISUAL_ANGLE_DEFAULT = 30.0
BIAS_VISUAL_STRENGTH_DEFAULT = 0.5

# Cognitive delay defaults
USE_COGNITIVE_DELAY_DEFAULT = True
DELAY_STEPS_DEFAULT = 2
DELAY_SMOOTHING_DEFAULT = False
DELAY_SMOOTHING_FACTOR_DEFAULT = 0.3

# Cognitive perception defaults
USE_COGNITIVE_PERCEPTION_DEFAULT = True
PERCEPTION_SIGMA0_DEFAULT = 0.1
PERCEPTION_SIGMA_MAX_DEFAULT = 0.8
PERCEPTION_USE_KF_DEFAULT = True
PERCEPTION_KF_DT_DEFAULT = 0.1
PERCEPTION_KF_Q_SCALE_DEFAULT = 100.0
PERCEPTION_FAR_DISTANCE_DEFAULT = 150.0
PERCEPTION_RHO_DEFAULT = 0.8
PERCEPTION_USE_AR1_DEFAULT = True

# Cognitive visualization defaults
ENABLE_RADAR_BEAM_VIZ_DEFAULT = False

# System defaults
DEVICE_DEFAULT = "cuda"
SEED_DEFAULT = 101
RESUME_FROM_DEFAULT = None

# Speed control defaults. Direct speed shaping is opt-in; the base task reward
# remains MetaDrive progress plus terminal rewards.
USE_SPEED_CONTROL_REWARD_DEFAULT = False
SPEED_CONTROL_ENABLE_TRACKING_DEFAULT = True
SPEED_CONTROL_ENABLE_SOFT_WALL_DEFAULT = True
SPEED_CONTROL_ENABLE_BEHAVIOR_GUIDANCE_DEFAULT = True
SPEED_CONTROL_K_DEFAULT = 0.12
SPEED_CONTROL_KAPPA_DEFAULT = 0.15
SPEED_CONTROL_MU_DEFAULT = 0.3
SPEED_CONTROL_NU_DEFAULT = 0.2
SPEED_CONTROL_V_TOLERANCE_DEFAULT = 1.0
SPEED_CONTROL_V_REF_DEFAULT = 10.0

# Network serialization defaults
NETWORK_ACTION_DIM_DEFAULT = 2
NETWORK_HIDDEN_DIM_DEFAULT = 256
NETWORK_ACTIVATION_DEFAULT = "tanh"

def build_termination_config():
    """Build the saved termination config with a fresh dict."""
    return {
        "out_of_road_done": True,
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "on_continuous_line_done": False,
        "on_broken_line_done": False
    }


def build_vehicle_config():
    """Build the saved vehicle config with fresh nested dicts."""
    return {
        "lidar": {
            "num_lasers": 240,
            "distance": 50,
            "num_others": 4,
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0
        },
        "side_detector": {
            "num_lasers": 0,
            "distance": 50,
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0
        },
        "lane_line_detector": {
            "num_lasers": 0,
            "distance": 20,
            "gaussian_noise": 0.0,
            "dropout_prob": 0.0
        }
    }


def _validated_perception_sigmas(args):
    """Return paper-domain perception sigmas, both expressed directly in meters."""
    return validate_perception_sigmas(
        args.perception_sigma0,
        args.perception_sigma_max,
    )


def build_serialized_perception_config(args, legacy_direct_concat):
    """Build the saved perception config without mixing runtime-only keys."""
    sigma0, sigma_max = _validated_perception_sigmas(args)
    if legacy_direct_concat:
        return {
            "enabled": args.use_cognitive_perception,
            "sigma0": sigma0,
            "sigma_max": sigma_max,
            "distance_unit": PERCEPTION_SIGMA_UNIT,
            "use_kf": args.perception_use_kf,
            "kf_dt": args.perception_kf_dt,
            "kf_q_scale": args.perception_kf_q_scale,
            "enable_radar_beam_viz": getattr(args, 'enable_radar_beam_viz', False)
        }
    return {
        "enabled": args.use_cognitive_perception,
        "sigma0": sigma0,
        "sigma_max": sigma_max,
        "distance_unit": PERCEPTION_SIGMA_UNIT,
        "use_kf": args.perception_use_kf,
        "kf_dt": args.perception_kf_dt,
        "kf_q_scale": args.perception_kf_q_scale,
        "enable_radar_beam_viz": getattr(args, 'enable_radar_beam_viz', False)
    }


def build_runtime_perception_config(args):
    """Build the runtime perception config consumed by CognitivePerceptionModule."""
    sigma0, sigma_max = _validated_perception_sigmas(args)
    return {
        'sigma0': sigma0,
        'sigma_max': sigma_max,
        'distance_unit': PERCEPTION_SIGMA_UNIT,
        'far_distance': PERCEPTION_FAR_DISTANCE_DEFAULT,
        'use_ar1': PERCEPTION_USE_AR1_DEFAULT,
        'rho': PERCEPTION_RHO_DEFAULT,
        'use_kf': args.perception_use_kf,
        'kf_dt': args.perception_kf_dt,
        'kf_q_scale': args.perception_kf_q_scale
    }


def build_serialized_sampler_config(args, legacy_direct_concat):
    """Build the saved sampler config using sigma_max perception keys."""
    if legacy_direct_concat:
        return {
            "enabled": args.use_cognitive_parameter_sampling,
            "sampler_type": args.cognitive_sampler_type,
            "update_steps": args.cognitive_param_update_steps,
            "bias_inverse_tta_coef_range": args.bias_inverse_tta_coef_range,
            "perception_sigma0_range": args.perception_sigma0_range,
            "perception_sigma_max_range": args.perception_sigma_max_range,
            "delay_steps_range": args.delay_steps_range,
            "bias_inverse_tta_coef_density": args.bias_inverse_tta_coef_density,
            "perception_sigma0_density": args.perception_sigma0_density,
            "perception_sigma_max_density": args.perception_sigma_max_density,
            "delay_steps_density": args.delay_steps_density
        }
    return {
        "enabled": args.use_cognitive_parameter_sampling,
        "sampler_type": args.cognitive_sampler_type,
        "update_steps": args.cognitive_param_update_steps,
        "bias_inverse_tta_coef_range": args.bias_inverse_tta_coef_range,
        "perception_sigma0_range": args.perception_sigma0_range,
        "perception_sigma_max_range": args.perception_sigma_max_range,
        "delay_steps_range": args.delay_steps_range,
        "bias_inverse_tta_coef_density": args.bias_inverse_tta_coef_density,
        "perception_sigma0_density": args.perception_sigma0_density,
        "perception_sigma_max_density": args.perception_sigma_max_density,
        "delay_steps_density": args.delay_steps_density
    }
