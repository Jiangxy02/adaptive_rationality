"""Resolve the exact runtime facts consumed by training and its artifacts."""

import copy
import logging
from typing import Any, Dict

from common.random_seed import SeedDomain, derive_seed
from ppo_train.config.defaults import build_termination_config, build_vehicle_config


RUNTIME_CONFIG_SCHEMA_VERSION = 2
NUM_SCENARIOS = 1000
MAP_TYPE = "straight_road_segments"
MIN_STRAIGHT_SEGMENTS = 5
MAX_STRAIGHT_SEGMENTS = 7
STRAIGHT_SEGMENT_LENGTH_RANGE_METERS = (40, 80)
HORIZON = 800
RANDOM_TRAFFIC = True

CURRICULUM_STAGE_EDGES = (0.0, 0.02, 0.05, 0.1, 1.0)
CURRICULUM_DENSITY_RANGES = (
    (0.03, 0.06),
    (0.06, 0.09),
    (0.09, 0.12),
    (0.12, 0.15),
)


def _segment_count(scenario_index: int) -> int:
    """Map every scenario index onto the inclusive configured segment range."""
    value_count = MAX_STRAIGHT_SEGMENTS - MIN_STRAIGHT_SEGMENTS + 1
    bucket = min(value_count - 1, scenario_index * value_count // NUM_SCENARIOS)
    return MIN_STRAIGHT_SEGMENTS + bucket


def resolve_scenario(seed: int) -> Dict[str, Any]:
    """Resolve the deterministic scenario facts for one environment seed."""
    seed = int(seed)
    scenario_index = seed % NUM_SCENARIOS
    num_segments = _segment_count(scenario_index)
    min_segment_length, max_segment_length = STRAIGHT_SEGMENT_LENGTH_RANGE_METERS
    return {
        "seed": seed,
        "scenario_index": scenario_index,
        "num_segments": num_segments,
        "map": "S" * num_segments,
        "road_length_range_meters": [
            num_segments * min_segment_length,
            num_segments * max_segment_length,
        ],
    }


def _reward_config(args) -> Dict[str, Any]:
    return {
        "success_reward": args.success_reward,
        "driving_reward": args.driving_reward,
        "speed_reward": args.speed_reward,
        "use_lateral_reward": args.use_lateral_reward,
        "out_of_road_penalty": args.out_of_road_penalty,
        "crash_vehicle_penalty": args.crash_penalty,
        "crash_object_penalty": args.crash_penalty,
        "crash_sidewalk_penalty": args.crash_sidewalk_penalty,
        "use_speed_control_reward": args.use_speed_control_reward,
        "speed_control_enable_tracking": args.speed_control_enable_tracking,
        "speed_control_enable_soft_wall": args.speed_control_enable_soft_wall,
        "speed_control_enable_behavior_guidance": args.speed_control_enable_behavior_guidance,
        "speed_control_k": args.speed_control_k,
        "speed_control_kappa": args.speed_control_kappa,
        "speed_control_mu": args.speed_control_mu,
        "speed_control_nu": args.speed_control_nu,
        "speed_control_v_tolerance": args.speed_control_v_tolerance,
        "speed_control_v_ref": args.speed_control_v_ref,
    }


def _ppo_config(args) -> Dict[str, Any]:
    return {
        "learning_rate": args.lr,
        "n_steps": args.n_steps,
        "n_envs": args.n_envs,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "entropy_coef": args.entropy_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
    }


def _training_config(args) -> Dict[str, Any]:
    return {
        "total_timesteps": args.total_timesteps,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "log_freq": args.log_freq,
    }


def _curriculum_config(args) -> Dict[str, Any]:
    config = {
        "enabled": True,
        "mode": args.curriculum_mode,
        "alpha": args.curriculum_alpha,
        "stage_edges": list(CURRICULUM_STAGE_EDGES),
        "density_ranges": [list(values) for values in CURRICULUM_DENSITY_RANGES],
    }
    if args.curriculum_mode == "gate":
        config["promotion_thresholds"] = {
            "success_rate_min": args.gate_succ_threshold,
            "collision_rate_max": args.gate_coll_threshold,
        }
    return config


def _metadrive_config(
    args,
    *,
    start_seed: int,
    map_string: str,
    initial_traffic_density: float,
) -> Dict[str, Any]:
    reward = _reward_config(args)
    config = {
        "num_scenarios": NUM_SCENARIOS,
        "map": map_string,
        "traffic_density": float(initial_traffic_density),
        "random_traffic": RANDOM_TRAFFIC,
        "horizon": HORIZON,
        "start_seed": int(start_seed),
        "use_render": False,
        "debug": False,
        "image_observation": False,
        "log_level": logging.WARNING,
        "success_reward": reward["success_reward"],
        "driving_reward": reward["driving_reward"],
        "speed_reward": reward["speed_reward"],
        "use_lateral_reward": reward["use_lateral_reward"],
        "out_of_road_penalty": reward["out_of_road_penalty"],
        "crash_vehicle_penalty": reward["crash_vehicle_penalty"],
        "crash_object_penalty": reward["crash_object_penalty"],
        "crash_sidewalk_penalty": reward["crash_sidewalk_penalty"],
        **build_termination_config(),
        "vehicle_config": build_vehicle_config(),
    }
    if reward["use_speed_control_reward"]:
        config.update({
            key: value
            for key, value in reward.items()
            if key.startswith("speed_control_") or key == "use_speed_control_reward"
        })
    return config


def resolve_runtime_config(args, initial_traffic_density: float) -> Dict[str, Any]:
    """Resolve every runtime value before environments or artifacts are created."""
    n_envs = int(args.n_envs)
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    root_seed = int(args.seed)
    environments = []
    for rank in range(n_envs):
        worker_seed = derive_seed(root_seed, SeedDomain.WORKER, rank)
        start_seed = derive_seed(root_seed, SeedDomain.ENVIRONMENT, rank)
        scenario = resolve_scenario(start_seed)
        environments.append({
            "rank": rank,
            "worker_seed": worker_seed,
            "scenario": scenario,
            "metadrive_config": _metadrive_config(
                args,
                start_seed=start_seed,
                map_string=scenario["map"],
                initial_traffic_density=initial_traffic_density,
            ),
        })

    min_segment_length, max_segment_length = STRAIGHT_SEGMENT_LENGTH_RANGE_METERS
    sample_scenarios = [
        resolve_scenario(
            derive_seed(root_seed, SeedDomain.ENVIRONMENT, n_envs + sample_index)
        )
        for sample_index in range(4)
    ]
    return {
        "schema_version": RUNTIME_CONFIG_SCHEMA_VERSION,
        "scenario": {
            "scenario_type": MAP_TYPE,
            "total_scenarios": NUM_SCENARIOS,
            "segments_range": [MIN_STRAIGHT_SEGMENTS, MAX_STRAIGHT_SEGMENTS],
            "segment_length_range_meters": list(STRAIGHT_SEGMENT_LENGTH_RANGE_METERS),
            "road_length_range_meters": [
                MIN_STRAIGHT_SEGMENTS * min_segment_length,
                MAX_STRAIGHT_SEGMENTS * max_segment_length,
            ],
            "seed_derivation": "SeedSequence(root_seed, domain, rank)",
            "sample_scenarios": sample_scenarios,
        },
        "environment": {
            "initial_traffic_density": float(initial_traffic_density),
            "dynamic_traffic": True,
            "random_traffic": RANDOM_TRAFFIC,
            "horizon": HORIZON,
            "reward_config": _reward_config(args),
            "termination_config": build_termination_config(),
            "vehicle_config": build_vehicle_config(),
        },
        "environments": environments,
        "ppo": _ppo_config(args),
        "training": _training_config(args),
        "curriculum": _curriculum_config(args),
    }


def resolved_env_config(runtime_config: Dict[str, Any], rank: int) -> Dict[str, Any]:
    """Return an isolated copy of one pre-resolved environment configuration."""
    environments = runtime_config["environments"]
    if rank < 0 or rank >= len(environments):
        raise IndexError(f"environment rank {rank} is outside 0..{len(environments) - 1}")
    resolved = environments[rank]
    if resolved["rank"] != rank:
        raise ValueError(
            f"resolved environment rank mismatch: expected {rank}, got {resolved['rank']}"
        )
    return copy.deepcopy(resolved)
