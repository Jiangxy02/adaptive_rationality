"""Configuration serialization mixin for PPO expert training."""


import copy
import json
import os
from datetime import datetime
from typing import Any, Dict

from ppo_train.config.defaults import (
    NETWORK_ACTION_DIM_DEFAULT,
    NETWORK_ACTIVATION_DEFAULT,
    NETWORK_HIDDEN_DIM_DEFAULT,
    build_serialized_perception_config,
    build_serialized_sampler_config,
)


class ConfigMixin:
    """Mixin holding methods split from PPOExpertReproduction."""

    def _build_config(self) -> Dict[str, Any]:
        """Build the full configuration"""
        legacy_direct_concat = (
            self.cognitive_modulation == "concat"
            and self.cognitive_param_dim == 4
        )
        if legacy_direct_concat:
            network_config = {
                "observation_dim": self.network_input_dim,
                "action_dim": NETWORK_ACTION_DIM_DEFAULT,
                "hidden_dim": NETWORK_HIDDEN_DIM_DEFAULT,
                "activation": NETWORK_ACTIVATION_DEFAULT,
                "cognitive_params_integration": True,
                "cognitive_params_dim": self.cognitive_param_dim,
                "cognitive_mask_dim": self.cognitive_mask_dim,
            }
            perception_config = build_serialized_perception_config(self.args, legacy_direct_concat)
            sampler_config = build_serialized_sampler_config(self.args, legacy_direct_concat)
        else:
            network_config = {
                "observation_dim": self.network_input_dim,
                "base_observation_dim": self.base_obs_dim,
                "raw_observation_dim": self.raw_obs_dim,
                "action_dim": NETWORK_ACTION_DIM_DEFAULT,
                "hidden_dim": NETWORK_HIDDEN_DIM_DEFAULT,
                "activation": NETWORK_ACTIVATION_DEFAULT,
                "cognitive_params_integration": True,
                "cognitive_params_dim": self.cognitive_param_dim,
                "cognitive_mask_dim": self.cognitive_mask_dim,
                "cognitive_modulation": self.cognitive_modulation,
            }
            perception_config = build_serialized_perception_config(self.args, legacy_direct_concat)
            sampler_config = build_serialized_sampler_config(self.args, legacy_direct_concat)

        effects_enabled = bool(self.args.use_cognitive_modules)
        perception_config["enabled"] = bool(
            effects_enabled and self.args.use_cognitive_perception
        )
        sampler_config["enabled"] = bool(
            effects_enabled and self.args.use_cognitive_parameter_sampling
        )

        runtime_config = copy.deepcopy(self.resolved_runtime_config)
        scenario_config = runtime_config["scenario"]
        environment_config = runtime_config["environment"]
        serialized_env_config = {
            "scenario_type": scenario_config["scenario_type"],
            "num_scenarios": scenario_config["total_scenarios"],
            "dynamic_road_length": True,
            "segments_range": copy.deepcopy(scenario_config["segments_range"]),
            "segment_length_range_meters": copy.deepcopy(
                scenario_config["segment_length_range_meters"]
            ),
            "road_length_range_meters": copy.deepcopy(
                scenario_config["road_length_range_meters"]
            ),
            "dynamic_traffic": environment_config["dynamic_traffic"],
            "random_traffic": environment_config["random_traffic"],
            "initial_traffic_density": environment_config["initial_traffic_density"],
            "horizon": environment_config["horizon"],
            "reward_config": copy.deepcopy(environment_config["reward_config"]),
            "termination_config": copy.deepcopy(environment_config["termination_config"]),
            "vehicle_config": copy.deepcopy(environment_config["vehicle_config"]),
            "environments": copy.deepcopy(runtime_config["environments"]),
        }
        evaluation_benchmark = {
            "num_envs": 1,
            "scenario_seed": runtime_config["environments"][0]["scenario"]["seed"],
            "traffic_density": environment_config["initial_traffic_density"],
            "reset_same_scenarios_each_evaluation": True,
            "cognitive_params": {
                "bias_inverse_tta_coef": self.args.bias_inverse_tta_coef,
                "perception_sigma0": self.args.perception_sigma0,
                "perception_sigma_max": self.args.perception_sigma_max,
                "delay_steps": self.args.delay_steps,
            },
        }

        return {
            # ===== Reproduction metadata =====
            "reproduction_target": "MetaDrive PPO Expert",
            "experiment_name": os.path.basename(self.exp_dir),
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.args.seed,
            "device": str(self.device),
            "reward_contract": "metadrive_tuple_speed_control_v1",

            # ===== System settings =====
            "system": {
                "device": str(self.device),
                "seed": self.args.seed,
                "save_dir": self.args.save_dir,
                "resume_from": self.args.resume_from,
                "warm_start_from": getattr(self.args, "warm_start_from", None),
            },

            # ===== Network structure (expert-aligned with cognitive parameter integration) =====
            "network": network_config,

            # Single source of truth consumed by runtime, manifest, and reports.
            "resolved_runtime_config": runtime_config,

            # ===== Environment configuration (straight-road scenario generation) =====
            "env_config": serialized_env_config,
            "evaluation_benchmark": evaluation_benchmark,

            # ===== Key hyperparameters =====
            "hyperparameters": copy.deepcopy(runtime_config["ppo"]),

            # ===== Learning-rate schedule =====
            "learning_rate_schedule": {
                "schedule_type": self.args.lr_schedule,
                "lr_min": self.args.lr_min,
                "warmup_ratio": self.args.warmup_ratio
            },

            # ===== Entropy coefficient decay =====
            "entropy_decay": {
                "entropy_coef_start": self.args.entropy_coef_start,
                "entropy_coef_end": self.args.entropy_coef_end,
                "entropy_decay_end_ratio": self.args.entropy_decay_end_ratio
            },

            # ===== Training settings =====
            "training": copy.deepcopy(runtime_config["training"]),

            # ===== Curriculum settings =====
            "curriculum_learning": copy.deepcopy(runtime_config["curriculum"]),

            # ===== Cognitive module settings =====
            "cognitive_modules": {
                "enabled": self.args.use_cognitive_modules,
                "visualization": self.args.cognitive_visualization,

                # Cognitive bias module (risk aversion)
                "cognitive_bias": {
                    "enabled": bool(
                        self.args.use_cognitive_modules and self.args.use_cognitive_bias
                    ),
                    "inverse_tta_coef": self.args.bias_inverse_tta_coef,
                    "tta_threshold": self.args.bias_tta_threshold,
                    "adaptive_bias": self.args.bias_adaptive,
                    "adaptation_rate": self.args.bias_adaptation_rate,
                    "visual_detection_distance": self.args.bias_visual_distance,
                    "visual_detection_angle": self.args.bias_visual_angle,
                    "visual_aversion_strength": self.args.bias_visual_strength
                },

                # Cognitive delay module (action delay)
                "cognitive_delay": {
                    "enabled": bool(
                        self.args.use_cognitive_modules and self.args.use_cognitive_delay
                    ),
                    "delay_steps": self.args.delay_steps,
                    "enable_smoothing": self.args.delay_smoothing,
                    "smoothing_factor": self.args.delay_smoothing_factor
                },

                # Cognitive perception module (observation noise)
                "cognitive_perception": perception_config,

                # Cognitive parameter sampler
                "cognitive_parameter_sampler": sampler_config
            }
        }

    def _save_config(self):
        """Save the configuration file"""
        config_path = os.path.join(self.exp_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
