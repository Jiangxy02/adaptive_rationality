"""Exact-resume and explicit warm-start support for PPO training."""

import copy
import hashlib
import json
import math
import os
import random
from collections import deque
from typing import Any, Dict

import numpy as np
import torch
import torch.optim as optim

from common.cognitive_input import PERCEPTION_SIGMA_UNIT, PUBLIC_OBSERVATION_DIM

CHECKPOINT_SCHEMA_VERSION = 6
REWARD_CONTRACT = "metadrive_tuple_speed_control_v1"
PPO_UPDATE_CONTRACT = "latent_tanh_squashed_entropy_transactional_kl_v2"
_CHECKPOINT_KIND = "exact_training_state"
_NON_SEMANTIC_ARGS = {"resume_from", "warm_start_from", "save_dir"}
_RETIRED_NON_SEMANTIC_ARGS = {"action_penalty_coef"}
_METRIC_DEQUES = (
    "episode_rewards",
    "episode_lengths",
    "episode_speeds",
    "episode_lane_deviations",
    "episode_lane_changes",
    "episode_path_completions",
    "episode_timeouts",
    "episode_steer_means",
    "episode_throttle_means",
)


def _plain(value: Any) -> Any:
    """Convert configuration values to stable, weights-only-safe primitives."""
    if isinstance(value, dict):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _network_signature(state_dict: Dict[str, torch.Tensor]) -> list:
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in state_dict.items()
    ]


class CheckpointingMixin:
    """Checkpoint methods shared by the PPO trainer."""

    def _semantic_args(self) -> Dict[str, Any]:
        return {
            key: _plain(value)
            for key, value in sorted(vars(self.args).items())
            if key not in _NON_SEMANTIC_ARGS
        }

    def _optimizer_signature(self) -> Dict[str, Any]:
        groups = []
        for group in self.optimizer.param_groups:
            # lr is dynamic schedule state and is restored from optimizer_state_dict;
            # the configured initial/schedule values are already covered by semantic_args.
            options = {
                key: _plain(value)
                for key, value in group.items()
                if key not in {"params", "lr", "initial_lr"}
            }
            groups.append({"parameter_count": len(group["params"]), "options": options})
        return {"class": type(self.optimizer).__qualname__, "groups": groups}

    def _compatibility_state(self) -> Dict[str, Any]:
        return {
            "perception_sigma_unit": PERCEPTION_SIGMA_UNIT,
            "reward_contract": REWARD_CONTRACT,
            "ppo_update_contract": PPO_UPDATE_CONTRACT,
            "semantic_args": self._semantic_args(),
            "network": _network_signature(self.network.state_dict()),
            "optimizer": self._optimizer_signature(),
            "progressive_structure": {
                "enabled": bool(getattr(self.network, "is_progressive_training", False)),
                "checkpoint_obs_dim": getattr(self.network, "checkpoint_obs_dim", None),
                "freeze_threshold_steps": int(getattr(self.network, "freeze_threshold_steps", 0)),
            },
        }

    @staticmethod
    def _compatibility_hash(state: Dict[str, Any]) -> str:
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_compatibility_state(state: Dict[str, Any]) -> Dict[str, Any]:
        normalized = copy.deepcopy(state)
        semantic_args = normalized.get("semantic_args")
        if isinstance(semantic_args, dict):
            for key in _RETIRED_NON_SEMANTIC_ARGS:
                semantic_args.pop(key, None)
        return normalized

    @staticmethod
    def _capture_rng_state() -> Dict[str, Any]:
        algorithm, keys, position, has_gauss, cached = np.random.get_state()
        state = {
            "python": random.getstate(),
            "numpy": {
                "algorithm": algorithm,
                "keys": torch.as_tensor(keys.astype(np.int64)),
                "position": int(position),
                "has_gauss": int(has_gauss),
                "cached_gaussian": float(cached),
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Dict[str, Any]) -> None:
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (
                numpy_state["algorithm"],
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_cpu"].cpu())
        if state.get("torch_cuda"):
            if not torch.cuda.is_available():
                raise ValueError("Checkpoint contains CUDA RNG state, but CUDA is unavailable in the current runtime")
            torch.cuda.set_rng_state_all([
                rng_state.cpu() for rng_state in state["torch_cuda"]
            ])

    def _capture_runtime_state(self) -> Dict[str, Any]:
        metrics = {}
        for name in _METRIC_DEQUES:
            value = getattr(self, name, deque(maxlen=100))
            metrics[name] = _plain(list(value))
        return {
            "episode_count": int(getattr(self, "episode_count", 0)),
            "train_stats": _plain(getattr(self, "train_stats", [])),
            "current_entropy_coef": float(getattr(self, "current_entropy_coef", 0.0)),
            "best_reward": float(getattr(self, "best_reward", float("-inf"))),
            "best_metrics": _plain(getattr(self, "best_metrics", {})),
            "metric_deques": metrics,
            "last_lane_index": copy.deepcopy(getattr(self, "_last_lane_index", {})),
        }

    def _restore_runtime_state(self, state: Dict[str, Any]) -> None:
        self.episode_count = int(state["episode_count"])
        self.train_stats = copy.deepcopy(state["train_stats"])
        self.current_entropy_coef = float(state["current_entropy_coef"])
        self.best_reward = float(state["best_reward"])
        self.best_metrics = copy.deepcopy(state["best_metrics"])
        for name in _METRIC_DEQUES:
            target = getattr(self, name, deque(maxlen=100))
            setattr(self, name, deque(state["metric_deques"][name], maxlen=target.maxlen or 100))
        self._last_lane_index = copy.deepcopy(state["last_lane_index"])

    def _capture_cognitive_state(self) -> Dict[str, Any]:
        state = {}
        sampler = getattr(self, "cognitive_parameter_sampler", None)
        if sampler is not None:
            if not hasattr(sampler, "state_dict"):
                raise RuntimeError("The cognitive-parameter sampler does not implement state_dict, so an exact-resume checkpoint cannot be created")
            state["parameter_sampler"] = sampler.state_dict()
        return state

    def _restore_cognitive_state(self, state: Dict[str, Any]) -> None:
        sampler = getattr(self, "cognitive_parameter_sampler", None)
        saved_sampler = state.get("parameter_sampler")
        if (sampler is None) != (saved_sampler is None):
            raise ValueError("Cognitive-parameter sampler structure mismatch; use --warm_start_from for structural migration")
        if sampler is not None:
            sampler.load_state_dict(saved_sampler)

    def _capture_environment_state(self) -> list:
        envs = getattr(self, "envs", None)
        if envs is None:
            return []
        if not hasattr(envs, "env_method"):
            raise RuntimeError("The vectorized environment does not support get_resume_state, so an exact-resume checkpoint cannot be created")
        return envs.env_method("get_resume_state")

    def _restore_environment_state(self, states: list) -> None:
        envs = getattr(self, "envs", None)
        if envs is None:
            if states:
                raise ValueError("Checkpoint contains environment state, but the current trainer has no environment")
            return
        current = envs.env_method("get_resume_state")
        if len(current) != len(states):
            raise ValueError(
                f"Environment count mismatch: checkpoint={len(states)}, current={len(current)}; "
                "use --warm_start_from for structural migration"
            )
        for index, state in enumerate(states):
            envs.env_method("set_resume_state", state, indices=index)

    def _build_checkpoint_payload(self, iteration: int) -> Dict[str, Any]:
        compatibility = self._compatibility_state()
        checkpoint_config = copy.deepcopy(self.config)
        if isinstance(checkpoint_config, dict):
            checkpoint_config["cognitive_param_slot2"] = "sigma_max"
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_kind": _CHECKPOINT_KIND,
            "perception_sigma_unit": PERCEPTION_SIGMA_UNIT,
            "reward_contract": REWARD_CONTRACT,
            "ppo_update_contract": PPO_UPDATE_CONTRACT,
            "checkpoint_boundary": "completed_iteration",
            "iteration": int(iteration),
            "global_step": int(self.global_step),
            "curriculum_mode": self.curriculum_mode,
            "curriculum_stage": int(self.curriculum_stage),
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "compatibility": compatibility,
            "compatibility_hash": self._compatibility_hash(compatibility),
            "runtime_state": self._capture_runtime_state(),
            "cognitive_states": self._capture_cognitive_state(),
            "environment_states": self._capture_environment_state(),
            "rng_state": self._capture_rng_state(),
            "progressive_training": {
                "enabled": bool(getattr(self.network, "is_progressive_training", False)),
                "training_stage": int(getattr(self.network, "training_stage", 1)),
                "checkpoint_obs_dim": getattr(self.network, "checkpoint_obs_dim", None),
                "freeze_threshold_steps": int(getattr(self.network, "freeze_threshold_steps", 0)),
            },
            "config": checkpoint_config,
            "args": _plain(vars(self.args)),
            "cognitive_param_slot2": "sigma_max",
            "best_reward": float(getattr(self, "best_reward", float("-inf"))),
            "best_metrics": _plain(getattr(self, "best_metrics", {})),
        }

    @staticmethod
    def _atomic_torch_save(payload: Dict[str, Any], path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp.{os.getpid()}"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def save_checkpoint(self, iteration: int, is_best: bool = False):
        """Persist a complete iteration-boundary state atomically."""
        payload = self._build_checkpoint_payload(iteration)
        checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self._atomic_torch_save(payload, os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt"))
        self._atomic_torch_save(payload, os.path.join(checkpoint_dir, "latest_model.pt"))
        if is_best:
            self._atomic_torch_save(payload, os.path.join(checkpoint_dir, "best_model.pt"))

    def maybe_save_best_checkpoint(self, iteration: int, eval_metrics: Dict[str, Any]) -> bool:
        """Save every newly observed best immediately; never overwrite it with a worse result."""
        reward = float(eval_metrics["eval_reward_mean"])
        if not math.isfinite(reward) or reward <= float(getattr(self, "best_reward", float("-inf"))):
            return False
        self.best_reward = reward
        self.best_metrics = _plain(eval_metrics)
        payload = self._build_checkpoint_payload(iteration)
        self._atomic_torch_save(payload, os.path.join(self.exp_dir, "checkpoints", "best_model.pt"))
        return True

    def _validate_exact_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        required = {
            "schema_version", "checkpoint_kind", "perception_sigma_unit",
            "reward_contract", "ppo_update_contract",
            "iteration", "global_step",
            "curriculum_mode", "curriculum_stage", "network_state_dict",
            "optimizer_state_dict", "compatibility", "runtime_state",
            "cognitive_states", "environment_states", "rng_state", "progressive_training",
            "compatibility_hash",
        }
        missing = sorted(required.difference(checkpoint))
        if missing or checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION or checkpoint.get("checkpoint_kind") != _CHECKPOINT_KIND:
            detail = f"missing {missing}" if missing else "schema is not the exact-resume format"
            raise ValueError(f"Checkpoint cannot be strict-resumed ({detail}); use --warm_start_from to migrate weights only")
        if checkpoint.get("perception_sigma_unit") != PERCEPTION_SIGMA_UNIT:
            raise ValueError(
                "Checkpoint cannot be strict-resumed because the perception-sigma unit is not meters; "
                "retrain with the current code"
            )
        if checkpoint.get("reward_contract") != REWARD_CONTRACT:
            raise ValueError(
                "Checkpoint cannot be strict-resumed because the reward contract is not the current version; "
                "retrain with the current code"
            )
        if checkpoint.get("ppo_update_contract") != PPO_UPDATE_CONTRACT:
            raise ValueError(
                "Checkpoint cannot be strict-resumed because the PPO update contract is not the current version; "
                "retrain with the current code"
            )

        saved_compatibility = checkpoint["compatibility"]
        if checkpoint["compatibility_hash"] != self._compatibility_hash(saved_compatibility):
            raise ValueError("Checkpoint compatibility fingerprint is corrupted; use a different checkpoint")
        current_compatibility = self._compatibility_state()
        saved_for_comparison = self._normalize_compatibility_state(saved_compatibility)
        current_for_comparison = self._normalize_compatibility_state(current_compatibility)
        if saved_for_comparison != current_for_comparison:
            saved_network = saved_for_comparison.get("network")
            current_network = current_for_comparison.get("network")
            if saved_network != current_network:
                detail = "network structure or parameter signature mismatch"
            elif saved_for_comparison.get("optimizer") != current_for_comparison.get("optimizer"):
                detail = "optimizer structure or options mismatch"
            else:
                detail = "training configuration mismatch"
            raise ValueError(f"Strict resume rejected: {detail}; use --warm_start_from for structural or configuration migration")

        if checkpoint["curriculum_mode"] != self.curriculum_mode:
            raise ValueError("Curriculum mode mismatch; use --warm_start_from for structural migration")
        progressive = checkpoint["progressive_training"]
        if not isinstance(progressive, dict) or not {
            "enabled", "training_stage", "checkpoint_obs_dim", "freeze_threshold_steps"
        }.issubset(progressive):
            raise ValueError("Checkpoint progressive-training state is incomplete; use --warm_start_from")
        if progressive["training_stage"] not in (1, 2):
            raise ValueError("Checkpoint progressive-training stage is invalid; use --warm_start_from")

        runtime_required = {
            "episode_count", "train_stats", "current_entropy_coef", "best_reward",
            "best_metrics", "metric_deques", "last_lane_index",
        }
        if (
            not isinstance(checkpoint["runtime_state"], dict)
            or not runtime_required.issubset(checkpoint["runtime_state"])
        ):
            raise ValueError("Checkpoint runtime_state is incomplete; use --warm_start_from")
        if set(checkpoint["runtime_state"]["metric_deques"]) != set(_METRIC_DEQUES):
            raise ValueError("Checkpoint metrics buffer structure mismatch; use --warm_start_from")
        rng = checkpoint["rng_state"]
        if not isinstance(rng, dict) or not {"python", "numpy", "torch_cpu", "torch_cuda"}.issubset(rng):
            raise ValueError("Checkpoint RNG state is incomplete; use --warm_start_from")
        numpy_rng = rng["numpy"]
        if not isinstance(numpy_rng, dict) or not {
            "algorithm", "keys", "position", "has_gauss", "cached_gaussian"
        }.issubset(numpy_rng) or not torch.is_tensor(numpy_rng["keys"]):
            raise ValueError("Checkpoint NumPy RNG state is invalid; use --warm_start_from")
        if not torch.is_tensor(rng["torch_cpu"]):
            raise ValueError("Checkpoint Torch RNG state is invalid; use --warm_start_from")
        if rng.get("torch_cuda") and not torch.cuda.is_available():
            raise ValueError("Checkpoint contains CUDA RNG state, but CUDA is unavailable in the current runtime")

        # Validate model/optimizer/sampler on copies before touching live state.
        candidate_network = copy.deepcopy(self.network)
        candidate_network.load_state_dict(checkpoint["network_state_dict"], strict=True)
        self._validate_optimizer_state_shapes(checkpoint["optimizer_state_dict"])
        candidate_optimizer = copy.deepcopy(self.optimizer)
        candidate_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        sampler = getattr(self, "cognitive_parameter_sampler", None)
        saved_sampler = checkpoint["cognitive_states"].get("parameter_sampler")
        if (sampler is None) != (saved_sampler is None):
            raise ValueError("Cognitive-parameter sampler structure mismatch; use --warm_start_from for structural migration")
        if sampler is not None:
            copy.deepcopy(sampler).load_state_dict(saved_sampler)
        current_env_states = (
            self.envs.env_method("get_resume_state")
            if getattr(self, "envs", None) is not None
            else []
        )
        if len(current_env_states) != len(checkpoint["environment_states"]):
            raise ValueError("Environment count mismatch; use --warm_start_from for structural migration")
        # Env restore methods validate their complete child state before assignment.
        # Round-trip the preflight so a rejection leaves the live env byte-for-byte unchanged.
        for index, (current_state, saved_state) in enumerate(
            zip(current_env_states, checkpoint["environment_states"])
        ):
            self.envs.env_method("set_resume_state", saved_state, indices=index)
            self.envs.env_method("set_resume_state", current_state, indices=index)

    def _validate_optimizer_state_shapes(self, state_dict: Dict[str, Any]) -> None:
        saved_groups = state_dict.get("param_groups", [])
        if len(saved_groups) != len(self.optimizer.param_groups):
            raise ValueError("Strict resume rejected: optimizer parameter-group count mismatch; use --warm_start_from")
        for group_index, (saved_group, current_group) in enumerate(zip(saved_groups, self.optimizer.param_groups)):
            saved_ids = saved_group.get("params", [])
            current_params = current_group["params"]
            if len(saved_ids) != len(current_params):
                raise ValueError(
                    f"Strict resume rejected: optimizer parameter group {group_index} size mismatch; "
                    "use --warm_start_from"
                )
            for saved_id, current_param in zip(saved_ids, current_params):
                for value in state_dict.get("state", {}).get(saved_id, {}).values():
                    if torch.is_tensor(value) and value.ndim > 0 and value.shape != current_param.shape:
                        raise ValueError(
                            "Strict resume rejected: optimizer slot shape does not match the model parameter; "
                            "use --warm_start_from"
                        )

    def load_checkpoint(self, checkpoint_path: str):
        """Strictly restore a complete checkpoint after full preflight validation."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        cached_path = getattr(self, "_preloaded_checkpoint_path", None)
        if cached_path == os.path.abspath(checkpoint_path):
            checkpoint = self._preloaded_checkpoint
            del self._preloaded_checkpoint
            del self._preloaded_checkpoint_path
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self._validate_exact_checkpoint(checkpoint)

        self.network.load_state_dict(checkpoint["network_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint["global_step"])
        self.start_iteration = int(checkpoint["iteration"])
        self.curriculum_stage = int(checkpoint["curriculum_stage"])
        self._restore_runtime_state(checkpoint["runtime_state"])
        self._restore_cognitive_state(checkpoint["cognitive_states"])
        self._restore_environment_state(checkpoint["environment_states"])
        progressive = checkpoint["progressive_training"]
        self.checkpoint_obs_dim = progressive["checkpoint_obs_dim"]
        self.network.checkpoint_obs_dim = progressive["checkpoint_obs_dim"]
        self.network.freeze_threshold_steps = int(progressive["freeze_threshold_steps"])
        self.network.training_stage = int(progressive["training_stage"])
        self.network.is_progressive_training = bool(progressive["enabled"])
        self.is_progressive_training = self.network.is_progressive_training
        if self.is_progressive_training:
            if self.network.training_stage == 2 and hasattr(self.network, "_unfreeze_weights"):
                self.network._unfreeze_weights()
        # RNG restore is deliberately last: object construction and state loading may consume randomness.
        self._restore_rng_state(checkpoint["rng_state"])
        print(f"Exact-resume state restored: iteration={self.start_iteration}, global_step={self.global_step:,}")

    def preflight_strict_resume(self, checkpoint_path: str) -> None:
        """Reject incompatibility before the trainer writes logs or mutates live state."""
        cached_path = getattr(self, "_preloaded_checkpoint_path", None)
        if cached_path == os.path.abspath(checkpoint_path):
            checkpoint = self._preloaded_checkpoint
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self._validate_exact_checkpoint(checkpoint)

    def get_strict_resume_network_context(self, checkpoint_path: str):
        """Read the saved progressive-network constructor context before trainer creation."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("checkpoint_kind") != _CHECKPOINT_KIND
            or checkpoint.get("perception_sigma_unit") != PERCEPTION_SIGMA_UNIT
            or checkpoint.get("reward_contract") != REWARD_CONTRACT
            or checkpoint.get("ppo_update_contract") != PPO_UPDATE_CONTRACT
        ):
            raise ValueError(
                "Checkpoint is not in the current meter-based exact-resume format; legacy-unit checkpoints must be retrained"
            )
        progressive = checkpoint.get("progressive_training")
        if not isinstance(progressive, dict):
            raise ValueError("Checkpoint is missing progressive-training state; use --warm_start_from instead")
        self._preloaded_checkpoint_path = os.path.abspath(checkpoint_path)
        self._preloaded_checkpoint = checkpoint
        if progressive.get("enabled"):
            return progressive.get("checkpoint_obs_dim")
        return None

    def load_warm_start(self, checkpoint_path: str):
        """Load weights only, adapting input width explicitly and resetting all training state."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or checkpoint.get("checkpoint_kind") != _CHECKPOINT_KIND
            or checkpoint.get("perception_sigma_unit") != PERCEPTION_SIGMA_UNIT
            or checkpoint.get("reward_contract") != REWARD_CONTRACT
            or checkpoint.get("ppo_update_contract") != PPO_UPDATE_CONTRACT
        ):
            raise ValueError(
                "warm-start only accepts the current checkpoint schema with "
                "perception sigmas in meters"
            )
        if "network_state_dict" not in checkpoint:
            raise ValueError("Warm-start checkpoint is missing network_state_dict")
        source = checkpoint["network_state_dict"]
        source_dim = None
        for key in ("actor_fc1.weight", "critic_fc1.weight"):
            if key in source:
                source_dim = int(source[key].shape[1])
                break
        if source_dim is None:
            raise ValueError("Warm-start checkpoint input dimension could not be identified")
        if (
            getattr(self, "raw_obs_dim", None) == PUBLIC_OBSERVATION_DIM
            and source_dim != PUBLIC_OBSERVATION_DIM
        ):
            raise ValueError(
                "current public warm-start requires a 283-dimensional checkpoint; "
                f"got observation_dim={source_dim}"
            )

        self.network = self._create_network(checkpoint_obs_dim=source_dim)
        target_dim = self.network.actor_fc1.in_features
        if source_dim < target_dim:
            state = self.network._load_and_extend_weights(source)
        elif source_dim > target_dim:
            state = self.network._load_and_truncate_weights(source)
        else:
            state = source
        self.network.load_state_dict(state, strict=True)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.args.lr)
        self.checkpoint_obs_dim = source_dim
        self.is_progressive_training = getattr(self.network, "is_progressive_training", False)
        self.global_step = 0
        self.start_iteration = 0
        self.curriculum_stage = 0
        self.best_reward = float("-inf")
        self.best_metrics = {}
        print(f"Warm-start weight migration completed: {source_dim}->{target_dim}; optimizer, progress, and runtime state were reset")
