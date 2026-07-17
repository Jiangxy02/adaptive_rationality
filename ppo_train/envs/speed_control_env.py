"""Utilities for constructing MetaDrive environments with optional speed-control reward."""


import copy
from typing import Any, Dict, Mapping

import numpy as np
import torch


from metadrive.envs.metadrive_env import MetaDriveEnv

from common.random_seed import SeedDomain, derive_seed, seed_global_generators
from cognitive_module.cognitive_bias_module import CognitiveBiasModule
from cognitive_module.cognitive_perception_module import CognitivePerceptionModule
from ppo_train.config.defaults import build_runtime_perception_config


# Legacy args=None fallbacks intentionally differ from argparse defaults.
_LEGACY_SPEED_CONTROL_K_DEFAULT = 1.0
_LEGACY_SPEED_CONTROL_KAPPA_DEFAULT = 0.5
_LEGACY_SPEED_CONTROL_MU_DEFAULT = 0.3
_LEGACY_SPEED_CONTROL_NU_DEFAULT = 0.2
_LEGACY_SPEED_CONTROL_V_TOLERANCE_DEFAULT = 1.0
_LEGACY_SPEED_CONTROL_V_REF_DEFAULT = 10.0
_LEGACY_SPEED_CONTROL_ENABLE_TRACKING_DEFAULT = True
_LEGACY_SPEED_CONTROL_ENABLE_SOFT_WALL_DEFAULT = True
_LEGACY_SPEED_CONTROL_ENABLE_BEHAVIOR_GUIDANCE_DEFAULT = True

_ENV_RESUME_STATE_VERSION = 1
_PERCEPTION_MODULE_STATE_FIELDS = (
    "_last_true_x",
    "_last_true_y",
    "_last_noisy_x",
    "_last_noisy_y",
    "_last_filtered_x",
    "_last_filtered_y",
    "_last_effective_sigma",
)
_PERCEPTION_LIDAR_STATE_FIELDS = (
    "num_beams",
    "ar1_states",
    "prev_distances",
    "initialized",
    "kf_state",
    "kf_P",
    "last_original_distances",
    "last_noisy_distances",
    "last_noise_levels",
    "front_beam_index",
    "front_beam_history",
    "_beam_history",
    "_clip_log_counter",
    "sigma0",
    "sigma_max",
    "k",
)
_BIAS_HISTORY_FIELDS = (
    "bias_history",
    "reward_history",
    "tta_history",
    "detection_history",
    "distance_history",
    "threat_count_history",
)
_BIAS_SCALAR_STATE_FIELDS = ("_step_count", "_total_bias", "_active_steps")
_ENCODED_NDARRAY_KEY = "__env_resume_ndarray__"


def _encode_resume_value(value):
    """Convert NumPy values to weights_only-compatible checkpoint objects."""
    if isinstance(value, np.ndarray):
        return {
            _ENCODED_NDARRAY_KEY: True,
            "dtype": value.dtype.str,
            "data": torch.from_numpy(value.copy()),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {key: _encode_resume_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_resume_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_resume_value(item) for item in value)
    return copy.deepcopy(value)


def _decode_resume_value(value):
    if isinstance(value, Mapping):
        if value.get(_ENCODED_NDARRAY_KEY) is True:
            array = value["data"].detach().cpu().numpy()
            return array.astype(np.dtype(value["dtype"]), copy=True)
        return {key: _decode_resume_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_resume_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_resume_value(item) for item in value)
    return copy.deepcopy(value)


class CognitivePerceptionEnvBridge:
    """Own perception sensor and filter state inside one environment process."""

    def __init__(self, env, args, random_seed):
        perception_config = build_runtime_perception_config(args)
        perception_config["random_seed"] = int(random_seed)
        self.env = env
        self.module = CognitivePerceptionModule(perception_config)
        self._pending_resume_state = None

    def attach(self):
        if self.module.noise_lidar is None:
            self.module.attach_to_env(self.env)
        if self._pending_resume_state is not None and self.module.noise_lidar is not None:
            pending_state = self._pending_resume_state
            self.validate_resume_state(pending_state)
            self._restore_state(pending_state)
            self._pending_resume_state = None

    def set_parameters(self, sigma0_meters, sigma_max):
        sigma0_meters = float(sigma0_meters)
        sigma_max = float(sigma_max)
        self.module.noise_config["sigma0"] = sigma0_meters
        self.module.noise_config["sigma_max"] = sigma_max
        self.module.sigma0 = sigma0_meters
        self.module.sigma_max = sigma_max

        noise_lidar = self.module.noise_lidar
        if noise_lidar is None:
            return
        noise_lidar.sigma0 = sigma0_meters
        noise_lidar.sigma_max = sigma_max
        noise_lidar.recompute_k_from_sigma_max()

    def reset(self):
        self.module.reset()

    def get_resume_state(self):
        """Return all perception state that can affect later observations."""
        if self.module.noise_lidar is None and self._pending_resume_state is not None:
            return copy.deepcopy(self._pending_resume_state)

        module_state = {
            name: copy.deepcopy(getattr(self.module, name))
            for name in _PERCEPTION_MODULE_STATE_FIELDS
            if hasattr(self.module, name)
        }
        lidar = self.module.noise_lidar
        lidar_state = None
        if lidar is not None:
            lidar_state = {
                name: copy.deepcopy(getattr(lidar, name))
                for name in _PERCEPTION_LIDAR_STATE_FIELDS
                if hasattr(lidar, name)
            }
            lidar_state["rng_state"] = copy.deepcopy(lidar.rng.bit_generator.state)

        return {
            "version": _ENV_RESUME_STATE_VERSION,
            "module": module_state,
            "lidar": lidar_state,
        }

    def set_resume_state(self, state):
        self.validate_resume_state(state)
        if self.module.noise_lidar is None and state.get("lidar") is not None:
            self._pending_resume_state = copy.deepcopy(state)
            return
        self._restore_state(state)

    def validate_resume_state(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("cognitive perception resume state must be a mapping")
        if state.get("version") != _ENV_RESUME_STATE_VERSION:
            raise ValueError(
                "unsupported cognitive perception resume state version: "
                f"{state.get('version')!r}"
            )
        unknown_module_fields = set(state.get("module", {})) - set(
            _PERCEPTION_MODULE_STATE_FIELDS
        )
        if unknown_module_fields:
            raise ValueError(
                "unknown cognitive perception module state fields: "
                f"{sorted(unknown_module_fields)}"
            )
        lidar_state = state.get("lidar")
        if lidar_state is None:
            return
        if not isinstance(lidar_state, Mapping):
            raise TypeError("cognitive perception lidar resume state must be a mapping")
        unknown_fields = set(lidar_state) - set(_PERCEPTION_LIDAR_STATE_FIELDS) - {"rng_state"}
        if unknown_fields:
            raise ValueError(
                "unknown cognitive perception lidar state fields: "
                f"{sorted(unknown_fields)}"
            )
        rng_state = lidar_state.get("rng_state")
        if rng_state is not None:
            if not isinstance(rng_state, Mapping) or "bit_generator" not in rng_state:
                raise ValueError("invalid cognitive perception RNG state")
            lidar = self.module.noise_lidar
            expected_bit_generator = (
                type(lidar.rng.bit_generator).__name__ if lidar is not None else "PCG64"
            )
            if rng_state.get("bit_generator") != expected_bit_generator:
                raise ValueError(
                    "cognitive perception RNG mismatch: "
                    f"checkpoint={rng_state.get('bit_generator')!r}, "
                    f"runtime={expected_bit_generator!r}"
                )
            bit_generator = (
                type(lidar.rng.bit_generator)() if lidar is not None else np.random.PCG64()
            )
            bit_generator.state = copy.deepcopy(rng_state)

    def _restore_state(self, state):
        for name, value in state.get("module", {}).items():
            if name not in _PERCEPTION_MODULE_STATE_FIELDS:
                raise ValueError(f"unknown cognitive perception module state field: {name}")
            setattr(self.module, name, copy.deepcopy(value))

        lidar_state = state.get("lidar")
        if lidar_state is None:
            return
        lidar = self.module.noise_lidar
        if lidar is None:
            raise RuntimeError("cannot restore perception lidar state before attachment")
        unknown_fields = set(lidar_state) - set(_PERCEPTION_LIDAR_STATE_FIELDS) - {"rng_state"}
        if unknown_fields:
            raise ValueError(
                "unknown cognitive perception lidar state fields: "
                f"{sorted(unknown_fields)}"
            )
        for name in _PERCEPTION_LIDAR_STATE_FIELDS:
            if name in lidar_state:
                setattr(lidar, name, copy.deepcopy(lidar_state[name]))
        if "rng_state" in lidar_state:
            lidar.rng.bit_generator.state = copy.deepcopy(lidar_state["rng_state"])


class CognitiveBiasEnvBridge:
    """Run reward bias next to the real environment, including subprocess envs."""

    def __init__(self, env, args, perception_module=None):
        self.env = env
        self.module = CognitiveBiasModule({
            "inverse_tta_coef": args.bias_inverse_tta_coef,
            "tta_threshold": args.bias_tta_threshold,
            "adaptive_bias": args.bias_adaptive,
            "adaptation_rate": args.bias_adaptation_rate,
            "visual_detection_distance": args.bias_visual_distance,
            "visual_detection_angle": args.bias_visual_angle,
            "visual_aversion_strength": args.bias_visual_strength,
        }, cognitive_perception_module=perception_module)

    def process_reward(self, original_reward, info):
        return self.module.process_reward(
            original_reward=original_reward,
            env=self.env,
            info=info,
            is_ppo_mode=True,
        )

    def set_inverse_tta_coef(self, value):
        self.module.inverse_tta_coef = float(value)

    def reset(self):
        self.module.reset()

    def get_resume_state(self):
        histories = {}
        for name in _BIAS_HISTORY_FIELDS:
            history = getattr(self.module, name)
            histories[name] = {
                "values": copy.deepcopy(list(history)),
                "maxlen": history.maxlen,
            }
        return {
            "version": _ENV_RESUME_STATE_VERSION,
            "histories": histories,
            "scalars": {
                name: copy.deepcopy(getattr(self.module, name))
                for name in _BIAS_SCALAR_STATE_FIELDS
            },
        }

    def set_resume_state(self, state):
        self.validate_resume_state(state)
        histories = state.get("histories", {})
        for name, history_state in histories.items():
            history = getattr(self.module, name)
            history.clear()
            history.extend(copy.deepcopy(history_state.get("values", [])))

        for name, value in state.get("scalars", {}).items():
            setattr(self.module, name, copy.deepcopy(value))

    def validate_resume_state(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("cognitive bias resume state must be a mapping")
        if state.get("version") != _ENV_RESUME_STATE_VERSION:
            raise ValueError(
                f"unsupported cognitive bias resume state version: {state.get('version')!r}"
            )
        histories = state.get("histories", {})
        unknown_histories = set(histories) - set(_BIAS_HISTORY_FIELDS)
        if unknown_histories:
            raise ValueError(f"unknown cognitive bias histories: {sorted(unknown_histories)}")
        for name, history_state in histories.items():
            history = getattr(self.module, name)
            if history.maxlen != history_state.get("maxlen"):
                raise ValueError(
                    f"cognitive bias history length mismatch for {name}: "
                    f"checkpoint={history_state.get('maxlen')}, runtime={history.maxlen}"
                )

        scalars = state.get("scalars", {})
        unknown_scalars = set(scalars) - set(_BIAS_SCALAR_STATE_FIELDS)
        if unknown_scalars:
            raise ValueError(f"unknown cognitive bias scalar fields: {sorted(unknown_scalars)}")


class _DeterministicResumeEnvMixin:
    """Own the reset cursor and child-process state needed by strict resume."""

    @staticmethod
    def _has_crossed_destination_plane(vehicle):
        """Return whether this step crossed the final lane end inside the road corridor."""
        final_lane = vehicle.navigation.final_lane
        previous_long, previous_lateral = final_lane.local_coordinates(vehicle.last_position)
        current_long, current_lateral = final_lane.local_coordinates(vehicle.position)
        lane_width = vehicle.navigation.get_current_lane_width()
        lane_count = vehicle.navigation.get_current_lane_num()
        lower_lateral = (0.5 - lane_count) * lane_width
        upper_lateral = lane_width / 2

        previous_inside = lower_lateral <= previous_lateral <= upper_lateral
        current_inside = lower_lateral <= current_lateral <= upper_lateral
        crossed_forward = previous_long < final_lane.length <= current_long
        return crossed_forward and previous_inside and current_inside

    def _is_arrive_destination(self, vehicle):
        """Treat a forward crossing of the destination plane as successful arrival."""
        return (
            super()._is_arrive_destination(vehicle)
            or self._has_crossed_destination_plane(vehicle)
        )

    def _is_out_of_road(self, vehicle):
        """Give destination success priority over the missing lane surface past its end."""
        if self._is_arrive_destination(vehicle):
            return False
        return super()._is_out_of_road(vehicle)

    def get_resolved_env_config(self):
        """Return the exact pre-resolved configuration consumed by this env."""
        if not hasattr(self, "_resolved_env_config"):
            raise RuntimeError("resolved environment config was not attached by make_env")
        return copy.deepcopy(self._resolved_env_config)

    def _configure_resume_state(self, *, rank, base_seed, num_scenarios):
        num_scenarios = int(num_scenarios)
        if num_scenarios <= 0:
            raise ValueError("num_scenarios must be positive")
        self._resume_rank = int(rank)
        self._resume_base_seed = int(base_seed)
        self._resume_num_scenarios = num_scenarios
        self._reset_cursor = 0
        self._last_reset_seed = None

    def _prepare_deterministic_reset(self, args, kwargs):
        if not hasattr(self, "_reset_cursor"):
            raise RuntimeError("environment resume state was not configured by make_env")

        args = list(args)
        kwargs = dict(kwargs)
        caller_seed = kwargs.get("seed")
        if args:
            if caller_seed is not None:
                raise TypeError("reset seed was provided both positionally and by keyword")
            caller_seed = args[0]

        if caller_seed is None:
            scenario_offset = derive_seed(
                self._resume_base_seed,
                SeedDomain.ENVIRONMENT,
                self._reset_cursor,
            ) % self._resume_num_scenarios
            reset_seed = self._resume_base_seed + scenario_offset
            if args:
                args[0] = reset_seed
            else:
                kwargs["seed"] = reset_seed
        else:
            reset_seed = int(caller_seed)

        self._last_reset_seed = reset_seed
        self._reset_cursor += 1
        return tuple(args), kwargs

    def get_resume_state(self):
        """VecEnv-callable snapshot of deterministic environment-owned state."""
        if not hasattr(self, "_reset_cursor"):
            raise RuntimeError("environment resume state was not configured by make_env")
        perception_bridge = getattr(self, "_cognitive_perception_bridge", None)
        bias_bridge = getattr(self, "_cognitive_bias_bridge", None)
        state = {
            "version": _ENV_RESUME_STATE_VERSION,
            "rank": self._resume_rank,
            "base_seed": self._resume_base_seed,
            "num_scenarios": self._resume_num_scenarios,
            "reset_cursor": self._reset_cursor,
            "last_reset_seed": self._last_reset_seed,
            "traffic_density": copy.deepcopy(self.config.get("traffic_density")),
            "last_speed": copy.deepcopy(getattr(self, "_last_speed", None)),
            "cognitive_perception": (
                perception_bridge.get_resume_state() if perception_bridge is not None else None
            ),
            "cognitive_bias": (
                bias_bridge.get_resume_state() if bias_bridge is not None else None
            ),
        }
        return _encode_resume_value(state)

    def set_resume_state(self, state):
        """Restore a strict-resume snapshot, rejecting a different env structure."""
        if not isinstance(state, Mapping):
            raise TypeError("environment resume state must be a mapping")
        state = _decode_resume_value(state)
        if state.get("version") != _ENV_RESUME_STATE_VERSION:
            raise ValueError(
                f"unsupported environment resume state version: {state.get('version')!r}"
            )
        expected_identity = {
            "rank": self._resume_rank,
            "base_seed": self._resume_base_seed,
            "num_scenarios": self._resume_num_scenarios,
        }
        for name, expected in expected_identity.items():
            if state.get(name) != expected:
                raise ValueError(
                    f"environment {name} mismatch: checkpoint={state.get(name)!r}, "
                    f"runtime={expected!r}"
                )

        reset_cursor = state.get("reset_cursor")
        if not isinstance(reset_cursor, int) or reset_cursor < 0:
            raise ValueError(f"invalid environment reset_cursor: {reset_cursor!r}")

        perception_state = state.get("cognitive_perception")
        perception_bridge = getattr(self, "_cognitive_perception_bridge", None)
        if (perception_state is None) != (perception_bridge is None):
            raise ValueError("cognitive perception structure differs from checkpoint")
        if perception_bridge is not None:
            perception_bridge.validate_resume_state(perception_state)

        bias_state = state.get("cognitive_bias")
        bias_bridge = getattr(self, "_cognitive_bias_bridge", None)
        if (bias_state is None) != (bias_bridge is None):
            raise ValueError("cognitive bias structure differs from checkpoint")
        if bias_bridge is not None:
            bias_bridge.validate_resume_state(bias_state)

        self._reset_cursor = reset_cursor
        self._last_reset_seed = state.get("last_reset_seed")
        if "traffic_density" in state:
            self.config["traffic_density"] = copy.deepcopy(state["traffic_density"])
        if hasattr(self, "_last_speed"):
            self._last_speed = copy.deepcopy(state.get("last_speed"))
        if perception_bridge is not None:
            perception_bridge.set_resume_state(perception_state)
        if bias_bridge is not None:
            bias_bridge.set_resume_state(bias_state)


def _reset_cognitive_bridges(env):
    perception_bridge = getattr(env, "_cognitive_perception_bridge", None)
    if perception_bridge is not None:
        perception_bridge.reset()

    bias_bridge = getattr(env, "_cognitive_bias_bridge", None)
    if bias_bridge is not None:
        bias_bridge.reset()


def _attach_cognitive_bridges(env):
    perception_bridge = getattr(env, "_cognitive_perception_bridge", None)
    if perception_bridge is not None:
        perception_bridge.attach()


class CognitiveMetaDriveEnv(_DeterministicResumeEnvMixin, MetaDriveEnv):
    """MetaDrive environment that resets child-owned cognitive state per episode."""

    def setup_engine(self):
        super().setup_engine()
        _attach_cognitive_bridges(self)

    def reset(self, *args, **kwargs):
        args, kwargs = self._prepare_deterministic_reset(args, kwargs)
        _reset_cognitive_bridges(self)
        return super().reset(*args, **kwargs)


class SpeedControlMetaDriveEnv(_DeterministicResumeEnvMixin, MetaDriveEnv):
    def __init__(self, config, args=None):
        self.use_speed_control_reward = bool(
            config.get("use_speed_control_reward", False)
        )
        # Load speed-control parameters, preferring config values and falling back to CLI defaults.
        if args is not None:
            self.k = config.get("speed_control_k", args.speed_control_k)
            self.kappa = config.get("speed_control_kappa", args.speed_control_kappa)
            self.mu = config.get("speed_control_mu", args.speed_control_mu)
            self.nu = config.get("speed_control_nu", args.speed_control_nu)
            self.v_tol = config.get("speed_control_v_tolerance", args.speed_control_v_tolerance)
            self.v_ref = config.get("speed_control_v_ref", args.speed_control_v_ref)

            # Load the submodule enable flags.
            self.enable_tracking = config.get("speed_control_enable_tracking", args.speed_control_enable_tracking)
            self.enable_soft_wall = config.get("speed_control_enable_soft_wall", args.speed_control_enable_soft_wall)
            self.enable_behavior_guidance = config.get("speed_control_enable_behavior_guidance", args.speed_control_enable_behavior_guidance)
        else:
            # Backward compatibility: if args are missing, use the legacy hard-coded defaults.
            self.k = config.get("speed_control_k", _LEGACY_SPEED_CONTROL_K_DEFAULT)
            self.kappa = config.get("speed_control_kappa", _LEGACY_SPEED_CONTROL_KAPPA_DEFAULT)
            self.mu = config.get("speed_control_mu", _LEGACY_SPEED_CONTROL_MU_DEFAULT)
            self.nu = config.get("speed_control_nu", _LEGACY_SPEED_CONTROL_NU_DEFAULT)
            self.v_tol = config.get("speed_control_v_tolerance", _LEGACY_SPEED_CONTROL_V_TOLERANCE_DEFAULT)
            self.v_ref = config.get("speed_control_v_ref", _LEGACY_SPEED_CONTROL_V_REF_DEFAULT)

            self.enable_tracking = config.get("speed_control_enable_tracking", _LEGACY_SPEED_CONTROL_ENABLE_TRACKING_DEFAULT)
            self.enable_soft_wall = config.get("speed_control_enable_soft_wall", _LEGACY_SPEED_CONTROL_ENABLE_SOFT_WALL_DEFAULT)
            self.enable_behavior_guidance = config.get("speed_control_enable_behavior_guidance", _LEGACY_SPEED_CONTROL_ENABLE_BEHAVIOR_GUIDANCE_DEFAULT)

        # Create a MetaDrive-compatible config by stripping custom keys.
        metadrive_config = config.copy()
        speed_control_keys = [
            "use_speed_control_reward", "speed_control_k", "speed_control_kappa",
            "speed_control_mu", "speed_control_nu", "speed_control_v_tolerance", "speed_control_v_ref",
            "speed_control_enable_tracking", "speed_control_enable_soft_wall", "speed_control_enable_behavior_guidance"
        ]
        for key in speed_control_keys:
            if key in metadrive_config:
                del metadrive_config[key]

        # Call the parent constructor.
        super().__init__(metadrive_config)

        # Use dt from the engine or config when possible; the value below is only a fallback.
        self._dt = getattr(self.engine, "controller_step_interval", None) \
                   or getattr(self, "control_interval", None) \
                   or 0.05

        self._last_speed = None

        # The base speed reward is typically disabled in the external config, for example:
        # config["reward_config"]["speed_reward"] = 0.0

    def reset(self, *args, **kwargs):
        args, kwargs = self._prepare_deterministic_reset(args, kwargs)
        _reset_cognitive_bridges(self)
        obs = super().reset(*args, **kwargs)
        self._last_speed = None
        return obs

    def setup_engine(self):
        super().setup_engine()
        _attach_cognitive_bridges(self)

    @staticmethod
    def _huber(x, delta):
        ax = abs(x)
        return 0.5*ax*ax if ax <= delta else delta*(ax - 0.5*delta)

    def _compute_speed_control_reward(self, vehicle, action):
        import math

        v = float(vehicle.speed)  # m/s
        # Estimate acceleration from finite speed differences for better stability.
        if self._last_speed is None:
            a = 0.0
        else:
            a = (v - self._last_speed) / self._dt
        a_pos = max(a, 0.0)
        a_neg = max(-a, 0.0)

        dv = v - self.v_ref
        is_over = 1.0 if dv > 0.0 else 0.0

        # === Submodule reward terms (respecting enable flags) ===
        r_track = 0.0
        r_wall = 0.0
        r_act_over = 0.0

        # A: target-speed tracking term
        if self.enable_tracking:
            r_track = -self.k * self._huber(dv, self.v_tol)

        # B1: soft speed-wall term
        if self.enable_soft_wall:
            # softplus = log(1 + exp(x)); clamp first for numerical stability.
            dv_clip = max(min(dv, 20.0), -20.0)
            softp = math.log1p(math.exp(dv_clip))
            r_wall = -self.kappa * (softp ** 2) * is_over

        # B2: behavior-guidance term
        if self.enable_behavior_guidance:
            # Shape actions while speeding: encourage deceleration and penalize continued acceleration.
            r_act_over = is_over * (self.mu * a_neg - self.nu * a_pos)

        r_total = r_track + r_wall + r_act_over

        # Cache state for the next step.
        self._last_speed = v

        # Optionally expose each component through step_infos for debugging/info consumers.
        vid = vehicle.id
        if hasattr(self, "step_infos"):
            self.step_infos.setdefault(vid, {})
            self.step_infos[vid].update({
                "sc_r_total": r_total,
                "sc_r_track": r_track,
                "sc_r_wall": r_wall,
                "sc_r_act_over": r_act_over,
                "sc_v": v, "sc_v_ref": self.v_ref, "sc_dv": dv, "sc_a": a,
                # Record which submodules were enabled.
                "sc_enable_tracking": self.enable_tracking,
                "sc_enable_soft_wall": self.enable_soft_wall,
                "sc_enable_behavior_guidance": self.enable_behavior_guidance
            })

        return r_total

    def reward_function(self, vehicle_id: str) -> tuple:
        base_reward, step_info = super().reward_function(vehicle_id)

        if self.use_speed_control_reward:
            vehicle = self.agents[vehicle_id]
            terminal = (
                self._is_arrive_destination(vehicle)
                or self._is_out_of_road(vehicle)
                or vehicle.crash_vehicle
                or vehicle.crash_object
                or vehicle.crash_sidewalk
            )
            if terminal:
                step_info["speed_control_reward"] = 0.0
                return base_reward, step_info

            current_action = getattr(vehicle, 'current_action', [0.0, 0.0])
            sc_reward = self._compute_speed_control_reward(vehicle, current_action)
            reward = base_reward + sc_reward
            step_info["speed_control_reward"] = sc_reward
            step_info["step_reward"] = reward
            return reward, step_info

        return base_reward, step_info

def make_env(rank: int, resolved_config: Dict[str, Any], args):
    """
    Environment factory used to create vectorized environments
    Each subprocess runs an independent MetaDrive environment instance.
    Supports dynamic straight-road scenario generation so each environment gets distinct road lengths and traffic settings.

    Args:
        rank: environment index
        resolved_config: per-environment config for this rank from the resolved runtime config
        args: command-line arguments object containing the defaults from add_arguments

    Returns:
        environment constructor
    """
    def _init():
        if resolved_config.get("rank") != rank:
            raise ValueError(
                "resolved environment rank mismatch: "
                f"expected {rank}, got {resolved_config.get('rank')!r}"
            )
        worker_seed = int(resolved_config["worker_seed"])
        if int(getattr(args, "n_envs", 1)) > 1:
            seed_global_generators(worker_seed)
        env_config = copy.deepcopy(resolved_config["metadrive_config"])
        env_seed = int(env_config["start_seed"])
        scenario = resolved_config["scenario"]
        if env_config["map"] != scenario["map"] or env_seed != scenario["seed"]:
            raise ValueError("resolved scenario and MetaDrive config disagree")

        # Select the environment class from the resolved config.
        if env_config.get("use_speed_control_reward", False):
            env = SpeedControlMetaDriveEnv(env_config, args)
        else:
            # Use the standard MetaDrive environment.
            env = CognitiveMetaDriveEnv(env_config)

        env._resolved_env_config = copy.deepcopy(resolved_config)
        env._configure_resume_state(
            rank=rank,
            base_seed=env_seed,
            num_scenarios=env_config.get("num_scenarios", 1),
        )

        # Attach a helper for dynamically updating traffic density.
        def update_traffic_density(new_density):
            """Dynamically update traffic density"""
            if hasattr(env, 'config'):
                env.config['traffic_density'] = new_density
                # If the environment ever supports runtime config updates, extend it here.

        env.update_traffic_density = update_traffic_density

        perception_bridge = None
        cognitive_enabled = getattr(args, "use_cognitive_modules", False)
        if cognitive_enabled and getattr(args, "use_cognitive_perception", False):
            perception_bridge = CognitivePerceptionEnvBridge(env, args, env_seed)
            env._cognitive_perception_bridge = perception_bridge
            env.set_cognitive_perception_parameters = perception_bridge.set_parameters
            env.reset_cognitive_perception = perception_bridge.reset

        if cognitive_enabled and getattr(args, "use_cognitive_bias", False):
            bias_bridge = CognitiveBiasEnvBridge(
                env,
                args,
                perception_module=(
                    perception_bridge.module if perception_bridge is not None else None
                ),
            )
            env._cognitive_bias_bridge = bias_bridge
            env.process_cognitive_bias_reward = bias_bridge.process_reward
            env.set_cognitive_bias_inverse_tta_coef = bias_bridge.set_inverse_tta_coef
            env.reset_cognitive_bias = bias_bridge.reset

        return env

    return _init
