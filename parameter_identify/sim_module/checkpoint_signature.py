"""Checkpoint network signature parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from common.cognitive_input import (
    BASE_OBSERVATION_DIM,
    COGNITIVE_MASK_DIM,
    COGNITIVE_PARAM_DIM,
    PERCEPTION_SIGMA_UNIT,
    PUBLIC_OBSERVATION_DIM,
)
from common.ppo_network_base import PPONetworkBase


DEFAULT_BASE_OBS_DIM = BASE_OBSERVATION_DIM
CURRENT_CHECKPOINT_SCHEMA_VERSION = 6
CURRENT_CHECKPOINT_KIND = "exact_training_state"
CURRENT_REWARD_CONTRACT = "metadrive_tuple_speed_control_v1"
CURRENT_PPO_UPDATE_CONTRACT = "latent_tanh_squashed_entropy_transactional_kl_v2"
CURRENT_PUBLIC_OBSERVATION_DIM = PUBLIC_OBSERVATION_DIM
CURRENT_PUBLIC_COGNITIVE_PARAM_DIM = COGNITIVE_PARAM_DIM
CURRENT_PUBLIC_COGNITIVE_MASK_DIM = COGNITIVE_MASK_DIM


@dataclass(frozen=True)
class CheckpointSignature:
    """Network dimensions required to recreate a checkpoint exactly."""

    observation_dim: int
    base_obs_dim: int
    action_dim: int
    hidden_dim: int
    cognitive_param_dim: int
    cognitive_mask_dim: int
    cognitive_modulation: str

    @property
    def uses_cognitive_modules(self) -> bool:
        return self.cognitive_modulation != "none" and self.cognitive_param_dim > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_dim": self.observation_dim,
            "obs_dim": self.observation_dim,
            "base_obs_dim": self.base_obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "cognitive_param_dim": self.cognitive_param_dim,
            "cognitive_params_dim": self.cognitive_param_dim,
            "cognitive_mask_dim": self.cognitive_mask_dim,
            "cognitive_masks_dim": self.cognitive_mask_dim,
            "cognitive_modulation": self.cognitive_modulation,
            "uses_cognitive_modules": self.uses_cognitive_modules,
        }

    def network_kwargs(self) -> Dict[str, Any]:
        return {
            "base_obs_dim": self.base_obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "cognitive_param_dim": self.cognitive_param_dim,
            "cognitive_mask_dim": self.cognitive_mask_dim,
            "cognitive_modulation": self.cognitive_modulation,
            "checkpoint_obs_dim": self.observation_dim,
        }


def load_checkpoint_file(path: str, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load tensor-only checkpoint data without executing pickle globals."""
    return torch.load(path, map_location=map_location, weights_only=True)


def validate_current_public_checkpoint(checkpoint: Mapping[str, Any]) -> None:
    """Reject unpublished legacy checkpoint formats at every public loader."""
    schema_version = checkpoint.get("schema_version")
    checkpoint_kind = checkpoint.get("checkpoint_kind")
    perception_sigma_unit = checkpoint.get("perception_sigma_unit")
    reward_contract = checkpoint.get("reward_contract")
    ppo_update_contract = checkpoint.get("ppo_update_contract")
    if (
        schema_version != CURRENT_CHECKPOINT_SCHEMA_VERSION
        or checkpoint_kind != CURRENT_CHECKPOINT_KIND
        or perception_sigma_unit != PERCEPTION_SIGMA_UNIT
        or reward_contract != CURRENT_REWARD_CONTRACT
        or ppo_update_contract != CURRENT_PPO_UPDATE_CONTRACT
    ):
        raise ValueError(
            "current public checkpoint required: "
            f"schema_version={CURRENT_CHECKPOINT_SCHEMA_VERSION}, "
            f"checkpoint_kind={CURRENT_CHECKPOINT_KIND!r}; "
            f"perception_sigma_unit={PERCEPTION_SIGMA_UNIT!r}; "
            f"reward_contract={CURRENT_REWARD_CONTRACT!r}; "
            f"ppo_update_contract={CURRENT_PPO_UPDATE_CONTRACT!r}; "
            f"got schema_version={schema_version!r}, checkpoint_kind={checkpoint_kind!r}, "
            f"perception_sigma_unit={perception_sigma_unit!r}, "
            f"reward_contract={reward_contract!r}, "
            f"ppo_update_contract={ppo_update_contract!r}"
        )

    config = checkpoint.get("config")
    network = config.get("network") if isinstance(config, Mapping) else None
    if not isinstance(network, Mapping):
        raise ValueError("current public checkpoint missing config.network signature")


def validate_current_public_signature(signature: "CheckpointSignature") -> None:
    """Require the single network structure supported by the public release."""
    if (
        signature.observation_dim != CURRENT_PUBLIC_OBSERVATION_DIM
        or signature.base_obs_dim != DEFAULT_BASE_OBS_DIM
        or signature.cognitive_param_dim != CURRENT_PUBLIC_COGNITIVE_PARAM_DIM
        or signature.cognitive_mask_dim != CURRENT_PUBLIC_COGNITIVE_MASK_DIM
        or signature.cognitive_modulation != "concat"
    ):
        raise ValueError(
            "current public checkpoint must describe the 283-dimensional concat network; "
            f"got signature={signature.to_dict()}"
        )


def load_current_public_network(
    path: str,
    map_location: Any = "cpu",
) -> Tuple[Dict[str, Any], "CheckpointSignature", PPONetworkBase]:
    """Construct the current 283-dimensional network from its checkpoint signature."""
    checkpoint = load_checkpoint_file(path, map_location=map_location)
    validate_current_public_checkpoint(checkpoint)
    state_dict = extract_network_state_dict(checkpoint)
    signature = parse_checkpoint_signature(checkpoint, state_dict)
    validate_current_public_signature(signature)

    network = PPONetworkBase(**signature.network_kwargs()).to(map_location)
    network.checkpoint_signature = signature.to_dict()
    try:
        network.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "strict current checkpoint load failed; "
            f"signature={signature.to_dict()}. Original error: {exc}"
        ) from exc
    network.eval()
    checkpoint["network_state_dict"] = state_dict
    return checkpoint, signature, network


def normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove common wrapper prefixes from a network state dict."""
    normalized: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("_orig_mod.", "module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def extract_network_state_dict(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a normalized network state dict from a checkpoint."""
    if "network_state_dict" not in checkpoint:
        raise KeyError("checkpoint missing required key 'network_state_dict'")
    state_dict = checkpoint["network_state_dict"]
    if not isinstance(state_dict, Mapping):
        raise TypeError("checkpoint['network_state_dict'] must be a mapping")
    return normalize_state_dict_keys(state_dict)


def parse_checkpoint_signature(
    checkpoint: Mapping[str, Any],
    state_dict: Optional[Mapping[str, Any]] = None,
    base_default: int = DEFAULT_BASE_OBS_DIM,
) -> CheckpointSignature:
    """Parse or infer the network signature needed for strict checkpoint loading."""
    sd = normalize_state_dict_keys(state_dict or extract_network_state_dict(checkpoint))
    network_cfg = _network_config(checkpoint)
    args = _args_config(checkpoint)

    actor_fc1 = _required_tensor(sd, "actor_fc1.weight")
    observation_dim = _as_int(
        _first_present(network_cfg, "observation_dim", "obs_dim", "input_dim"),
        actor_fc1.shape[1],
    )
    weight_observation_dim = int(actor_fc1.shape[1])
    if observation_dim != weight_observation_dim:
        observation_dim = weight_observation_dim

    hidden_dim = _as_int(_first_present(network_cfg, "hidden_dim"), int(actor_fc1.shape[0]))
    action_dim = _infer_action_dim(sd, network_cfg)
    base_obs_dim = _infer_base_obs_dim(sd, network_cfg, args, base_default)

    modulation = _normalize_modulation(
        _first_present(network_cfg, "cognitive_modulation")
        or _first_present(args, "cognitive_modulation")
    )

    cognitive_param_dim = _as_optional_int(
        _coalesce_present(
            _first_present(
                network_cfg,
                "cognitive_params_dim",
                "cognitive_param_dim",
                "cognitive_dim",
            ),
            _first_present(args, "cognitive_params_dim", "cognitive_param_dim"),
        )
    )

    if modulation is None:
        modulation = "concat" if observation_dim > base_obs_dim else "none"

    cognitive_norm = sd.get("cognitive_norm.weight")
    if cognitive_param_dim is None and cognitive_norm is not None and modulation == "concat":
        cognitive_param_dim = min(int(cognitive_norm.shape[0]), max(observation_dim - base_obs_dim, 0))
    if cognitive_param_dim is None:
        cognitive_param_dim = max(observation_dim - base_obs_dim, 0) if modulation == "concat" else 0

    cognitive_param_dim = max(int(cognitive_param_dim or 0), 0)

    cognitive_mask_dim = _as_optional_int(
        _coalesce_present(
            _first_present(
                network_cfg,
                "cognitive_mask_dim",
                "cognitive_masks_dim",
            ),
            _first_present(args, "cognitive_mask_dim", "cognitive_masks_dim"),
        )
    )
    if cognitive_mask_dim is None:
        cognitive_mask_dim = (
            max(observation_dim - base_obs_dim - cognitive_param_dim, 0)
            if modulation == "concat"
            else 0
        )
    cognitive_mask_dim = max(int(cognitive_mask_dim), 0)

    cognitive_enabled = _first_present(network_cfg, "cognitive_params_integration")
    if cognitive_enabled is False and observation_dim == base_obs_dim:
        cognitive_param_dim = 0
        cognitive_mask_dim = 0
        modulation = "none"

    if cognitive_param_dim == 0 and observation_dim == base_obs_dim:
        modulation = "none"

    if modulation == "none":
        cognitive_param_dim = 0
        cognitive_mask_dim = 0

    signature = CheckpointSignature(
        observation_dim=int(observation_dim),
        base_obs_dim=int(base_obs_dim),
        action_dim=int(action_dim),
        hidden_dim=int(hidden_dim),
        cognitive_param_dim=int(cognitive_param_dim),
        cognitive_mask_dim=int(cognitive_mask_dim),
        cognitive_modulation=str(modulation),
    )
    _validate_signature(signature, sd)
    return signature


def parse_checkpoint_signature_from_path(path: str, map_location: Any = "cpu") -> CheckpointSignature:
    checkpoint = load_checkpoint_file(path, map_location=map_location)
    validate_current_public_checkpoint(checkpoint)
    signature = parse_checkpoint_signature(checkpoint)
    validate_current_public_signature(signature)
    return signature


def _network_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        return {}
    network = config.get("network", {})
    return dict(network) if isinstance(network, Mapping) else {}


def _args_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    args = checkpoint.get("args", {})
    if isinstance(args, Mapping):
        return dict(args)
    return vars(args) if hasattr(args, "__dict__") else {}


def _first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _coalesce_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    return int(value)


def _as_int(value: Any, default: int) -> int:
    return default if value is None else int(value)


def _normalize_modulation(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).lower()
    return None if value == "auto" else value


def _required_tensor(state_dict: Mapping[str, Any], key: str) -> torch.Tensor:
    if key not in state_dict:
        raise KeyError(f"checkpoint network_state_dict missing required key '{key}'")
    tensor = state_dict[key]
    if not hasattr(tensor, "shape"):
        raise TypeError(f"checkpoint key '{key}' is not tensor-like")
    return tensor


def _infer_action_dim(state_dict: Mapping[str, Any], network_cfg: Mapping[str, Any]) -> int:
    configured = _first_present(network_cfg, "action_dim")
    if configured is not None:
        return int(configured)
    actor_out = _required_tensor(state_dict, "actor_out.weight")
    return int(actor_out.shape[0] // 2)


def _infer_base_obs_dim(
    state_dict: Mapping[str, Any],
    network_cfg: Mapping[str, Any],
    args: Mapping[str, Any],
    base_default: int,
) -> int:
    configured = (
        _first_present(network_cfg, "base_observation_dim", "base_obs_dim")
        or _first_present(args, "base_observation_dim", "base_obs_dim")
    )
    if configured is not None:
        return int(configured)
    obs_norm = state_dict.get("obs_norm.weight")
    if obs_norm is not None and hasattr(obs_norm, "shape"):
        return int(obs_norm.shape[0])
    return int(base_default)


def _validate_signature(signature: CheckpointSignature, state_dict: Mapping[str, Any]) -> None:
    if signature.observation_dim <= 0:
        raise ValueError(f"invalid checkpoint observation_dim={signature.observation_dim}")
    if signature.base_obs_dim <= 0:
        raise ValueError(f"invalid checkpoint base_obs_dim={signature.base_obs_dim}")
    if signature.cognitive_modulation not in {"none", "concat"}:
        raise ValueError(f"invalid checkpoint cognitive_modulation={signature.cognitive_modulation!r}")

    actor_in = int(_required_tensor(state_dict, "actor_fc1.weight").shape[1])
    if actor_in != signature.observation_dim:
        raise ValueError(
            "checkpoint signature disagrees with actor_fc1.weight: "
            f"signature observation_dim={signature.observation_dim}, weight input={actor_in}"
        )

    if signature.cognitive_modulation == "none":
        expected_actor_in = signature.base_obs_dim
    else:
        expected_actor_in = (
            signature.base_obs_dim
            + signature.cognitive_param_dim
            + signature.cognitive_mask_dim
        )

    if signature.observation_dim != expected_actor_in:
        raise ValueError(
            "checkpoint signature invariant failed: "
            f"modulation={signature.cognitive_modulation}, "
            f"observation_dim={signature.observation_dim}, "
            f"expected={expected_actor_in}, "
            f"base_obs_dim={signature.base_obs_dim}, "
            f"cognitive_param_dim={signature.cognitive_param_dim}, "
            f"cognitive_mask_dim={signature.cognitive_mask_dim}"
        )
