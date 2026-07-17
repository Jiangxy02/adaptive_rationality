"""
Checkpoint loader utilities.

This module handles PPO checkpoint signature parsing, network initialization,
and strict weight loading.
"""


import os
from typing import Any, Dict, Optional

import torch

from common.ppo_network_base import PPONetworkBase as PPONetwork
from parameter_identify.sim_module.checkpoint_signature import (
    CheckpointSignature,
    extract_network_state_dict,
    load_checkpoint_file,
    normalize_state_dict_keys,
    parse_checkpoint_signature,
    validate_current_public_checkpoint,
    validate_current_public_signature,
)


class CheckpointLoader:
    """PPO checkpoint loader.

    The loader only trusts the checkpoint's self-described signature and
    weight shapes. The constructed network must accept the original weights
    with ``strict=True`` or loading fails immediately.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        self.checkpoint: Optional[Dict[str, Any]] = None
        self.network: Optional[PPONetwork] = None
        self.signature: Optional[CheckpointSignature] = None
        self._checkpoint_obs_dim: Optional[int] = None

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    def load_checkpoint(
        self,
        use_cognitive_modules: bool = False,
        cognitive_modulation_override: Optional[str] = None,
    ) -> PPONetwork:
        """Load the checkpoint and create a shape-compatible network."""
        print(f"Loading checkpoint: {os.path.basename(self.checkpoint_path)}")

        self.checkpoint = load_checkpoint_file(self.checkpoint_path, map_location=self.device)
        validate_current_public_checkpoint(self.checkpoint)
        self._normalize_state_dict_keys()

        self.signature = parse_checkpoint_signature(self.checkpoint, self.checkpoint["network_state_dict"])
        validate_current_public_signature(self.signature)
        self._checkpoint_obs_dim = self.signature.observation_dim
        self._validate_requested_override(cognitive_modulation_override)

        network_config = self._build_network_config()
        self.network = PPONetwork(
            base_obs_dim=network_config["base_obs_dim"],
            action_dim=network_config["action_dim"],
            hidden_dim=network_config["hidden_dim"],
            cognitive_param_dim=network_config["cognitive_param_dim"],
            cognitive_mask_dim=network_config["cognitive_mask_dim"],
            cognitive_modulation=network_config["cognitive_modulation"],
            checkpoint_obs_dim=self.signature.observation_dim,
        ).to(self.device)
        self.network.checkpoint_signature = self.signature.to_dict()

        self._validate_architecture(self.checkpoint["network_state_dict"], self.network.state_dict())
        self._load_network_weights()

        self.network.eval()
        return self.network

    def _validate_requested_override(self, cognitive_modulation_override: Optional[str]) -> None:
        if not cognitive_modulation_override or cognitive_modulation_override == "auto":
            return
        if self.signature is None:
            raise RuntimeError("checkpoint signature has not been parsed")
        if cognitive_modulation_override != self.signature.cognitive_modulation:
            raise ValueError(
                "cognitive_modulation override conflicts with checkpoint signature: "
                f"override={cognitive_modulation_override}, "
                f"checkpoint={self.signature.cognitive_modulation}. "
                "Use --cognitive_modulation auto or a matching value."
            )

    def _validate_architecture(self, ckpt_sd: Dict[str, Any], model_sd: Dict[str, Any]) -> None:
        """Check all shared weight names and fail immediately on shape mismatch."""
        mismatches = []
        for key, ckpt_tensor in ckpt_sd.items():
            if key in model_sd and tuple(ckpt_tensor.shape) != tuple(model_sd[key].shape):
                mismatches.append((key, tuple(ckpt_tensor.shape), tuple(model_sd[key].shape)))

        if mismatches:
            details = "; ".join(f"{key}: ckpt{ckpt_shape} vs model{model_shape}" for key, ckpt_shape, model_shape in mismatches[:10])
            if len(mismatches) > 10:
                details += f"; ... plus {len(mismatches) - 10} more layers"
            raise RuntimeError(
                "Checkpoint weight shapes do not match the network built from the signature; refusing to load: "
                f"{details}"
            )

        print("Critical layer shapes match the checkpoint signature")

    def _load_network_weights(self) -> None:
        """Load network weights strictly."""
        if self.checkpoint is None or self.network is None:
            raise RuntimeError("checkpoint and network must be ready before loading weights")

        state_dict = self.checkpoint["network_state_dict"]
        try:
            self.network.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            signature = self.signature.to_dict() if self.signature else {}
            raise RuntimeError(
                "strict checkpoint load failed; checkpoint and constructed network differ. "
                f"signature={signature}. Original error: {exc}"
            ) from exc
        print("Weights loaded successfully (strict=True)")

    def _normalize_state_dict_keys(self) -> None:
        """Normalize state-dict keys from the checkpoint by removing wrapper prefixes."""
        if not self.checkpoint or "network_state_dict" not in self.checkpoint:
            return
        self.checkpoint["network_state_dict"] = normalize_state_dict_keys(
            extract_network_state_dict(self.checkpoint)
        )

    def get_network(self) -> Optional[PPONetwork]:
        """Return the loaded network instance."""
        return self.network

    def get_checkpoint_info(self) -> dict:
        """Return checkpoint metadata."""
        if not self.checkpoint:
            return {}

        info = {
            "iteration": self.checkpoint.get("iteration", "Unknown"),
            "global_step": self.checkpoint.get("global_step", "Unknown"),
            "checkpoint_obs_dim": self._checkpoint_obs_dim,
            "checkpoint_path": self.checkpoint_path,
        }
        if self.signature:
            info["signature"] = self.signature.to_dict()
        return info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_network_config(self) -> Dict[str, Any]:
        """Build the network initialization config from the checkpoint signature."""
        if self.signature is None:
            raise RuntimeError("checkpoint signature has not been parsed")

        print(
            "Checkpoint signature: "
            f"observation_dim={self.signature.observation_dim}, "
            f"base={self.signature.base_obs_dim}, "
            f"cognitive_param_dim={self.signature.cognitive_param_dim}, "
            f"cognitive_mask_dim={self.signature.cognitive_mask_dim}, "
            f"cognitive_modulation={self.signature.cognitive_modulation}"
        )

        return {
            "base_obs_dim": self.signature.base_obs_dim,
            "action_dim": self.signature.action_dim,
            "hidden_dim": self.signature.hidden_dim,
            "cognitive_param_dim": self.signature.cognitive_param_dim,
            "cognitive_mask_dim": self.signature.cognitive_mask_dim,
            "cognitive_modulation": self.signature.cognitive_modulation,
        }
