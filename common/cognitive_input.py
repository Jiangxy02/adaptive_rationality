"""Single source of truth for the paper-aligned cognitive input schema."""

import math

BASE_OBSERVATION_DIM = 275
COGNITIVE_PARAM_DIM = 4
COGNITIVE_MASK_DIM = 4
PERCEPTION_SIGMA_UNIT = "meters"
PUBLIC_OBSERVATION_DIM = (
    BASE_OBSERVATION_DIM + COGNITIVE_PARAM_DIM + COGNITIVE_MASK_DIM
)


def validate_perception_sigmas(sigma0, sigma_max):
    """Return finite, non-negative paper-domain perception sigmas in meters."""
    sigma0 = float(sigma0)
    sigma_max = float(sigma_max)
    if not math.isfinite(sigma0) or not math.isfinite(sigma_max):
        raise ValueError("perception sigma0 and sigma_max must be finite meters")
    if sigma0 < 0 or sigma_max < 0:
        raise ValueError("perception sigma0 and sigma_max must be non-negative meters")
    if sigma0 > sigma_max:
        raise ValueError(
            "perception sigma0 must be <= perception sigma_max "
            f"(got sigma0={sigma0}, sigma_max={sigma_max})"
        )
    return sigma0, sigma_max


def cognitive_mask_values(
    *,
    effects_enabled: bool,
    bias_enabled: bool,
    perception_enabled: bool,
    delay_enabled: bool,
):
    """Return masks in paper order: bias, sigma0, sigma_max, delay."""
    if not effects_enabled:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(bool(bias_enabled)),
        float(bool(perception_enabled)),
        float(bool(perception_enabled)),
        float(bool(delay_enabled)),
    )
