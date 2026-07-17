"""Deterministic random-seed setup and hierarchical seed derivation."""

import os
import random
from enum import IntEnum
from typing import Union

import numpy as np


class SeedDomain(IntEnum):
    """Stable namespaces used in ``SeedSequence.spawn_key`` paths."""

    SAMPLER = 1
    WORKER = 2
    ENVIRONMENT = 3
    WINDOW = 4
    PARTICLE = 5
    SAMPLE = 6
    POLICY = 7
    PERCEPTION = 8
    MC_RANDOM = 9


SeedPathPart = Union[int, SeedDomain]


def _validate_seed(seed: int) -> int:
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return seed


def derive_seed(root_seed: int, *path: SeedPathPart) -> int:
    """Derive a stable positive 31-bit seed from a root seed and logical path."""
    root_seed = _validate_seed(root_seed)
    spawn_key = tuple(_validate_seed(part) for part in path)
    sequence = np.random.SeedSequence(root_seed, spawn_key=spawn_key)
    raw_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
    return raw_seed % (2**31 - 2) + 1


def seed_global_generators(seed: int) -> int:
    """Seed Python, NumPy, Torch, and every visible CUDA device."""
    seed = _validate_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
