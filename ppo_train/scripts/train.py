#!/usr/bin/env python3
"""Thin training entrypoint for the refactored PPO trainer."""


import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.cognitive_input import PUBLIC_OBSERVATION_DIM
from common.headless import apply_headless_guard
apply_headless_guard()

from ppo_train.training.trainer import PPOExpertReproduction
from ppo_train.config.arguments import add_arguments


def main():
    """Command-line entrypoint"""
    parser = add_arguments()
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resume_from:
        mode = (
            f"Strict resume | checkpoint={args.resume_from} | "
            f"exists={os.path.exists(args.resume_from)}"
        )
    elif args.warm_start_from:
        mode = (
            f"Weight warm start | checkpoint={args.warm_start_from} | "
            f"exists={os.path.exists(args.warm_start_from)}"
        )
    else:
        mode = "Fresh training"

    if args.use_cognitive_modules:
        active_modules = "+".join(
            name
            for name, enabled in (
                ("bias", args.use_cognitive_bias),
                ("perception", args.use_cognitive_perception),
                ("delay", args.use_cognitive_delay),
            )
            if enabled
        ) or "no active effect modules"
        if args.use_cognitive_parameter_sampling:
            sampler_type = (
                "discrete" if args.cognitive_sampler_type == "discrete" else "continuous"
            )
            sampler = (
                f"Parameter sampler: enabled ({sampler_type}) | "
                f"every {args.cognitive_param_update_steps} steps"
            )
        else:
            sampler = "Parameter sampler: disabled (fixed parameters)"
        cognitive_summary = (
            f"{active_modules} | modulation={args.cognitive_modulation} | {sampler}"
        )
    else:
        cognitive_summary = "disabled"

    print("PPO Training")
    print("=" * 50)
    print(f"Mode: {mode}")
    print(
        f"Training setup: {args.total_timesteps:,}  steps | {args.n_envs} environments | "
        f"input {PUBLIC_OBSERVATION_DIM} dims | {args.device} | seed={args.seed}"
    )
    print(
        f"PPO: lr={args.lr} | rollout={args.n_steps} | batch={args.batch_size} | "
        f"epochs={args.n_epochs} | clip={args.clip_range}"
    )
    print(f"Cognitive: {cognitive_summary}")
    print("=" * 50)

    trainer = PPOExpertReproduction(args)
    trainer.train()


if __name__ == "__main__":
    main()
