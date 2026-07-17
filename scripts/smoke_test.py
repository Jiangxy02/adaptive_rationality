#!/usr/bin/env python3
"""Verify that the public installation can import its required runtime modules."""

import importlib
import sys


def version_of(module, name):
    if name == "metadrive":
        return importlib.import_module("metadrive.version").VERSION
    return getattr(module, "__version__", "unknown")


def check_dependencies():
    for name in (
        "numpy",
        "pandas",
        "torch",
        "gymnasium",
        "stable_baselines3",
        "metadrive",
    ):
        module = importlib.import_module(name)
        print(f"{name} {version_of(module, name)}")


def check_project_modules():
    module_names = (
        "ppo_train.models.ppo_network",
        "ppo_train.envs.speed_control_env",
        "ppo_train.envs.env_factory",
        "ppo_train.config.arguments",
        "ppo_train.config.build_config",
        "ppo_train.training.trainer",
        "ppo_train.training.scheduling",
        "ppo_train.training.rollout",
        "ppo_train.training.ppo_update",
        "ppo_train.training.evaluation",
        "ppo_train.training.checkpointing",
        "ppo_train.training.metrics_logging",
        "ppo_train.training.reporting",
        "ppo_train.training.cognitive_integration",
        "evaluation.evaluate_model",
        "ppo_train.scripts.train",
    )
    for name in module_names:
        importlib.import_module(name)
        print(f"import {name}")

    from ppo_train.config.arguments import add_arguments

    add_arguments().parse_args([])
    print("add_arguments parse_args [] OK")


def main():
    ok = True
    for title, check in (
        ("Dependency imports", check_dependencies),
        ("Project module imports", check_project_modules),
    ):
        try:
            check()
        except Exception as exc:
            print(f"[FAIL] {title}: {exc}")
            ok = False

    if ok:
        print("SMOKE PASS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
