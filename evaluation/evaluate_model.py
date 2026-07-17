#!/usr/bin/env python3
"""Evaluate a current public PPO checkpoint under a fixed protocol."""


import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_path = str(REPO_ROOT)
if repo_root_path in sys.path:
    sys.path.remove(repo_root_path)
sys.path.insert(0, repo_root_path)

import json
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from typing import Dict, List
import matplotlib.pyplot as plt


from metadrive.envs.metadrive_env import MetaDriveEnv
from parameter_identify.sim_module.checkpoint_signature import load_current_public_network
from ppo_train.envs.speed_control_env import SpeedControlMetaDriveEnv


EVALUATION_PROTOCOL = "fixed_no_disturbance_v3"
EVALUATION_ACTION_MODE = "deterministic"
EVALUATION_RENDER = False
LANE_CHANGE_DEFINITION = "same_road_segment_integer_lane_id_change"
LEGACY_EVALUATION_ARTIFACTS = (
    "performance_metrics.png",
    "performance_radar.png",
)


def _is_lateral_lane_change(previous_lane_index, current_lane_index) -> bool:
    """Count only an integer lane-id change within the same road segment."""
    if not isinstance(previous_lane_index, (tuple, list)):
        return False
    if not isinstance(current_lane_index, (tuple, list)):
        return False
    if len(previous_lane_index) < 3 or len(current_lane_index) < 3:
        return False
    previous_lane_id = previous_lane_index[2]
    current_lane_id = current_lane_index[2]
    if isinstance(previous_lane_id, (bool, np.bool_)) or isinstance(
        current_lane_id, (bool, np.bool_)
    ):
        return False
    integer_types = (int, np.integer)
    if not isinstance(previous_lane_id, integer_types) or not isinstance(
        current_lane_id, integer_types
    ):
        return False
    return (
        tuple(previous_lane_index[:2]) == tuple(current_lane_index[:2])
        and previous_lane_id != current_lane_id
    )


class ModelEvaluator:
    """Model evaluator"""

    def __init__(self, model_path: str, config_path: str, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")

        # Load the config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.checkpoint, self.signature, self.network = load_current_public_network(
            model_path,
            map_location=self.device,
        )
        self.neutral_cognitive_params = np.zeros(
            self.signature.cognitive_param_dim,
            dtype=np.float32,
        )
        self.neutral_cognitive_mask = np.zeros(
            self.signature.cognitive_mask_dim,
            dtype=np.float32,
        )

        print(f"Loaded model: {model_path}")
        print(f"Device: {self.device}")
        print(f"Evaluation protocol: {EVALUATION_PROTOCOL}")

    def _prepare_observation(self, observation: np.ndarray) -> np.ndarray:
        """Append neutral cognitive parameters and masks for the 283-D model."""
        base_observation = np.asarray(observation, dtype=np.float32)
        if base_observation.shape != (self.signature.base_obs_dim,):
            raise ValueError(
                "evaluation observation does not match checkpoint base dimension: "
                f"expected {(self.signature.base_obs_dim,)}, got {base_observation.shape}"
            )
        return np.concatenate((
            base_observation,
            self.neutral_cognitive_params,
            self.neutral_cognitive_mask,
        ))

    def _evaluation_env_config(self, seed: int) -> Dict:
        """Build one fixed-seed environment from the current resolved runtime config."""
        runtime = self.config.get("resolved_runtime_config")
        environments = runtime.get("environments") if isinstance(runtime, dict) else None
        if not environments or not isinstance(environments[0], dict):
            raise ValueError("current config missing resolved_runtime_config.environments[0]")
        metadrive_config = environments[0].get("metadrive_config")
        if not isinstance(metadrive_config, dict):
            raise ValueError("current config missing resolved MetaDrive configuration")

        env_config = copy.deepcopy(metadrive_config)
        env_config["start_seed"] = int(seed)
        env_config["num_scenarios"] = 1
        env_config["use_render"] = EVALUATION_RENDER
        return env_config

    @staticmethod
    def _create_evaluation_env(env_config: Dict):
        if env_config.get("use_speed_control_reward", False):
            env = SpeedControlMetaDriveEnv(env_config, args=None)
            env._configure_resume_state(
                rank=0,
                base_seed=int(env_config["start_seed"]),
                num_scenarios=int(env_config["num_scenarios"]),
            )
            return env
        return MetaDriveEnv(env_config)

    def evaluate_single_episode(self, env) -> Dict:
        """Evaluate one episode"""
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        collision = False
        offroad = False
        success = False
        timeout = False

        lane_changes = 0
        prev_lane_idx = None

        while True:
            model_observation = self._prepare_observation(obs)
            obs_tensor = torch.as_tensor(
                model_observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                action = self.network.act_deterministic(obs_tensor).cpu().numpy()[0]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Count lane changes
            navigation = getattr(env.agent, 'navigation', None)
            current_lane = getattr(navigation, 'current_lane', None)
            if current_lane is not None:
                current_lane_idx = getattr(current_lane, 'index', None)
                if _is_lateral_lane_change(prev_lane_idx, current_lane_idx):
                    lane_changes += 1
                prev_lane_idx = current_lane_idx

            if terminated or truncated:
                # Determine the termination reason
                if info.get("crash", False) or info.get("crash_vehicle", False):
                    collision = True
                elif info.get("out_of_road", False):
                    offroad = True
                elif info.get("arrive_dest", False) or info.get("success", False):
                    success = True
                else:
                    timeout = True
                break

        return {
            "reward": episode_reward,
            "length": episode_length,
            "collision": collision,
            "offroad": offroad,
            "success": success,
            "timeout": timeout,
            "lane_changes": lane_changes
        }

    def evaluate_multiple_episodes(self, num_episodes: int = 50, seeds: List[int] = None) -> Dict:
        """Evaluate multiple episodes"""
        if seeds is None:
            seeds = list(range(100, 100 + num_episodes))

        all_results = []

        for i, seed in enumerate(seeds[:num_episodes]):
            # Build the environment config
            env_config = self._evaluation_env_config(seed)
            env = self._create_evaluation_env(env_config)

            try:
                result = self.evaluate_single_episode(env)
                result["seed"] = seed
                result["episode"] = i
                all_results.append(result)

                if (i + 1) % 10 == 0:
                    print(f"Completed evaluation: {i + 1}/{num_episodes}")

            finally:
                env.close()

        # Compute aggregate statistics
        df = pd.DataFrame(all_results)

        stats = {
            "protocol": EVALUATION_PROTOCOL,
            "lane_change_definition": LANE_CHANGE_DEFINITION,
            "seed_list": [int(result["seed"]) for result in all_results],
            "checkpoint_signature": self.signature.to_dict(),
            "neutral_cognitive_params": self.neutral_cognitive_params.tolist(),
            "neutral_cognitive_mask": self.neutral_cognitive_mask.tolist(),
            "num_episodes": len(all_results),
            "reward_mean": df["reward"].mean(),
            "reward_std": df["reward"].std(),
            "reward_min": df["reward"].min(),
            "reward_max": df["reward"].max(),
            "length_mean": df["length"].mean(),
            "length_std": df["length"].std(),
            "collision_rate": df["collision"].mean(),
            "offroad_rate": df["offroad"].mean(),
            "success_rate": df["success"].mean(),
            "timeout_rate": df["timeout"].mean(),
            "lane_changes_mean": df["lane_changes"].mean(),
            "raw_results": all_results
        }

        return stats

    def generate_evaluation_report(self, output_dir: str, num_episodes: int = 50) -> str:
        """Generate the evaluation report"""
        print(f"Starting full evaluation ({num_episodes} episodes)...")

        # Run the evaluation
        stats = self.evaluate_multiple_episodes(num_episodes=num_episodes)

        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)
        for artifact_name in LEGACY_EVALUATION_ARTIFACTS:
            artifact_path = os.path.join(output_dir, artifact_name)
            if os.path.lexists(artifact_path):
                os.remove(artifact_path)

        # Save raw data
        results_df = pd.DataFrame(stats["raw_results"])
        results_csv = os.path.join(output_dir, "evaluation_results.csv")
        results_df.to_csv(results_csv, index=False)

        protocol_path = os.path.join(output_dir, "evaluation_protocol.json")
        with open(protocol_path, "w", encoding="utf-8") as protocol_file:
            json.dump(
                {
                    "protocol": stats["protocol"],
                    "action_mode": EVALUATION_ACTION_MODE,
                    "render": EVALUATION_RENDER,
                    "lane_change_definition": stats["lane_change_definition"],
                    "scenario_seeds": stats["seed_list"],
                    "checkpoint_signature": stats["checkpoint_signature"],
                    "cognitive_disturbances": {
                        "perception": False,
                        "action_delay": False,
                        "reward_bias": False,
                    },
                    "cognitive_input": stats["neutral_cognitive_params"],
                    "cognitive_mask": stats["neutral_cognitive_mask"],
                },
                protocol_file,
                indent=2,
                ensure_ascii=False,
            )

        # Generate visualizations
        self._create_visualizations(results_df, output_dir)

        # Generate the report
        report_path = os.path.join(output_dir, "evaluation_report.md")
        report_content = f"""# PPO Model Evaluation Report

## Evaluation Overview

- **evaluation episodes**: {stats['num_episodes']}
- **model type**: current public PPO checkpoint
- **evaluation protocol**: `{stats['protocol']}`
- **action mode**: `{EVALUATION_ACTION_MODE}`
- **rendering**: `{EVALUATION_RENDER}`
- **scenario seeds**: `{stats['seed_list']}`
- **cognitive input**: `{stats['neutral_cognitive_params']}` (perception/delay/bias all disabled)
- **evaluation time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Core Metrics

### Reward Metrics
- **average reward**: {stats['reward_mean']:.3f} ± {stats['reward_std']:.3f}
- **maximum reward**: {stats['reward_max']:.3f}
- **minimum reward**: {stats['reward_min']:.3f}

### Episode Length
- **average length**: {stats['length_mean']:.1f} ± {stats['length_std']:.1f} steps

### Safety Metrics
- **collision rate**: {stats['collision_rate']:.3f} ({stats['collision_rate']*100:.1f}%)
- **off-road rate**: {stats['offroad_rate']:.3f} ({stats['offroad_rate']*100:.1f}%)
- **success rate**: {stats['success_rate']:.3f} ({stats['success_rate']*100:.1f}%)
- **timeout rate**: {stats['timeout_rate']:.3f} ({stats['timeout_rate']*100:.1f}%)

### Driving Behavior Metrics
- **average lane changes**: {stats['lane_changes_mean']:.1f}
- **lane-change definition**: `{stats['lane_change_definition']}` (counts only integer lane-ID changes within the same road segment; cross-segment transitions are ignored)

## Performance Analysis

### Reward Distribution
- **median reward**: {results_df['reward'].median():.3f}
- **75th percentile reward**: {results_df['reward'].quantile(0.75):.3f}
- **25th percentile reward**: {results_df['reward'].quantile(0.25):.3f}

## Evaluation Artifacts

- `evaluation_results.csv`: detailed per-episode results
- `evaluation_protocol.json`: evaluation protocol, action mode, checkpoint signature, and scenario seed list
- `reward_distribution.png`: reward distribution plot
- `episode_timeline.png`: episode performance timeline

## Notes

This report provides descriptive statistics for the current checkpoint under a fixed protocol only. It does not assign grades from unverified baselines or thresholds.

---
**evaluation completed at**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**evaluation data**: `{results_csv}`
"""

        # Save the report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"Evaluation report written to: {report_path}")
        print(f"Evaluation data saved to: {results_csv}")

        return report_path

    def _create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create visualization charts"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

        # 1. reward distribution plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(df['reward'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Episode Reward')
        plt.ylabel('Frequency')
        plt.title('Reward Distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 2)
        plt.hist(df['length'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Episode Length')
        plt.ylabel('Frequency')
        plt.title('Episode Length Distribution')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 3, 3)
        termination_counts = [
            df['collision'].sum(),
            df['offroad'].sum(),
            df['success'].sum(),
            df['timeout'].sum()
        ]
        labels = ['Collision', 'Offroad', 'Success', 'Timeout']
        plt.pie(termination_counts, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Termination Reasons')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reward_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Performance timelines
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 3, 1)
        plt.plot(df['episode'], df['reward'], alpha=0.7, marker='o', markersize=3)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward Timeline')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 2)
        plt.plot(df['episode'], df['length'], alpha=0.7, marker='o', markersize=3, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.title('Episode Length Timeline')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 3)
        rolling_reward = df['reward'].rolling(window=10, min_periods=1).mean()
        plt.plot(df['episode'], rolling_reward, linewidth=2, color='red')
        plt.xlabel('Episode')
        plt.ylabel('Rolling Mean Reward')
        plt.title('Reward Trend (10-episode average)')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 4)
        plt.plot(df['episode'], df['lane_changes'], alpha=0.7, marker='o', markersize=3, color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Lane Changes')
        plt.title('Lane Changes per Episode')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 5)
        # Cumulative success rate
        cumulative_success = df['success'].cumsum() / (df.index + 1)
        plt.plot(df['episode'], cumulative_success, linewidth=2, color='purple')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Success Rate')
        plt.title('Success Rate Over Time')
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 3, 6)
        # Safety metrics
        safety_scores = 1 - (df['collision'].astype(int) + df['offroad'].astype(int))
        rolling_safety = pd.Series(safety_scores).rolling(window=10, min_periods=1).mean()
        plt.plot(df['episode'], rolling_safety, linewidth=2, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Safety Score (10-ep avg)')
        plt.title('Safety Trend')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'episode_timeline.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Visualization charts generated")


def main():
    """Main entrypoint"""
    parser = argparse.ArgumentParser(description="PPO checkpoint evaluation")
    parser.add_argument("experiment_dir", type=str,
                       help="Experiment directory path (must contain config.json and checkpoints/)")
    parser.add_argument("--model", type=str, default="best_model.pt",
                       help="Model filename (default: best_model.pt)")
    parser.add_argument("--num_episodes", type=int, default=50,
                       help="Number of evaluation episodes (default: 50)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Compute device")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Evaluation output directory (default: experiment_dir/evaluation)")

    args = parser.parse_args()

    # Validate the experiment directory
    if not os.path.exists(args.experiment_dir):
        print(f"Experiment directory does not exist: {args.experiment_dir}")
        return 1

    # Build paths
    config_path = os.path.join(args.experiment_dir, "config.json")
    model_path = os.path.join(args.experiment_dir, "checkpoints", args.model)

    if not os.path.exists(config_path):
        print(f"Config file does not exist: {config_path}")
        return 1

    if not os.path.exists(model_path):
        print(f"Model file does not exist: {model_path}")
        return 1

    # Set the output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "evaluation")

    print("PPO checkpoint evaluation")
    print("=" * 50)
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Model file: {args.model}")
    print(f"evaluation episodes: {args.num_episodes}")
    print(f"action mode: {EVALUATION_ACTION_MODE}")
    print(f"rendering: {'enabled' if EVALUATION_RENDER else 'disabled'}")
    print("=" * 50)

    try:
        # Create the evaluator
        evaluator = ModelEvaluator(model_path, config_path, args.device)

        # Generate the evaluation report
        report_path = evaluator.generate_evaluation_report(
            args.output_dir, args.num_episodes
        )

        print("\nEvaluation complete!")
        print(f"Evaluation report: {report_path}")
        print(f"Evaluation output: {args.output_dir}")

        return 0

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
