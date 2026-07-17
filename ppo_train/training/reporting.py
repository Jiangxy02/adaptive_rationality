"""Generate training artifacts from the resolved runtime configuration."""

import json
import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
import torch


class ReportingMixin:
    """Write the scenario manifest and final report from one runtime snapshot."""

    def generate_final_report(self, final_eval: Dict):
        """Generate a report without reconstructing configuration facts."""
        report_path = os.path.join(self.exp_dir, "report.md")
        df = pd.read_csv(self.csv_path)

        runtime = self.resolved_runtime_config
        scenario = runtime["scenario"]
        environment = runtime["environment"]
        ppo = runtime["ppo"]
        training = runtime["training"]
        curriculum = runtime["curriculum"]
        network = self.config["network"]

        environment_rows = "\n".join(
            "- Environment {rank}: seed={seed}, map=`{map}`, {segments} segments".format(
                rank=resolved["rank"],
                seed=resolved["scenario"]["seed"],
                map=resolved["scenario"]["map"],
                segments=resolved["scenario"]["num_segments"],
            )
            for resolved in runtime["environments"]
        )

        report_content = f"""# PPO Training Report

## Experiment Configuration

This report, `config.json`, `scenario_config.json`, and the actual environments all come from resolved runtime config schema {runtime['schema_version']}.

### Network Structure

- **Observation dimension**: {network['observation_dim']}
- **Action dimension**: {network['action_dim']}
- **Hidden dimension**: {network['hidden_dim']}
- **Activation function**: {network['activation']}

### Environment and Scenarios

- **Scenario type**: {scenario['scenario_type']}
- **Scenario count**: {scenario['total_scenarios']}
- **Straight-segment count range**: {scenario['segments_range'][0]}-{scenario['segments_range'][1]}
- **Per-segment length range**: {scenario['segment_length_range_meters'][0]}-{scenario['segment_length_range_meters'][1]} meters
- **Road length range**: {scenario['road_length_range_meters'][0]}-{scenario['road_length_range_meters'][1]} meters
- **Initial traffic density**: {environment['initial_traffic_density']}
- **Traffic-density curriculum**: {curriculum['mode']}
- **Traffic randomization**: {environment['random_traffic']}
- **episode horizon**: {environment['horizon']}
- **Parallel environments**: {ppo['n_envs']}

{environment_rows}

### PPO and Training Controls

- **Learning rate**: {ppo['learning_rate']}
- **Rollout steps**: {ppo['n_steps']}
- **Environment count**: {ppo['n_envs']}
- **Batch size**: {ppo['batch_size']}
- **Training epochs**: {ppo['n_epochs']}
- **Discount factor**: {ppo['gamma']}
- **GAE Lambda**: {ppo['gae_lambda']}
- **Clip range**: {ppo['clip_range']}
- **Entropy coefficient**: {ppo['entropy_coef']}
- **Total training steps**: {training['total_timesteps']}
- **Checkpoint frequency**: {training['checkpoint_freq']}
- **Evaluation frequency**: {training['eval_freq']}

## Training Results

- **Average reward**: {final_eval['eval_reward_mean']:.3f} ± {final_eval['eval_reward_std']:.3f}
- **Average episode length**: {final_eval['eval_length_mean']:.1f}
- **Collision rate**: {final_eval['eval_collision_rate']:.3f}
- **Off-road rate**: {final_eval['eval_offroad_rate']:.3f}
- **Success rate**: {final_eval['eval_success_rate']:.3f}
- **Executed environment steps**: {self.global_step:,}
- **Final policy loss**: {df['policy_loss'].iloc[-1]:.6f}
- **Final value loss**: {df['value_loss'].iloc[-1]:.6f}
- **Final entropy**: {df['entropy'].iloc[-1]:.6f}

## Runtime Parameters

```bash
python ppo_train/scripts/train.py \\
    --n_steps {ppo['n_steps']} \\
    --n_envs {ppo['n_envs']} \\
    --batch_size {ppo['batch_size']} \\
    --n_epochs {ppo['n_epochs']} \\
    --total_timesteps {training['total_timesteps']}
```

## Reproducibility Information

- Python: {sys.version.split()[0]}
- PyTorch: {torch.__version__}
- NumPy: {np.__version__}
- Random seed: {self.config['random_seed']}
- Compute device: {self.device}
- Report generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Experiment directory: `{self.exp_dir}`
"""

        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(report_content)
        print(f"Final report generated: {report_path}")

    def _validate_and_log_scenarios(self):
        """Persist the already-resolved runtime snapshot."""
        runtime = self.resolved_runtime_config

        scenario_config_path = os.path.join(self.exp_dir, "scenario_config.json")
        with open(scenario_config_path, "w", encoding="utf-8") as manifest_file:
            json.dump(runtime, manifest_file, indent=2, ensure_ascii=False)

        print(f"Scenario configuration saved: {scenario_config_path}")
