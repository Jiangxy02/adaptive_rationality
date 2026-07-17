"""Metrics and logging mixin for PPO expert training."""


import csv
import os
from typing import Dict

import numpy as np

from ppo_train.config.defaults import PERCEPTION_SIGMA_MAX_DEFAULT


class MetricsLoggingMixin:
    """Metric logging helpers for CSV and console output."""

    def _init_csv_log(self):
        """Initialize the CSV log file."""
        headers = [
            "step", "episode", "ep_reward_mean", "ep_len_mean",
            "policy_loss", "value_loss", "entropy", "approx_kl",
            "max_candidate_kl", "max_committed_kl", "mean_step_scale",
            "kl_backtrack_count", "kl_rejected_steps", "kl_precheck_stops",
            "learning_rate", "entropy_coef", "collision_rate", "offroad_rate",
            "success_rate", "fps", "clipfrac", "explained_variance",
            "grad_norm", "avg_speed", "lane_deviation", "lane_change_count",
            "path_completion",
            "steer_mean", "steer_std", "throttle_mean", "throttle_std",
            "bias_inverse_tta_coef", "perception_sigma0", "perception_sigma_max", "delay_steps"
        ]

        # Resume mode must not silently append to an old schema; otherwise rows shift.
        if hasattr(self.args, 'resume_from') and self.args.resume_from and os.path.exists(self.csv_path):
            with open(self.csv_path, newline='', encoding='utf-8') as f:
                existing_headers = next(csv.reader(f), None)
            if existing_headers != headers:
                raise ValueError(
                    "Resumed CSV header does not match the current metric schema. "
                    "Migrate the old log or use a new save_dir."
                )
            print(f"Resuming CSV logging: {self.csv_path}")
            return

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        print(f"Created CSV log file: {self.csv_path}")

    def log_metrics(self, iteration: int, train_stats: Dict, eval_stats: Dict = None):
        """Log training and evaluation metrics."""
        # TensorBoard logging for training metrics.
        for key, value in train_stats.items():
            if key == "policy_loss":
                self.writer.add_scalar("loss/actor_loss", value, self.global_step)
                self.writer.add_scalar("train/policy_loss", value, self.global_step)
            elif key == "entropy_loss":
                self.writer.add_scalar("loss/entropy_loss", value, self.global_step)
            else:
                self.writer.add_scalar(f"train/{key}", value, self.global_step)

        # Record the learning rate.
        current_lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar("train/learning_rate", current_lr, self.global_step)

        # Record the entropy coefficient.
        self.writer.add_scalar("train/entropy_coef", self.current_entropy_coef, self.global_step)

        # Progressive-training monitoring.
        if self.is_progressive_training:
            # Record the training stage.
            self.writer.add_scalar("progressive_training/stage", self.network.training_stage, self.global_step)

            # Record frozen-weight stats.
            frozen_stats = self.network.get_frozen_weight_stats()
            if frozen_stats:
                self.writer.add_scalar("progressive_training/actor_frozen_norm", frozen_stats['actor_frozen_norm'], self.global_step)
                self.writer.add_scalar("progressive_training/critic_frozen_norm", frozen_stats['critic_frozen_norm'], self.global_step)
                self.writer.add_scalar("progressive_training/actor_new_norm", frozen_stats['actor_new_norm'], self.global_step)
                self.writer.add_scalar("progressive_training/critic_new_norm", frozen_stats['critic_new_norm'], self.global_step)

        # Record the curriculum stage for visualization.
        stage_for_log = self._current_curriculum_stage()
        self.writer.add_scalar("env/curriculum_stage", stage_for_log, self.global_step)

        # Record the current traffic density from the curriculum.
        current_traffic_density = self._curriculum_density()
        self.writer.add_scalar("env/curriculum_traffic_density", current_traffic_density, self.global_step)

        if self.args.use_speed_control_reward:
            speed_control_data = self._collect_speed_control_metrics()

            if speed_control_data:
                self.writer.add_scalar("speed_control/total_reward", speed_control_data.get('r_total', 0), self.global_step)
                self.writer.add_scalar("speed_control/speed_deviation", speed_control_data.get('speed_deviation', 0), self.global_step)
                self.writer.add_scalar("speed_control/overspeed_flag", speed_control_data.get('overspeed_flag', 0), self.global_step)
                self.writer.add_scalar("speed_control/target_speed", speed_control_data.get('target_speed', 0), self.global_step)

        # Episode-level environment statistics.
        if len(self.episode_rewards) > 0:
            self.writer.add_scalar("env/ep_rew_mean", np.mean(self.episode_rewards), self.global_step)
            self.writer.add_scalar("env/ep_rew_max", np.max(self.episode_rewards), self.global_step)
            self.writer.add_scalar("env/ep_rew_min", np.min(self.episode_rewards), self.global_step)

        if len(self.episode_lengths) > 0:
            self.writer.add_scalar("env/ep_len_mean", np.mean(self.episode_lengths), self.global_step)
            self.writer.add_scalar("env/ep_len_max", np.max(self.episode_lengths), self.global_step)
            self.writer.add_scalar("env/ep_len_min", np.min(self.episode_lengths), self.global_step)

        if len(self.episode_timeouts) > 0:
            self.writer.add_scalar("env/time_outs", np.mean(self.episode_timeouts), self.global_step)

        # Action statistics.
        if len(self.episode_steer_means) > 0:
            self.writer.add_scalar("actions/steer_mean", np.mean(self.episode_steer_means), self.global_step)
            self.writer.add_scalar("actions/steer_std", np.std(self.episode_steer_means), self.global_step)
            self.writer.add_scalar("actions/steer_min", np.min(self.episode_steer_means), self.global_step)
            self.writer.add_scalar("actions/steer_max", np.max(self.episode_steer_means), self.global_step)

        if len(self.episode_throttle_means) > 0:
            self.writer.add_scalar("actions/throttle_mean", np.mean(self.episode_throttle_means), self.global_step)
            self.writer.add_scalar("actions/throttle_std", np.std(self.episode_throttle_means), self.global_step)
            self.writer.add_scalar("actions/throttle_min", np.min(self.episode_throttle_means), self.global_step)
            self.writer.add_scalar("actions/throttle_max", np.max(self.episode_throttle_means), self.global_step)

        # Evaluation metrics.
        if eval_stats:
            for key, value in eval_stats.items():
                self.writer.add_scalar(f"eval/{key.replace('eval_', '')}", value, self.global_step)

        if self.cognitive_parameter_sampler:
            current_params = self.cognitive_parameter_sampler.get_current_parameters()
            self.writer.add_scalar("cognitive_params/bias_inverse_tta_coef",
                                current_params['bias_inverse_tta_coef'],
                                self.global_step)
            self.writer.add_scalar("cognitive_params/perception_sigma0",
                                current_params['perception_sigma0'],
                                self.global_step)
            self.writer.add_scalar("cognitive_params/perception_sigma_max",
                                current_params.get('perception_sigma_max', PERCEPTION_SIGMA_MAX_DEFAULT),
                                self.global_step)
            self.writer.add_scalar("cognitive_params/delay_steps",
                                current_params['delay_steps'],
                                self.global_step)

        if self.use_cognitive_modules:
            self.writer.add_scalar("cognitive_modules/bias_module",
                                1.0 if self.args.use_cognitive_bias else 0.0,
                                self.global_step)
            self.writer.add_scalar("cognitive_modules/perception_module",
                                1.0 if self.args.use_cognitive_perception else 0.0,
                                self.global_step)
            self.writer.add_scalar("cognitive_modules/delay_module",
                                1.0 if self.args.use_cognitive_delay else 0.0,
                                self.global_step)

            active_modules = sum([
                self.args.use_cognitive_bias,
                self.args.use_cognitive_perception,
                self.args.use_cognitive_delay
            ])
            self.writer.add_scalar("cognitive_modules/active_modules_count",
                                active_modules,
                                self.global_step)

        # CSV log output.
        cognitive_params = self._resolve_csv_cognitive_params()



        log_data = [
            self.global_step, iteration,
            eval_stats.get("eval_reward_mean", 0) if eval_stats else 0,
            eval_stats.get("eval_length_mean", 0) if eval_stats else 0,
            train_stats.get("policy_loss", 0),
            train_stats.get("value_loss", 0),
            train_stats.get("entropy", 0),
            train_stats.get("approx_kl", 0),
            train_stats.get("max_candidate_kl", 0),
            train_stats.get("max_committed_kl", 0),
            train_stats.get("mean_step_scale", 0),
            train_stats.get("kl_backtrack_count", 0),
            train_stats.get("kl_rejected_steps", 0),
            train_stats.get("kl_precheck_stops", 0),
            current_lr,
            self.current_entropy_coef,
            eval_stats.get("eval_collision_rate", 0) if eval_stats else 0,
            eval_stats.get("eval_offroad_rate", 0) if eval_stats else 0,
            eval_stats.get("eval_success_rate", 0) if eval_stats else 0,
            train_stats.get("fps", 0),
            train_stats.get('clipfrac', 0),
            train_stats.get('explained_variance', 0),
            train_stats.get('grad_norm', 0),
            eval_stats.get('eval_avg_speed', 0) if eval_stats else 0,
            eval_stats.get('eval_lane_deviation', 0) if eval_stats else 0,
            eval_stats.get('eval_lane_change_count', 0) if eval_stats else 0,
            eval_stats.get('eval_path_completion', 0) if eval_stats else 0,
            np.mean(self.episode_steer_means) if len(self.episode_steer_means) > 0 else 0,
            np.std(self.episode_steer_means) if len(self.episode_steer_means) > 0 else 0,
            np.mean(self.episode_throttle_means) if len(self.episode_throttle_means) > 0 else 0,
            np.std(self.episode_throttle_means) if len(self.episode_throttle_means) > 0 else 0,
            cognitive_params['bias_inverse_tta_coef'],
            cognitive_params['perception_sigma0'],
            cognitive_params.get('perception_sigma_max', PERCEPTION_SIGMA_MAX_DEFAULT),
            cognitive_params['delay_steps']
        ]

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(log_data)

        self._print_console_metrics(iteration, train_stats, eval_stats)

    def _resolve_csv_cognitive_params(self):
        cognitive_params = {
            'bias_inverse_tta_coef': 0.0,
            'perception_sigma0': 0.0,
            'perception_sigma_max': PERCEPTION_SIGMA_MAX_DEFAULT,
            'delay_steps': 1
        }

        if self.cognitive_parameter_sampler:
            try:
                cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()
            except Exception as e:
                print(f"Failed to read cognitive parameters from the sampler: {e}")
        elif self.use_cognitive_modules:
            if self.cognitive_bias_module:
                cognitive_params['bias_inverse_tta_coef'] = getattr(self.cognitive_bias_module, 'inverse_tta_coef', self.args.bias_inverse_tta_coef)

            if self.cognitive_perception_module:
                cognitive_params['perception_sigma0'] = getattr(
                    self.cognitive_perception_module,
                    'sigma0',
                    self.args.perception_sigma0,
                )
                default_sigma_max = getattr(self.args, 'perception_sigma_max', PERCEPTION_SIGMA_MAX_DEFAULT)
                cognitive_params['perception_sigma_max'] = getattr(self.cognitive_perception_module, 'sigma_max', default_sigma_max)

            if self.cognitive_delay_module:
                cognitive_params['delay_steps'] = getattr(self.cognitive_delay_module, 'delay_steps', self.args.delay_steps)

            print(f"Resolved the actually applied cognitive parameters: {cognitive_params}")
        return cognitive_params

    def _print_console_metrics(self, iteration, train_stats, eval_stats):
        # Console output.
        reward_str = f"{eval_stats.get('eval_reward_mean', 0):.3f}" if eval_stats else "N/A"
        print(f"Step {self.global_step:8d} | Iter {iteration:4d} | "
              f"Reward: {reward_str} | "
              f"Policy Loss: {train_stats.get('policy_loss', 0):.6f} | "
              f"Value Loss: {train_stats.get('value_loss', 0):.6f}")
        if train_stats.get("kl_backtrack_count", 0):
            print(
                "   KL transactional backtrack: "
                f"candidate={train_stats.get('max_candidate_kl', 0):.5f} | "
                f"committed={train_stats.get('max_committed_kl', 0):.5f} | "
                f"mean_scale={train_stats.get('mean_step_scale', 0):.4f} | "
                f"backtracks={train_stats.get('kl_backtrack_count', 0)}"
            )
        if train_stats.get("kl_rejected_steps", 0):
            print(
                "   KL transactional rejection: "
                f"steps={train_stats.get('kl_rejected_steps', 0)}"
            )

        # Emit a detailed summary at a lower frequency.
        if iteration % (self.args.eval_freq * 2) == 0 and eval_stats:
            print(f"Detailed statistics (step {self.global_step}):")
            print(f"   Average speed: {eval_stats.get('eval_avg_speed', 0):.2f}")
            print(f"   Lane deviation: {eval_stats.get('eval_lane_deviation', 0):.3f}")
            print(f"   Path completion: {eval_stats.get('eval_path_completion', 0):.3f}")
            print(f"   Success rate: {eval_stats.get('eval_success_rate', 0):.3f}")
            print(f"   Current entropy coefficient: {self.current_entropy_coef:.4f}")

            if self.is_progressive_training:
                print("Progressive-training status:")
                print(f"   Training stage: {self.network.training_stage} (1=frozen, 2=unfrozen)")
                print(f"   Freeze threshold: {self.network.freeze_threshold_steps:,} steps")
                print(f"   Steps until unfreeze: {max(0, self.network.freeze_threshold_steps - self.global_step):,}")

                frozen_stats = self.network.get_frozen_weight_stats()
                if frozen_stats:
                    print("   Frozen-weight statistics:")
                    print(f"     Actor first {self.base_obs_dim} dims: mean={frozen_stats['actor_frozen_mean']:.4f}, norm={frozen_stats['actor_frozen_norm']:.4f}")
                    print(f"     Critic first {self.base_obs_dim} dims: mean={frozen_stats['critic_frozen_mean']:.4f}, norm={frozen_stats['critic_frozen_norm']:.4f}")
                    print(f"     Added 4 dims: Actor={frozen_stats['actor_new_norm']:.4f}, Critic={frozen_stats['critic_new_norm']:.4f}")

            if len(self.episode_steer_means) > 0:
                print(f"   Steering mean: {np.mean(self.episode_steer_means):.3f} +- {np.std(self.episode_steer_means):.3f}")
            if len(self.episode_throttle_means) > 0:
                print(f"   Throttle mean: {np.mean(self.episode_throttle_means):.3f} +- {np.std(self.episode_throttle_means):.3f}")

            if train_stats.get('clipfrac', 0) > 0:
                print(f"   Clip Fraction: {train_stats.get('clipfrac', 0):.3f}")
                print(f"   Explained Var: {train_stats.get('explained_variance', 0):.3f}")

            if self.args.use_speed_control_reward:
                speed_control_data = self._collect_speed_control_metrics()
                if speed_control_data:
                    print("   Speed-control reward:")
                    print(f"     Total reward: {speed_control_data.get('r_total', 0):.3f}")
                    print(f"     Speed deviation: {speed_control_data.get('speed_deviation', 0):.2f} m/s")
                    print(f"     Overspeed flag: {'yes' if speed_control_data.get('overspeed_flag', 0) > 0 else 'no'}")
                    print(f"     Target speed: {speed_control_data.get('target_speed', 0):.2f} m/s")

                    tracking_status = "enabled" if speed_control_data.get('speed_control_enable_tracking', True) else "disabled"
                    soft_wall_status = "enabled" if speed_control_data.get('speed_control_enable_soft_wall', True) else "disabled"
                    behavior_status = "enabled" if speed_control_data.get('speed_control_enable_behavior_guidance', True) else "disabled"
                    print(f"     Submodule status: tracking={tracking_status} soft_wall={soft_wall_status} behavior={behavior_status}")

                    if speed_control_data.get('speed_control_enable_tracking', True):
                        print(f"     Tracking reward: {speed_control_data.get('r_track', 0):.3f}")
                    if speed_control_data.get('speed_control_enable_soft_wall', True):
                        print(f"     Soft-wall penalty: {speed_control_data.get('r_wall', 0):.3f}")
                    if speed_control_data.get('speed_control_enable_behavior_guidance', True):
                        print(f"     Behavior guidance: {speed_control_data.get('r_act_over', 0):.3f}")
