"""Evaluation mixin for PPO expert training."""


from typing import Dict

import numpy as np
import torch

from ppo_train.config.defaults import (
    BIAS_INVERSE_TTA_COEF_DEFAULT,
    DELAY_STEPS_DEFAULT,
    PERCEPTION_SIGMA0_DEFAULT,
    PERCEPTION_SIGMA_MAX_DEFAULT,
)


class EvaluationMixin:
    """Mixin holding methods split from PPOExpertReproduction."""

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the policy with the dedicated evaluation environment."""
        self._reset_evaluation_benchmark()
        eval_envs = self.eval_envs
        eval_rewards = []
        eval_lengths = []
        eval_collisions = 0
        eval_offroads = 0
        eval_successes = 0
        eval_speeds = []
        eval_lane_deviations = []
        eval_lane_changes = []
        eval_path_completions = []

        print(f"Starting policy evaluation ({num_episodes} episodes)...")

        for episode in range(num_episodes):
            obs = eval_envs.reset()
            eval_delay_module = getattr(self, "eval_delay_module", None)
            if eval_delay_module is not None:
                eval_delay_module.reset(env_id=0)
            episode_reward = 0
            episode_length = 0
            episode_speeds = []
            episode_lane_deviations = []
            episode_lane_changes = 0

            # Use the first environment instance for metric computation.
            current_env = self._resolve_eval_env(eval_envs)

            while True:
                current_cognitive_params = self._get_eval_cognitive_params()
                obs_with_cognitive = self._concatenate_cognitive_params(obs, current_cognitive_params)

                obs_tensor = torch.FloatTensor(obs_with_cognitive).to(self.device)

                with torch.no_grad():
                    action = self.network.act_deterministic(obs_tensor)

                action_np = action.cpu().numpy()
                action_np = self._apply_eval_action_delay(action_np)
                obs, reward, done, info = eval_envs.step(action_np)

                # Handle multi-env return values by evaluating the first env only.
                if isinstance(reward, (list, tuple, np.ndarray)):
                    episode_reward += reward[0]
                    done_flag = done[0]
                    info = info[0]
                else:
                    episode_reward += reward
                    done_flag = done

                episode_length += 1

                speed_value = self._extract_eval_speed(info, current_env)

                if speed_value > 0:
                    episode_speeds.append(speed_value)

                missing_metrics = {}
                if current_env is not None:
                    try:
                        missing_metrics = self._calculate_missing_metrics(current_env, info, env_idx=0)
                    except Exception:
                        pass

                # Lane-deviation statistics.
                if isinstance(info, dict) and 'lane_deviation' in info:
                    episode_lane_deviations.append(abs(info['lane_deviation']))
                elif 'lane_deviation' in missing_metrics:
                    episode_lane_deviations.append(abs(missing_metrics['lane_deviation']))
                else:
                    episode_lane_deviations.append(0.0)

                # Lane-change detection.
                if isinstance(info, dict) and 'lane_change' in info and info['lane_change']:
                    episode_lane_changes += 1
                    if self.debug_lane_change:
                        print("[eval] Lane change detected from info")
                elif missing_metrics.get('lane_change', False):
                    episode_lane_changes += 1
                    if self.debug_lane_change:
                        print("[eval] Lane change detected from missing_metrics")
                else:
                    if current_env is not None and hasattr(current_env, 'agent'):
                        if self._detect_lane_change_enhanced(current_env.agent, 0, info):
                            episode_lane_changes += 1
                            if self.debug_lane_change:
                                print("[eval] Lane change detected by enhanced logic")

                if done_flag:
                    if isinstance(info, dict):
                        if (info.get("crash", False) or
                            info.get("crash_vehicle", False) or
                            info.get("crash_object", False) or
                            info.get("collision", False)):
                            eval_collisions += 1
                        elif info.get("out_of_road", False):
                            eval_offroads += 1
                        elif info.get("arrive_dest", False):
                            eval_successes += 1

                    path_completion = 0.0
                    if isinstance(info, dict):
                        path_completion = info.get('route_completion', 0.0)
                        if info.get('arrive_dest', False):
                            path_completion = 1.0
                    eval_path_completions.append(path_completion)
                    break

            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_speeds.append(np.mean(episode_speeds) if episode_speeds else 0)
            eval_lane_deviations.append(np.mean(episode_lane_deviations) if episode_lane_deviations else 0)
            eval_lane_changes.append(episode_lane_changes)

        # Final aggregate statistics.
        collision_rate = eval_collisions / num_episodes
        offroad_rate = eval_offroads / num_episodes
        success_rate = eval_successes / num_episodes

        print(f"Evaluation complete: collision_rate={collision_rate:.3f}, offroad_rate={offroad_rate:.3f}, success_rate={success_rate:.3f}")

        return {
            "eval_reward_mean": np.mean(eval_rewards),
            "eval_reward_std": np.std(eval_rewards),
            "eval_length_mean": np.mean(eval_lengths),
            "eval_collision_rate": collision_rate,
            "eval_offroad_rate": offroad_rate,
            "eval_success_rate": success_rate,
            "eval_avg_speed": np.mean(eval_speeds),
            "eval_lane_deviation": np.mean(eval_lane_deviations),
            "eval_lane_change_count": np.mean(eval_lane_changes),
            "eval_path_completion": np.mean(eval_path_completions)
        }

    def _reset_evaluation_benchmark(self):
        """Restore the same scenarios and cognitive state before every evaluation."""
        if not hasattr(self, "eval_envs") or not hasattr(self.eval_envs, "env_method"):
            raise RuntimeError("fixed evaluation environment is not initialized")
        states = getattr(self, "_eval_initial_env_states", None)
        if not isinstance(states, list) or not states:
            raise RuntimeError("fixed evaluation environment state is not initialized")
        for index, state in enumerate(states):
            self.eval_envs.env_method("set_resume_state", state, indices=index)

    def _apply_eval_action_delay(self, actions):
        delay_module = getattr(self, "eval_delay_module", None)
        if not self.use_cognitive_modules or delay_module is None:
            return actions
        return np.asarray([
            delay_module.process_action(action, is_ppo_mode=True, env_id=index)
            for index, action in enumerate(actions)
        ])

    def _resolve_eval_env(self, eval_envs=None):
        eval_envs = eval_envs if eval_envs is not None else self.eval_envs
        try:
            if hasattr(eval_envs, 'envs') and len(eval_envs.envs) > 0:
                current_env = eval_envs.envs[0]
            elif hasattr(eval_envs, 'venv') and hasattr(eval_envs.venv, 'envs'):
                current_env = eval_envs.venv.envs[0] if len(eval_envs.venv.envs) > 0 else None
            else:
                current_env = None
        except Exception:
            current_env = None
        return current_env

    def _get_eval_cognitive_params(self):
        if not self.use_cognitive_modules:
            return {}
        return {
            'bias_inverse_tta_coef': getattr(
                self.args, 'bias_inverse_tta_coef', BIAS_INVERSE_TTA_COEF_DEFAULT
            ),
            'perception_sigma0': getattr(
                self.args, 'perception_sigma0', PERCEPTION_SIGMA0_DEFAULT
            ),
            'perception_sigma_max': getattr(
                self.args, 'perception_sigma_max', PERCEPTION_SIGMA_MAX_DEFAULT
            ),
            'delay_steps': getattr(self.args, 'delay_steps', DELAY_STEPS_DEFAULT),
        }

    def _extract_eval_speed(self, info, current_env):
        speed_value = 0.0
        if isinstance(info, dict):
            if 'velocity' in info:
                speed_value = abs(info['velocity'])
            elif 'speed' in info:
                speed_value = abs(info['speed'])
            elif hasattr(info, 'speed'):
                speed_value = abs(info.speed)
        # Fall back to the agent object when the info dict does not contain speed.
        if speed_value == 0.0 and current_env is not None:
            try:
                if hasattr(current_env, 'agent') and hasattr(current_env.agent, 'speed'):
                    speed_value = abs(current_env.agent.speed)
            except:
                pass
        return speed_value
