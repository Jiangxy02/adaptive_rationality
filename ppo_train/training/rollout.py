"""Rollout collection mixin for PPO expert training."""


from typing import Tuple

import numpy as np
import torch

from ppo_train.config.defaults import (
    BIAS_INVERSE_TTA_COEF_DEFAULT,
    DELAY_STEPS_DEFAULT,
    PERCEPTION_SIGMA0_DEFAULT,
    PERCEPTION_SIGMA_MAX_DEFAULT,
)


class RolloutMixin:
    """Mixin holding methods split from PPOExpertReproduction."""

    def _rollout_start_control_step(self) -> int:
        """Convert the completed transition count to a simulator control-step count."""
        n_envs = int(self.args.n_envs)
        assert n_envs > 0, "n_envs must be positive"
        assert self.global_step % n_envs == 0, (
            "global_step must be divisible by n_envs before collecting a rollout: "
            f"global_step={self.global_step}, n_envs={n_envs}"
        )
        return self.global_step // n_envs

    def collect_rollouts(self) -> Tuple[torch.Tensor, ...]:


        obs_batch = []
        pre_tanh_actions_batch = []
        log_probs_batch = []
        rewards_batch = []
        dones_batch = []
        truncations_batch = []
        timeout_values_batch = []
        values_batch = []


        self._reset_cognitive_state()


        obs = self.envs.reset()


        episode_rewards = np.zeros(self.args.n_envs)
        episode_lengths = np.zeros(self.args.n_envs)
        episode_speeds = [[] for _ in range(self.args.n_envs)]
        lane_deviations = [[] for _ in range(self.args.n_envs)]
        lane_changes = np.zeros(self.args.n_envs)


        episode_steer_means = [[] for _ in range(self.args.n_envs)]
        episode_throttle_means = [[] for _ in range(self.args.n_envs)]

        rollout_start_control_step = self._rollout_start_control_step()


        for step in range(self.args.n_steps):

            if self.use_cognitive_modules and self.cognitive_parameter_sampler:
                current_sim_step = rollout_start_control_step + step
                if self.cognitive_parameter_sampler.should_update_parameters(current_sim_step):
                    new_params = self.cognitive_parameter_sampler.update_parameters(current_sim_step)
                    self._apply_cognitive_parameters(new_params)



            current_cognitive_params = {}
            if self.use_cognitive_modules:
                if self.cognitive_parameter_sampler:
                    current_cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()

                else:

                    current_cognitive_params = {
                        'bias_inverse_tta_coef': self.args.bias_inverse_tta_coef,
                        'perception_sigma0': self.args.perception_sigma0,
                        'perception_sigma_max': self.args.perception_sigma_max,
                        'delay_steps': self.args.delay_steps
                    }


            obs_with_cognitive = self._concatenate_cognitive_params(obs, current_cognitive_params)


            obs_tensor = torch.as_tensor(obs_with_cognitive, dtype=torch.float32, device=self.device)

            with torch.no_grad():
                (
                    actions,
                    pre_tanh_actions,
                    log_probs,
                    _,
                    values,
                ) = self.network.sample_action_and_value(obs_tensor)

                steer_means, throttle_means = self.network.get_action_stats(obs_tensor)



            policy_actions_np = actions.cpu().numpy()
            pre_tanh_actions_np = pre_tanh_actions.cpu().numpy()


            executed_actions_np = self._apply_action_delay(policy_actions_np)

            next_obs, rewards, dones, infos = self.envs.step(executed_actions_np)



            truncations = np.asarray(
                [info.get("TimeLimit.truncated", False) for info in infos],
                dtype=np.float32,
            )
            timeout_values = np.zeros(self.args.n_envs, dtype=np.float32)
            timeout_indices = np.flatnonzero(truncations)
            if timeout_indices.size:
                missing_terminal_obs = [
                    int(env_idx)
                    for env_idx in timeout_indices
                    if "terminal_observation" not in infos[env_idx]
                ]
                if missing_terminal_obs:
                    raise RuntimeError(
                        "TimeLimit truncation is missing terminal_observation: "
                        f"env_indices={missing_terminal_obs}"
                    )

                terminal_obs = np.asarray(
                    [infos[env_idx]["terminal_observation"] for env_idx in timeout_indices],
                    dtype=np.float32,
                )
                terminal_obs_with_cognitive = self._concatenate_cognitive_params(
                    terminal_obs,
                    current_cognitive_params,
                )
                terminal_obs_tensor = torch.as_tensor(
                    terminal_obs_with_cognitive,
                    dtype=torch.float32,
                    device=self.device,
                )
                with torch.no_grad():
                    _, terminal_value_estimates = self.network(terminal_obs_tensor)
                timeout_values[timeout_indices] = (
                    terminal_value_estimates.squeeze(-1).cpu().numpy()
                )


            if self.use_cognitive_modules and self.cognitive_bias_module:
                for env_idx in range(self.args.n_envs):
                    reward_result = self.envs.env_method(
                        "process_cognitive_bias_reward",
                        float(rewards[env_idx]),
                        infos[env_idx],
                        indices=env_idx,
                    )[0]
                    adjusted_reward, bias_info = reward_result
                    rewards[env_idx] = float(adjusted_reward)

                    if (
                        bias_info.get('bias_active', False)
                        and env_idx == 0
                        and step % 100 == 0
                    ):
                        print(
                            f"Cognitive bias: {bias_info['original_reward']:.3f} -> "
                            f"{rewards[env_idx]:.3f} "
                            f"(bias: {bias_info['bias_applied']:+.3f}, "
                            f"TTA⁻¹: {bias_info['inverse_tta']:.3f})"
                        )


            episode_rewards += rewards
            episode_lengths += 1


            for env_idx, info in enumerate(infos):

                if 'velocity' in info:
                    episode_speeds[env_idx].append(info['velocity'])
                elif 'speed' in info:
                    episode_speeds[env_idx].append(info['speed'])
                elif hasattr(info, 'speed'):
                    episode_speeds[env_idx].append(info.speed)


                try:

                    current_env = None
                    if hasattr(self.envs, 'envs') and len(self.envs.envs) > env_idx:
                        current_env = self.envs.envs[env_idx]
                    elif hasattr(self.envs, 'venv') and hasattr(self.envs.venv, 'envs'):
                        current_env = self.envs.venv.envs[env_idx] if len(self.envs.venv.envs) > env_idx else None


                    if current_env is not None:
                        missing_metrics = self._calculate_missing_metrics(current_env, info, env_idx=0)
                    else:
                        missing_metrics = {}
                except Exception:
                    missing_metrics = {}


                if 'lane_deviation' in info:
                    lane_deviations[env_idx].append(info['lane_deviation'])
                elif 'lane_deviation' in missing_metrics:
                    lane_deviations[env_idx].append(missing_metrics['lane_deviation'])


                lane_change_detected = False
                if 'lane_change' in info and info['lane_change']:
                    lane_changes[env_idx] += 1
                    lane_change_detected = True
                    if self.debug_lane_change:
                        print(f"[env {env_idx}] Lane change detected from info")
                elif missing_metrics.get('lane_change', False):
                    lane_changes[env_idx] += 1
                    lane_change_detected = True
                    if self.debug_lane_change:
                        print(f"[env {env_idx}] Lane change detected from missing_metrics")
                else:

                    if current_env is not None and hasattr(current_env, 'agent'):
                        lane_change_detected = self._detect_lane_change_enhanced(current_env.agent, env_idx, info)
                        if lane_change_detected:
                            lane_changes[env_idx] += 1
                            if self.debug_lane_change:
                                print(f"[env {env_idx}] Lane change detected by enhanced logic")



                episode_steer_means[env_idx].append(steer_means[env_idx].item())
                episode_throttle_means[env_idx].append(throttle_means[env_idx].item())


            for env_idx in range(self.args.n_envs):
                if dones[env_idx]:
                    if self.use_cognitive_modules and self.cognitive_delay_module:
                        self.cognitive_delay_module.reset(env_id=env_idx)


                    self.episode_rewards.append(episode_rewards[env_idx])
                    self.episode_lengths.append(episode_lengths[env_idx])
                    self.episode_speeds.append(np.mean(episode_speeds[env_idx]) if episode_speeds[env_idx] else 0)
                    self.episode_lane_deviations.append(np.mean(lane_deviations[env_idx]) if lane_deviations[env_idx] else 0)
                    self.episode_lane_changes.append(lane_changes[env_idx])


                    self.episode_steer_means.append(np.mean(episode_steer_means[env_idx]) if episode_steer_means[env_idx] else 0)
                    self.episode_throttle_means.append(np.mean(episode_throttle_means[env_idx]) if episode_throttle_means[env_idx] else 0)


                    info = infos[env_idx]
                    path_completion = info.get('route_completion', 0.0)
                    if 'arrive_dest' in info and info['arrive_dest']:
                        path_completion = 1.0
                    self.episode_path_completions.append(path_completion)


                    timeout = (episode_lengths[env_idx] >= self.envs.get_attr('config')[env_idx]['horizon'] and
                              not info.get('arrive_dest', False) and not info.get('crash', False))
                    self.episode_timeouts.append(1 if timeout else 0)


                    episode_rewards[env_idx] = 0
                    episode_lengths[env_idx] = 0
                    episode_speeds[env_idx] = []
                    lane_deviations[env_idx] = []
                    lane_changes[env_idx] = 0

                    episode_steer_means[env_idx] = []
                    episode_throttle_means[env_idx] = []


            obs_batch.append(obs_with_cognitive.copy())
            pre_tanh_actions_batch.append(pre_tanh_actions_np)
            log_probs_batch.append(log_probs.cpu().numpy())
            rewards_batch.append(rewards)
            dones_batch.append(dones.astype(np.float32))
            truncations_batch.append(truncations)
            timeout_values_batch.append(timeout_values)
            values_batch.append(values.cpu().numpy())

            obs = next_obs



        current_cognitive_params = {}
        if self.use_cognitive_modules and self.cognitive_parameter_sampler:
            try:
                current_cognitive_params = self.cognitive_parameter_sampler.get_current_parameters()
            except Exception as e:
                print(f"Failed to read cognitive parameters: {e}")

                current_cognitive_params = {
                    'bias_inverse_tta_coef': BIAS_INVERSE_TTA_COEF_DEFAULT,
                    'perception_sigma0': PERCEPTION_SIGMA0_DEFAULT,
                    'perception_sigma_max': PERCEPTION_SIGMA_MAX_DEFAULT,
                    'delay_steps': DELAY_STEPS_DEFAULT
                }
        elif self.use_cognitive_modules:

            current_cognitive_params = {
                'bias_inverse_tta_coef': BIAS_INVERSE_TTA_COEF_DEFAULT,
                'perception_sigma0': PERCEPTION_SIGMA0_DEFAULT,
                'perception_sigma_max': PERCEPTION_SIGMA_MAX_DEFAULT,
                'delay_steps': DELAY_STEPS_DEFAULT
            }

        last_obs_with_cognitive = self._concatenate_cognitive_params(obs, current_cognitive_params)
        last_obs_tensor = torch.as_tensor(last_obs_with_cognitive, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, last_values = self.network(last_obs_tensor)
            last_values = last_values.squeeze(-1)


        self.global_step += self.args.n_steps * self.args.n_envs


        obs_batch     = torch.as_tensor(np.asarray(obs_batch,     dtype=np.float32), device=self.device)
        pre_tanh_actions_batch = torch.as_tensor(
            np.asarray(pre_tanh_actions_batch, dtype=np.float32),
            device=self.device,
        )
        log_probs_batch = torch.as_tensor(np.asarray(log_probs_batch, dtype=np.float32), device=self.device)
        rewards_batch = torch.as_tensor(np.asarray(rewards_batch, dtype=np.float32), device=self.device)
        dones_batch   = torch.as_tensor(np.asarray(dones_batch,   dtype=np.float32), device=self.device)
        truncations_batch = torch.as_tensor(
            np.asarray(truncations_batch, dtype=np.float32), device=self.device
        )
        timeout_values_batch = torch.as_tensor(
            np.asarray(timeout_values_batch, dtype=np.float32), device=self.device
        )
        values_batch  = torch.as_tensor(np.asarray(values_batch,  dtype=np.float32), device=self.device)



        advantages, returns = self.compute_gae(
            rewards_batch,
            values_batch,
            dones_batch,
            truncations_batch,
            timeout_values_batch,
            last_values,
        )

        return (obs_batch, pre_tanh_actions_batch, log_probs_batch,
                advantages, returns)

    def _reset_cognitive_state(self):
        if self.use_cognitive_modules:
            if self.cognitive_delay_module:
                self.cognitive_delay_module.reset()
            if self.cognitive_perception_module:
                self.cognitive_perception_module.reset()
                if hasattr(self.envs, "env_method"):
                    self.envs.env_method("reset_cognitive_perception")
            if self.cognitive_bias_module:
                self.cognitive_bias_module.reset()
                if hasattr(self.envs, "env_method"):
                    self.envs.env_method("reset_cognitive_bias")

    def _apply_action_delay(self, actions_np):
        if self.use_cognitive_modules and self.cognitive_delay_module:

            actions_delayed = []
            for env_idx in range(self.args.n_envs):
                single_action = actions_np[env_idx]

                delayed_action = self.cognitive_delay_module.process_action(
                    single_action,
                    is_ppo_mode=True,
                    env_id=env_idx,
                )
                actions_delayed.append(delayed_action)
            actions_np = np.array(actions_delayed)
        return actions_np
