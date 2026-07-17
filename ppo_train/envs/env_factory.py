"""Environment construction mixin for PPO expert training."""


import copy

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from ppo_train.config.runtime_config import resolve_runtime_config, resolved_env_config

from .speed_control_env import make_env


class EnvironmentMixin:
    """Mixin holding methods split from PPOExpertReproduction."""

    def _resolve_runtime_config(self):
        """Resolve environment and artifact facts exactly once."""
        return resolve_runtime_config(
            self.args,
            initial_traffic_density=self._curriculum_density(),
        )

    def _get_base_env_config(self):
        """Return the first pre-resolved environment for compatibility."""
        return copy.deepcopy(
            resolved_env_config(self.resolved_runtime_config, 0)["metadrive_config"]
        )

    def _update_env_curriculum(self):
        """Dynamically update curriculum-learning parameters in the environments"""
        # Compute the current traffic density
        current_traffic_density = self._curriculum_density()

        # Update the traffic density in all environments
        # SB3 VecEnv variants expose env_method, including SubprocVecEnv.
        if hasattr(self.envs, "env_method"):
            self.envs.env_method("update_traffic_density", current_traffic_density)
        elif hasattr(self.envs, "envs"):
            # DummyVecEnv case: update each underlying environment instance directly.
            for env in self.envs.envs:
                if hasattr(env, "update_traffic_density"):
                    env.update_traffic_density(current_traffic_density)
                elif hasattr(env, "config"):
                    env.config["traffic_density"] = current_traffic_density

    def _get_env_config(self):
        """Get the environment config for compatibility; internally use _get_base_env_config"""
        return self._get_base_env_config()

    def _create_environments(self):
        """Create vectorized environments with real multi-process parallelism"""
        if self.args.n_envs > 1:
            print(f"Training environments: {self.args.n_envs} parallel environments (SubprocVecEnv)")
            # Use SubprocVecEnv to create multi-process parallel environments.
            return SubprocVecEnv([
                make_env(
                    rank,
                    resolved_env_config(self.resolved_runtime_config, rank),
                    self.args,
                )
                for rank in range(self.args.n_envs)
            ])
        else:
            print("Training environments: 1 environment (DummyVecEnv)")
            # Keep the vectorized interface even for a single environment.
            return DummyVecEnv([
                make_env(
                    0,
                    resolved_env_config(self.resolved_runtime_config, 0),
                    self.args,
                )
            ])

    def _create_evaluation_environment(self):
        """Create one isolated environment for a repeatable evaluation benchmark."""
        eval_args = copy.copy(self.args)
        eval_args.n_envs = 1
        eval_config = copy.deepcopy(
            resolved_env_config(self.resolved_runtime_config, 0)
        )
        eval_config["metadrive_config"]["traffic_density"] = float(
            self.resolved_runtime_config["environment"]["initial_traffic_density"]
        )
        # MetaDrive owns a process-global engine.  Keep evaluation in its own
        # worker even when training itself uses a single DummyVecEnv.
        return SubprocVecEnv([make_env(0, eval_config, eval_args)])
