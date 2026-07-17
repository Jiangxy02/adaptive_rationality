"""PPO expert trainer assembled from focused mixins."""


import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from common.random_seed import SeedDomain, derive_seed, seed_global_generators
from common.cognitive_input import (
    BASE_OBSERVATION_DIM,
    COGNITIVE_MASK_DIM,
    COGNITIVE_PARAM_DIM,
    PUBLIC_OBSERVATION_DIM,
)
from cognitive_module.cognitive_bias_module import CognitiveBiasModule
from cognitive_module.cognitive_delay_module import CognitiveDelayModule
from cognitive_module.cognitive_parameter_sampler import CognitiveParameterSampler
from cognitive_module.cognitive_perception_module import CognitivePerceptionModule
from cognitive_module.discrete_cognitive_parameter_sampler import DiscreteCognitiveParameterSampler
from ppo_train.config.build_config import ConfigMixin
from ppo_train.config.defaults import build_runtime_perception_config
from ppo_train.envs.env_factory import EnvironmentMixin
from ppo_train.envs.env_curriculum import EnvCurriculumMixin
from ppo_train.training.checkpointing import CheckpointingMixin
from ppo_train.training.cognitive_integration import CognitiveIntegrationMixin
from ppo_train.training.evaluation import EvaluationMixin
from ppo_train.training.metrics_compute import MetricsComputeMixin
from ppo_train.training.metrics_logging import MetricsLoggingMixin
from ppo_train.training.ppo_update import PPOUpdateMixin
from ppo_train.training.reporting import ReportingMixin
from ppo_train.training.rollout import RolloutMixin
from ppo_train.training.scheduling import SchedulingMixin


class PPOExpertReproduction(ConfigMixin, EnvironmentMixin, EnvCurriculumMixin, SchedulingMixin, RolloutMixin, PPOUpdateMixin, EvaluationMixin, CheckpointingMixin, MetricsComputeMixin, MetricsLoggingMixin, ReportingMixin, CognitiveIntegrationMixin):

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")

        # Seed Python, NumPy, Torch, and CUDA in the main process from one source of truth.
        self.root_seed = seed_global_generators(args.seed)

        # Create the experiment directory.
        self.exp_dir = self._create_experiment_dir()



        # Initialize training statistics before environments are created.
        self.global_step = 0
        self.episode_count = 0
        self.train_stats = []
        self.start_iteration = 0
        self.best_reward = float('-inf')
        self.best_metrics = {}

        # Any runtime state that affects future training or metrics must exist
        # before checkpoint restoration runs.
        self.entropy_coef_start = args.entropy_coef_start
        self.entropy_coef_end = args.entropy_coef_end
        self.entropy_decay_end_ratio = args.entropy_decay_end_ratio
        self.current_entropy_coef = self.entropy_coef_start
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_speeds = deque(maxlen=100)
        self.episode_lane_deviations = deque(maxlen=100)
        self.episode_lane_changes = deque(maxlen=100)
        self.episode_path_completions = deque(maxlen=100)
        self.episode_timeouts = deque(maxlen=100)
        self.episode_steer_means = deque(maxlen=100)
        self.episode_throttle_means = deque(maxlen=100)
        self._last_lane_index = {}
        self.debug_lane_change = False

        # Curriculum-learning state must be initialized before environments are created.
        self.curriculum_mode = self.args.curriculum_mode
        self.curriculum_alpha = self.args.curriculum_alpha
        self.curriculum_stage = 0  # Used by gate mode; progress mode derives stage from progress.

        # Cognitive toggles do not change the paper-defined 283-dim input contract.
        self.use_cognitive_modules = args.use_cognitive_modules
        self.base_obs_dim = BASE_OBSERVATION_DIM  # lidar+state
        self.cognitive_param_dim = COGNITIVE_PARAM_DIM
        self.cognitive_mask_dim = COGNITIVE_MASK_DIM
        self.cognitive_modulation = "concat"
        self.raw_obs_dim = PUBLIC_OBSERVATION_DIM
        self.network_input_dim = self._determine_network_input_dim()
        self.cognitive_bias_module = None
        self.cognitive_delay_module = None
        self.cognitive_perception_module = None
        self.cognitive_parameter_sampler = None

        # Resolve the actual runtime config first; environments and artifacts all
        # consume this single snapshot of truth.
        self.resolved_runtime_config = self._resolve_runtime_config()

        # Save config.
        self.config = self._build_config()
        if not args.resume_from:
            self._save_config()

        resume_obs_dim = None
        if args.resume_from:
            resume_obs_dim = self.get_strict_resume_network_context(args.resume_from)

        self._init_cognitive_modules(args)


        # Cognitive-visualization data collection.
        self.enable_cognitive_visualization = args.cognitive_visualization
        if self.enable_cognitive_visualization and self.use_cognitive_modules:
            self.cognitive_viz_data = {
                'timestamps': [],
                'bias_strength': [],
                'bias_applied': [],
                'delay_steps': [],
                'delay_applied': [],
                'perception_noise': [],
                'perception_applied': [],
                'original_rewards': [],
                'modified_rewards': [],
                'original_actions': [],
                'delayed_actions': [],
                'original_observations': [],
                'noisy_observations': [],
                'step_count': []
            }
            print("Cognitive visualization: enabled")
        else:
            self.cognitive_viz_data = None

        # Create environments.
        self.envs = self._create_environments()
        self.eval_envs = self._create_evaluation_environment()
        self.eval_envs.reset()
        self._eval_initial_env_states = self.eval_envs.env_method("get_resume_state")
        self.eval_delay_module = None
        if self.use_cognitive_modules and args.use_cognitive_delay:
            self.eval_delay_module = CognitiveDelayModule(
                delay_steps=int(args.delay_steps),
                enable_smoothing=args.delay_smoothing,
                smoothing_factor=args.delay_smoothing_factor,
                enable_visualization=False,
            )

        # Progressive-training attributes.
        self.checkpoint_obs_dim = None  # Set during checkpoint load.
        self.is_progressive_training = False  # Whether progressive training is enabled.

        # The progressive warm-start constructor context is also part of strict-resume state.
        self.network = self._create_network(checkpoint_obs_dim=resume_obs_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=args.lr)
        self.is_progressive_training = getattr(self.network, "is_progressive_training", False)

        # Cached learning-rate schedule parameters.
        self.lr_init = self.args.lr              # Initial learning rate.
        self.lr_min = self.args.lr_min           # Lower decay bound.
        self.warmup_ratio = self.args.warmup_ratio
        self.lr_schedule = self.args.lr_schedule

        # Preflight strict resume before writing any logs or config.
        if args.resume_from:
            self.preflight_strict_resume(args.resume_from)
        elif getattr(args, "warm_start_from", None):
            self.load_warm_start(args.warm_start_from)
            print("Warm-start weight migration completed")

        # Validate and record straight-road scenario generation.
        self._validate_and_log_scenarios()

        # Create the TensorBoard writer.
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "tensorboard"))

        # Restore only after all objects are constructed. RNG is restored last inside load_checkpoint.
        if args.resume_from:
            self.load_checkpoint(args.resume_from)
            print("Strict-resume state restoration completed")


        # Create the CSV log.
        self.csv_path = os.path.join(self.exp_dir, "training_logs.csv")
        self._init_csv_log()


        self._log_init_summary()

    def _init_cognitive_modules(self, args):
        if self.use_cognitive_modules:
            # Initialize cognitive perception first because cognitive bias depends on it.
            if args.use_cognitive_perception:
                perception_config = build_runtime_perception_config(args)
                self.cognitive_perception_module = CognitivePerceptionModule(perception_config)

                # Enable radar-beam visualization when requested.
                if getattr(args, 'enable_radar_beam_viz', False):
                    self.cognitive_perception_module.enable_radar_visualization(True)

            # Initialize cognitive bias and pass the perception-module reference.
            if args.use_cognitive_bias:
                bias_config = {
                    'inverse_tta_coef': args.bias_inverse_tta_coef,
                    'tta_threshold': args.bias_tta_threshold,
                    'adaptive_bias': args.bias_adaptive,
                    'adaptation_rate': args.bias_adaptation_rate,
                    'visual_detection_distance': args.bias_visual_distance,
                    'visual_detection_angle': args.bias_visual_angle,
                    'visual_aversion_strength': args.bias_visual_strength
                }
                self.cognitive_bias_module = CognitiveBiasModule(
                    bias_config=bias_config,
                    cognitive_perception_module=self.cognitive_perception_module
                )

            # Initialize cognitive delay.
            if args.use_cognitive_delay:
                self.cognitive_delay_module = CognitiveDelayModule(
                    delay_steps=int(args.delay_steps),
                    enable_smoothing=args.delay_smoothing,
                    smoothing_factor=args.delay_smoothing_factor,
                    enable_visualization=args.cognitive_visualization
                )

            # Initialize the cognitive-parameter sampler.
            if args.use_cognitive_parameter_sampling:
                # Select the sampler implementation from config.
                if args.cognitive_sampler_type == "discrete":
                    self.cognitive_parameter_sampler = DiscreteCognitiveParameterSampler(
                        update_steps=args.cognitive_param_update_steps,
                        bias_inverse_tta_coef_range=args.bias_inverse_tta_coef_range,
                        perception_sigma0_range=args.perception_sigma0_range,
                        perception_sigma_max_range=args.perception_sigma_max_range,
                        delay_steps_range=args.delay_steps_range,
                        bias_inverse_tta_coef_density=args.bias_inverse_tta_coef_density,
                        perception_sigma0_density=args.perception_sigma0_density,
                        perception_sigma_max_density=args.perception_sigma_max_density,
                        delay_steps_density=args.delay_steps_density,
                        enable_visualization=args.cognitive_visualization,
                        save_history=True,
                        seed=derive_seed(self.root_seed, SeedDomain.SAMPLER),
                    )
                elif args.cognitive_sampler_type == "continuous":
                    self.cognitive_parameter_sampler = CognitiveParameterSampler(
                        update_steps=args.cognitive_param_update_steps,
                        bias_inverse_tta_coef_range=args.bias_inverse_tta_coef_range,
                        perception_sigma0_range=args.perception_sigma0_range,
                        perception_sigma_max_range=args.perception_sigma_max_range,
                        delay_steps_range=args.delay_steps_range,
                        enable_visualization=args.cognitive_visualization,
                        save_history=True,
                        seed=derive_seed(self.root_seed, SeedDomain.SAMPLER),
                    )
                else:
                    raise ValueError(f"Unsupported cognitive sampler type: {args.cognitive_sampler_type}")

            else:
                self.cognitive_parameter_sampler = None

    def _log_init_summary(self):
        cognitive_status = "enabled" if self.use_cognitive_modules else "disabled"
        print(
            f"PPO training initialization complete: input_dim={self.network.obs_dim} | "
            f"cognitive modules {cognitive_status}"
        )

        # Progressive-training summary.
        if self.is_progressive_training:
            self.network.training_stage = 1
            print(
                f"Progressive training: {self.checkpoint_obs_dim} -> {self.network.obs_dim} dims | "
                f"unfreeze threshold {self.network.freeze_threshold_steps:,} steps"
            )
        elif self.checkpoint_obs_dim:
            print(f"Resuming training from a {self.checkpoint_obs_dim}-dim checkpoint")

    def _create_experiment_dir(self) -> str:
        # Resume training reuses the original experiment directory.
        if getattr(self.args, 'resume_from', None):
            checkpoint_path = Path(self.args.resume_from)
            # Strict resume reuses the original experiment directory so CSV,
            # TensorBoard, latest, and best keep continuous semantics.
            exp_dir = str(checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent)
            os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
            print(f"Strict-resume mode: reusing experiment directory {exp_dir}")
            return exp_dir

        # Standard mode creates a new experiment directory.
        if self.args.save_dir is None:
            # Set the default save_dir.
            self.args.save_dir = "./runs/ppo_train"

        requested_save_root = self.args.save_dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"ppo_expert_reproduction_{timestamp}"
        exp_dir = os.path.join(requested_save_root, "runs", exp_name)

        try:
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Could not create experiment directory {exp_dir!r} under the requested "
                f"training output root {requested_save_root!r}: {exc}"
            ) from exc

        print(f"Experiment directory created: {exp_dir}")
        return exp_dir

    def train(self):
        print(f"Starting PPO training with target steps: {self.args.total_timesteps:,}")

        # Set the starting iteration index.
        start_iteration = getattr(self, 'start_iteration', 0)
        iteration = start_iteration
        start_time = time.time()
        # If this is a resumed run, report the starting iteration.
        if hasattr(self, 'start_iteration') and self.start_iteration > 0:
            print(f"Resuming training from iteration {self.start_iteration}")

        while self.global_step < self.args.total_timesteps:
            iteration += 1

            # Update the entropy coefficient.
            current_entropy = self._update_entropy_coef()

            # Update curriculum-controlled environment parameters.
            self._update_env_curriculum()

            # Update the progressive-training stage.
            if self.is_progressive_training:
                self.network.update_training_stage(self.global_step)

            # Collect rollouts.
            rollout_start = time.time()
            rollouts = self.collect_rollouts()
            rollout_time = time.time() - rollout_start

            # Update the learning rate for this iteration.
            new_lr = self._compute_scheduled_lr()
            for pg in self.optimizer.param_groups:
                pg["lr"] = new_lr

            # Update the policy.
            update_start = time.time()
            train_stats = self.update_policy(*rollouts)
            update_time = time.time() - update_start

            # Compute FPS.
            fps = (self.args.n_steps * self.args.n_envs) / (rollout_time + update_time)
            train_stats["fps"] = fps
            train_stats["entropy_coef"] = self.current_entropy_coef

            # Periodic evaluation.
            eval_stats = None
            if iteration % self.args.eval_freq == 0:
                eval_stats = self.evaluate()

                # Gate-mode promotion check.
                if self.curriculum_mode == "gate":
                    succ = eval_stats.get("eval_success_rate", 0.0)
                    coll = eval_stats.get("eval_collision_rate", 1.0)
                    if self.curriculum_stage < 3 and succ >= self.args.gate_succ_threshold and coll <= self.args.gate_coll_threshold:
                        self.curriculum_stage += 1
                        print(f"Curriculum promotion -> Stage {self.curriculum_stage}")

                # Threshold-check gate for milestone metrics.
                reward_mean = eval_stats.get("eval_reward_mean", 0.0)
                success_rate = eval_stats.get("eval_success_rate", 0.0)
                collision_rate = eval_stats.get("eval_collision_rate", 1.0)
                offroad_rate = eval_stats.get("eval_offroad_rate", 1.0)

                # Start threshold checks only after curriculum stage 2.
                current_stage = self._current_curriculum_stage()

                # New best checkpoints are persisted atomically without waiting for periodic checkpoint cadence.
                self.maybe_save_best_checkpoint(iteration, eval_stats)

                # Milestone checkpoints also reuse full state and must carry the just-updated best metrics.
                self._maybe_save_milestone_checkpoint(iteration, current_stage, reward_mean, success_rate, collision_rate, offroad_rate)

                # Save the periodic checkpoint.
                if iteration % self.args.checkpoint_freq == 0:
                    self.save_checkpoint(iteration)

            # Log metrics.
            if iteration % self.args.log_freq == 0:
                self.log_metrics(iteration, train_stats, eval_stats)

        print(f"Training complete. Total runtime: {time.time() - start_time:.2f}s")

        # Final evaluation and persistence.
        final_eval = self.evaluate(num_episodes=20)
        self.maybe_save_best_checkpoint(iteration, final_eval)
        self.save_checkpoint(iteration)

        # Generate the final report.
        self.generate_final_report(final_eval)

        # Close resources.
        self.writer.close()

        # Detach cognitive modules and emit their visualizations.
        if self.use_cognitive_modules:
            if self.enable_cognitive_visualization and self.cognitive_perception_module:
                try:
                    viz_dir = os.path.join(self.exp_dir, "cognitive_visualization")
                    os.makedirs(viz_dir, exist_ok=True)
                    # Visualizations must be generated before the environments close.
                    if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                        self.cognitive_perception_module.generate_visualization(save_dir=viz_dir, env=self.envs.envs[0])
                        print("Generated cognitive-perception visualization")
                except Exception as e:
                    print(f"Failed to generate cognitive-perception visualization: {e}")

            if self.cognitive_bias_module:
                try:
                    viz_dir = os.path.join(self.exp_dir, "cognitive_visualization")
                    os.makedirs(viz_dir, exist_ok=True)
                    if hasattr(self.envs, 'envs') and len(self.envs.envs) > 0:
                        self.cognitive_bias_module.generate_visualization(env=self.envs.envs[0], save_dir=viz_dir)
                        print("Generated cognitive-bias visualization")
                except Exception as e:
                    print(f"Failed to generate cognitive-bias visualization: {e}")

            # Generate cognitive-parameter sampler visualizations.
            if self.cognitive_parameter_sampler:
                try:
                    viz_dir = os.path.join(self.exp_dir, "cognitive_visualization")
                    os.makedirs(viz_dir, exist_ok=True)

                    sampler_stats = self.cognitive_parameter_sampler.get_statistics()


                    if len(self.cognitive_parameter_sampler.param_history) < 2:
                        print("Not enough history is available; forcibly recording current parameters...")
                        current_params = self.cognitive_parameter_sampler.get_current_parameters()
                        self.cognitive_parameter_sampler._record_parameter_update(
                            self.global_step, "forced_record"
                        )
                        print(f"Forcibly recorded current parameters: {current_params}")

                    viz_file = self.cognitive_parameter_sampler.generate_parameter_visualization(
                        output_dir=os.path.join(viz_dir, "parameter_sampling"),
                        session_name=f"ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    if viz_file:
                        print("Generated cognitive-parameter sampler visualization")
                    else:
                        print("Failed to generate the cognitive-parameter sampler visualization")

                    history_file = os.path.join(viz_dir, "parameter_sampling_history.json")
                    self.cognitive_parameter_sampler.save_history_to_file(history_file)
                    print("Saved cognitive-parameter sampling history")

                except Exception as e:
                    print(f"Failed to generate cognitive-parameter sampler visualization: {e}")
                    import traceback
                    traceback.print_exc()

            self._detach_cognitive_modules_from_env()

        # Close vectorized environments.
        self.eval_envs.close()
        self.envs.close()
        print("Training environments closed safely")

    def _maybe_save_milestone_checkpoint(self, iteration, current_stage, reward_mean, success_rate, collision_rate, offroad_rate):
        if (current_stage >= 2 and  # Curriculum must reach stage 2 first.
            reward_mean >= 200 and
            success_rate >= 0.70 and
            collision_rate <= 0.15 and
            offroad_rate <= 0.15):

            print("Milestone metrics reached")
            print(f"   Current curriculum stage: Stage {current_stage}")
            print(f"   Average reward: {reward_mean:.3f} >= 200.0")
            print(f"   Success rate: {success_rate:.3f} >= 0.70")
            print(f"   Collision rate: {collision_rate:.3f} <= 0.15")
            print(f"   Off-road rate: {offroad_rate:.3f} <= 0.15")

            # Save a milestone checkpoint with a dedicated name.
            milestone_checkpoint_path = os.path.join(
                self.exp_dir, "checkpoints",
                f"milestone_checkpoint_iter{iteration}_stage{current_stage}_reward{reward_mean:.1f}_succ{success_rate:.2f}.pt"
            )

            milestone_checkpoint = self._build_checkpoint_payload(iteration)
            milestone_checkpoint["milestone_metrics"] = {
                "eval_reward_mean": reward_mean,
                "eval_success_rate": success_rate,
                "eval_collision_rate": collision_rate,
                "eval_offroad_rate": offroad_rate,
                "curriculum_stage": current_stage,
                "milestone_achieved": True,
                "milestone_timestamp": datetime.now().isoformat(),
            }
            self._atomic_torch_save(milestone_checkpoint, milestone_checkpoint_path)
            print(f"Milestone checkpoint saved: {os.path.basename(milestone_checkpoint_path)}")
            print(f"Full path: {milestone_checkpoint_path}")
