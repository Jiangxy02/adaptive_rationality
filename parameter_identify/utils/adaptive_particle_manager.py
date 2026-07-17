#!/usr/bin/env python3


import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass
import copy
from parameter_identify.utils.particle_manager import (
    DEFAULT_EVOLUTION_NOISE_RATIO,
    Particle,
    ParticleManager,
    _validate_particle_perception_sigmas,
    validate_evolution_noise_ratio,
)


class AdaptiveParticleManager(ParticleManager):


    def __init__(self, num_particles: int = 128,
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 evolution_noise: float = DEFAULT_EVOLUTION_NOISE_RATIO,
                 target_diversity_ratio: float = 0.8,
                 min_variance_ratio: float = 0.3,
                 seed: Optional[int] = None):

        super().__init__(num_particles, param_bounds, evolution_noise, seed=seed)

        self.target_diversity_ratio = target_diversity_ratio
        self.min_variance_ratio = min_variance_ratio


        self.initial_param_variances = {}


        self.diversity_strategies = {
            'adaptive_noise': True,
            'variance_injection': True,
            'hybrid_sampling': True,
            'guided_perturbation': True
        }

    def initialize_particles_with_diversity_tracking(self,
                                                   initialization_method: str = 'lhs',
                                                   prior_mean: Optional[Dict[str, float]] = None,
                                                   prior_std: Optional[Dict[str, float]] = None) -> List[Particle]:

        particles = self.initialize_particles(initialization_method, prior_mean, prior_std)


        for param_name in particles[0].theta.keys():
            values = [p.theta[param_name] for p in particles]
            self.initial_param_variances[param_name] = np.var(values)

        print(f"Initial parameter variance baseline:")
        for param, var in self.initial_param_variances.items():
            print(f"   {param}: {var:.6f}")

        return particles

    def adaptive_evolve_particles(self,
                                particles: List[Particle],
                                window_idx: int,
                                previous_mean: Optional[Dict[str, float]] = None,
                                noise_scale: Optional[float] = None) -> List[Particle]:

        if noise_scale is None:
            noise_scale = self.evolution_noise
        else:
            noise_scale = validate_evolution_noise_ratio(noise_scale)

        if noise_scale == 0.0:
            return [particle.copy() for particle in particles]


        current_diversity = self._compute_particle_diversity(particles)


        target_diversity = self._compute_target_diversity()


        if current_diversity['avg_variance_ratio'] < self.min_variance_ratio:
            print(f"Window {window_idx}: diversity too low ({current_diversity['avg_variance_ratio']:.3f}), running forced diversity rebuild")
            return self._diversity_reconstruction(particles, previous_mean, target_diversity)

        elif current_diversity['avg_variance_ratio'] < self.target_diversity_ratio:
            print(f"Window {window_idx}: diversity below target ({current_diversity['avg_variance_ratio']:.3f}), running strengthened evolution")
            return self._enhanced_evolution(particles, previous_mean, noise_scale)

        else:
            print(f"Window {window_idx}: diversity is healthy ({current_diversity['avg_variance_ratio']:.3f}), running standard evolution")
            return self._guided_evolution(particles, previous_mean, noise_scale)

    def _compute_particle_diversity(self, particles: List[Particle]) -> Dict[str, float]:
        diversity_metrics = {}
        param_variances = {}
        variance_ratios = {}

        for param_name in particles[0].theta.keys():
            values = np.array([p.theta[param_name] for p in particles])
            current_var = np.var(values)
            param_variances[param_name] = current_var


            if param_name in self.initial_param_variances:
                initial_var = self.initial_param_variances[param_name]
                variance_ratios[param_name] = current_var / (initial_var + 1e-10)
            else:
                variance_ratios[param_name] = 1.0

        diversity_metrics['param_variances'] = param_variances
        diversity_metrics['variance_ratios'] = variance_ratios
        diversity_metrics['avg_variance_ratio'] = np.mean(list(variance_ratios.values()))
        diversity_metrics['min_variance_ratio'] = np.min(list(variance_ratios.values()))

        return diversity_metrics

    def _compute_target_diversity(self) -> Dict[str, float]:
        target_variances = {}
        for param_name, initial_var in self.initial_param_variances.items():
            target_variances[param_name] = initial_var * self.target_diversity_ratio
        return target_variances

    def _diversity_reconstruction(self,
                                particles: List[Particle],
                                previous_mean: Optional[Dict[str, float]],
                                target_diversity: Dict[str, float]) -> List[Particle]:


        weights = np.array([p.weight for p in particles])
        keep_ratio = 0.2
        keep_count = max(1, int(self.num_particles * keep_ratio))


        top_indices = np.argsort(weights)[-keep_count:]
        kept_particles = [particles[i].copy() for i in top_indices]


        if previous_mean is not None:
            evolution_center = previous_mean.copy()
        else:
            evolution_center = {}
            total_weight = sum(p.weight for p in kept_particles)
            for param_name in particles[0].theta.keys():
                weighted_sum = sum(p.theta[param_name] * p.weight for p in kept_particles)
                evolution_center[param_name] = weighted_sum / (total_weight + 1e-10)


        new_particles = kept_particles.copy()
        remaining_count = self.num_particles - keep_count

        def draw_theta():
            theta = {}
            for param_name, (min_val, max_val) in self.param_bounds.items():
                center_val = evolution_center.get(param_name, (min_val + max_val) / 2)
                target_std = np.sqrt(target_diversity[param_name])

                if param_name == 'delay_steps':

                    choices = list(range(int(min_val), int(max_val) + 1))
                    new_val = self.rng.choice(choices)
                else:

                    new_val = self.rng.normal(center_val, target_std)
                    new_val = np.clip(new_val, min_val, max_val)

                theta[param_name] = new_val
            return theta


        new_particles.extend(
            self._collect_valid_particles(draw_theta, count=remaining_count)
        )


        for p in new_particles:
            p.weight = 1.0 / self.num_particles
            p.log_weight = np.log(1.0 / self.num_particles)

        return new_particles

    def _enhanced_evolution(self,
                          particles: List[Particle],
                          previous_mean: Optional[Dict[str, float]],
                          noise_scale: float) -> List[Particle]:

        evolved_particles = []

        for particle in particles:
            _validate_particle_perception_sigmas(particle.theta)
            new_particle = particle.copy()

            for param_name, value in new_particle.theta.items():
                min_val, max_val = self.param_bounds[param_name]


                if previous_mean is not None:

                    target_val = (0.3 * previous_mean.get(param_name, value) +
                                 0.7 * value)
                else:
                    target_val = value

                if param_name == 'delay_steps':

                    noise_prob = min(noise_scale * 10, 0.7)
                    if self.rng.random() < noise_prob:
                        delta = self.rng.choice([-3, -2, -1, 0, 1, 2, 3])
                        new_val = int(target_val + delta)
                        new_val = np.clip(new_val, int(min_val), int(max_val))
                        new_particle.theta[param_name] = new_val
                    else:
                        new_particle.theta[param_name] = int(target_val)
                else:

                    noise_std = noise_scale * (max_val - min_val)
                    noise = self.rng.normal(0, noise_std)
                    new_val = target_val + noise
                    new_val = np.clip(new_val, min_val, max_val)
                    new_particle.theta[param_name] = new_val

            try:
                _validate_particle_perception_sigmas(new_particle.theta)
            except ValueError:
                # A constrained proposal outside the paper domain is rejected,
                # preserving the previous valid particle without projection.
                new_particle = particle.copy()
            evolved_particles.append(new_particle)

        return evolved_particles

    def _guided_evolution(self,
                        particles: List[Particle],
                        previous_mean: Optional[Dict[str, float]],
                        noise_scale: float) -> List[Particle]:

        evolved_particles = []

        for particle in particles:
            _validate_particle_perception_sigmas(particle.theta)
            new_particle = particle.copy()

            for param_name, value in new_particle.theta.items():
                min_val, max_val = self.param_bounds[param_name]


                if previous_mean is not None:
                    guidance_strength = 0.1
                    prev_val = previous_mean.get(param_name, value)
                    guided_val = (1 - guidance_strength) * value + guidance_strength * prev_val
                else:
                    guided_val = value

                if param_name == 'delay_steps':
                    noise_prob = min(noise_scale * 5, 0.3)
                    if self.rng.random() < noise_prob:
                        delta = self.rng.choice([-1, 0, 1])
                        new_val = int(guided_val + delta)
                        new_val = np.clip(new_val, int(min_val), int(max_val))
                        new_particle.theta[param_name] = new_val
                    else:
                        new_particle.theta[param_name] = int(guided_val)
                else:
                    noise_std = noise_scale * (max_val - min_val)
                    noise = self.rng.normal(0, noise_std)
                    new_val = guided_val + noise
                    new_val = np.clip(new_val, min_val, max_val)
                    new_particle.theta[param_name] = new_val

            try:
                _validate_particle_perception_sigmas(new_particle.theta)
            except ValueError:
                # A constrained proposal outside the paper domain is rejected,
                # preserving the previous valid particle without projection.
                new_particle = particle.copy()
            evolved_particles.append(new_particle)

        return evolved_particles

    def hybrid_resample(self, particles: List[Particle],
                       previous_mean: Optional[Dict[str, float]] = None,
                       preserve_diversity_ratio: float = 0.3) -> List[Particle]:

        weights = np.array([p.weight for p in particles])


        diversity_count = int(self.num_particles * preserve_diversity_ratio)
        weighted_count = self.num_particles - diversity_count

        new_particles = []


        if weighted_count > 0:
            indices = self._systematic_resample(weights)[:weighted_count]
            for idx in indices:
                new_particle = particles[idx].copy()
                new_particle.weight = 1.0 / self.num_particles
                new_particle.log_weight = np.log(1.0 / self.num_particles)
                new_particles.append(new_particle)


        if diversity_count > 0:
            diversity_particles = self._generate_diversity_particles(
                diversity_count, previous_mean
            )
            new_particles.extend(diversity_particles)

        return new_particles

    def _generate_diversity_particles(self,
                                    count: int,
                                    guide_mean: Optional[Dict[str, float]] = None) -> List[Particle]:

        def draw_theta():
            theta = {}
            for param_name, (min_val, max_val) in self.param_bounds.items():
                if guide_mean is not None and param_name in guide_mean:
                    center = guide_mean[param_name]
                else:
                    center = (min_val + max_val) / 2


                if param_name in self.initial_param_variances:
                    target_std = np.sqrt(self.initial_param_variances[param_name] * self.target_diversity_ratio)
                else:
                    target_std = (max_val - min_val) / 4

                if param_name == 'delay_steps':

                    new_val = self.rng.integers(int(min_val), int(max_val) + 1)
                else:

                    new_val = self.rng.normal(center, target_std)
                    new_val = np.clip(new_val, min_val, max_val)

                theta[param_name] = new_val
            return theta


        return self._collect_valid_particles(draw_theta, count=count)
