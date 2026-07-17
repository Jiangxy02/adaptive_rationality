#!/usr/bin/env python3
"""Particle management for initialization, evolution, and resampling."""


import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass
import copy

from common.cognitive_input import validate_perception_sigmas


DEFAULT_EVOLUTION_NOISE_RATIO = 0.15
_MAX_CONSTRAINED_SAMPLING_ATTEMPTS_PER_PARTICLE = 1000


def validate_evolution_noise_ratio(value: float) -> float:
    """Return a finite evolution-noise ratio in the public ``[0, 1]`` domain."""
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"evolution_noise must be a finite ratio in [0, 1], got {value!r}"
        ) from exc
    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"evolution_noise must be a finite ratio in [0, 1], got {value!r}"
        )
    return ratio


def _validate_particle_perception_sigmas(theta: Dict[str, float]) -> Tuple[float, float]:
    return validate_perception_sigmas(
        theta.get('perception_sigma0', 0.1),
        theta.get('perception_sigma_max', 0.8),
    )


def perception_sigma_domain_has_positive_probability(
    sigma0_bounds: Tuple[float, float],
    sigma_max_bounds: Tuple[float, float],
) -> bool:
    """Whether independent uniform draws can satisfy sigma0 <= sigma_max."""
    sigma0_min, sigma0_max = map(float, sigma0_bounds)
    sigma_max_min, sigma_max_max = map(float, sigma_max_bounds)
    if sigma0_min < sigma_max_max:
        return True
    return (
        sigma0_min == sigma0_max
        and sigma0_max == sigma_max_min
        and sigma_max_min == sigma_max_max
    )


@dataclass
class Particle:
    """Particle state."""
    theta: Dict[str, float]
    weight: float = 1.0
    log_weight: float = 0.0

    def copy(self):
        """Deep-copy this particle."""
        return Particle(
            theta=copy.deepcopy(self.theta),
            weight=self.weight,
            log_weight=self.log_weight
        )


class ParticleManager:
    """Initialize, evolve, and resample particles."""

    def __init__(self, num_particles: int = 128,
                 param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 evolution_noise: float = DEFAULT_EVOLUTION_NOISE_RATIO,
                 seed: Optional[int] = None):
        """Initialize the particle manager.

        Args:
            num_particles: Number of particles to maintain.
            param_bounds: Mapping from parameter names to ``(minimum, maximum)``.
            evolution_noise: Standard-deviation ratio relative to each full
                continuous parameter range.
            seed: Seed used for initialization, evolution, and resampling.
        """
        self.num_particles = num_particles
        self.evolution_noise = validate_evolution_noise_ratio(evolution_noise)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if param_bounds is None:
            self.param_bounds = {
                'perception_sigma0': (0.0, 1.0),
                'perception_sigma_max': (0.0, 5.0),
                'bias_inverse_tta_coef': (0.5, 5.0),
                'delay_steps': (0, 10)
            }
        else:
            self.param_bounds = param_bounds

        sigma0_bounds = self.param_bounds.get('perception_sigma0')
        sigma_max_bounds = self.param_bounds.get('perception_sigma_max')
        for name, bounds in (
            ('perception_sigma0', sigma0_bounds),
            ('perception_sigma_max', sigma_max_bounds),
        ):
            if bounds is None:
                continue
            minimum, maximum = map(float, bounds)
            if not np.isfinite([minimum, maximum]).all() or minimum < 0 or minimum > maximum:
                raise ValueError(f"invalid non-negative meter bounds for {name}: {bounds!r}")
        if (
            sigma0_bounds is not None
            and sigma_max_bounds is not None
            and not perception_sigma_domain_has_positive_probability(
                sigma0_bounds, sigma_max_bounds
            )
        ):
            raise ValueError(
                "perception sigma bounds contain no positive-probability sample "
                "satisfying sigma0 <= sigma_max"
            )

        self.particles: List[Particle] = []
        self.effective_sample_size = num_particles

    def _particle_from_theta(self, theta: Dict[str, float]) -> Particle:
        _validate_particle_perception_sigmas(theta)
        weight = 1.0 / self.num_particles
        return Particle(theta=theta, weight=weight, log_weight=np.log(weight))

    def _collect_valid_particles(self, draw_theta,
                                 count: Optional[int] = None) -> List[Particle]:
        """Draw from the requested proposal and reject samples outside the paper domain."""
        if count is None:
            count = self.num_particles
        if count <= 0:
            return []
        particles = []
        max_attempts = max(
            count * _MAX_CONSTRAINED_SAMPLING_ATTEMPTS_PER_PARTICLE,
            1,
        )
        for _ in range(max_attempts):
            theta = draw_theta()
            try:
                particle = self._particle_from_theta(theta)
            except ValueError:
                continue
            particles.append(particle)
            if len(particles) == count:
                return particles
        raise ValueError(
            "unable to sample enough particles satisfying perception sigma0 <= sigma_max"
        )

    def initialize_particles(self,
                           initialization_method: str = 'lhs',
                           prior_mean: Optional[Dict[str, float]] = None,
                           prior_std: Optional[Dict[str, float]] = None) -> List[Particle]:
        """Initialize the particle set.

        Args:
            initialization_method: ``uniform``, ``gaussian``, or ``lhs``.
            prior_mean: Gaussian proposal means, used only for ``gaussian``.
            prior_std: Gaussian proposal standard deviations.

        Returns:
            The initialized particles with uniform weights.
        """
        if initialization_method == 'uniform':
            particles = self._initialize_uniform()
        elif initialization_method == 'gaussian':
            particles = self._initialize_gaussian(prior_mean, prior_std)
        elif initialization_method == 'lhs':
            particles = self._initialize_lhs()
        else:
            raise ValueError(f"unsupported initialization method: {initialization_method}")

        self.particles = particles
        return particles

    def _initialize_uniform(self) -> List[Particle]:
        """Initialize particles from a uniform distribution."""
        def draw_theta():
            theta = {}
            for param_name, (min_val, max_val) in self.param_bounds.items():
                if param_name == 'delay_steps':
                    theta[param_name] = self.rng.integers(int(min_val), int(max_val) + 1)
                else:
                    theta[param_name] = self.rng.uniform(min_val, max_val)
            return theta

        return self._collect_valid_particles(draw_theta)

    def _initialize_gaussian(self, prior_mean: Dict[str, float],
                           prior_std: Dict[str, float]) -> List[Particle]:
        """Initialize particles from a Gaussian proposal."""
        def draw_theta():
            theta = {}
            for param_name, (min_val, max_val) in self.param_bounds.items():
                mean = prior_mean.get(param_name, (min_val + max_val) / 2)
                std = prior_std.get(param_name, (max_val - min_val) / 6)

                value = self.rng.normal(mean, std)
                value = np.clip(value, min_val, max_val)

                if param_name == 'delay_steps':
                    value = int(round(value))

                theta[param_name] = value
            return theta

        return self._collect_valid_particles(draw_theta)

    def _initialize_lhs(self) -> List[Particle]:
        """Initialize particles with Latin hypercube sampling."""
        try:
            from scipy.stats import qmc
        except ImportError:
            print("scipy is not installed; falling back to uniform initialization")
            return self._initialize_uniform()

        param_names = list(self.param_bounds.keys())
        n_dims = len(param_names)
        lhs_seed = int(self.rng.integers(0, np.iinfo(np.uint32).max))
        sampler = qmc.LatinHypercube(d=n_dims, seed=lhs_seed)

        particles = []
        max_batches = _MAX_CONSTRAINED_SAMPLING_ATTEMPTS_PER_PARTICLE
        for _ in range(max_batches):
            remaining = self.num_particles - len(particles)
            samples = sampler.random(n=max(remaining, 1))
            for sample in samples:
                theta = {}
                for j, param_name in enumerate(param_names):
                    min_val, max_val = self.param_bounds[param_name]
                    value = min_val + sample[j] * (max_val - min_val)
                    if param_name == 'delay_steps':
                        value = int(round(value))
                    theta[param_name] = value
                try:
                    particles.append(self._particle_from_theta(theta))
                except ValueError:
                    continue
                if len(particles) == self.num_particles:
                    return particles
        raise ValueError(
            "unable to sample enough LHS particles satisfying "
            "perception sigma0 <= sigma_max"
        )

    def evolve_particles(self, particles: List[Particle],
                        noise_scale: Optional[float] = None,
                        diversity_boost: bool = False) -> List[Particle]:
        """Evolve particles by adding process noise.

        Args:
            particles: Current particle population.
            noise_scale: Standard-deviation ratio relative to the full search
                range, or ``None`` to use the configured default.
            diversity_boost: Compatibility argument; it does not alter the
                public noise ratio.

        Returns:
            The evolved particle population.
        """
        if noise_scale is None:
            noise_scale = self.evolution_noise
        else:
            noise_scale = validate_evolution_noise_ratio(noise_scale)

        evolved_particles = []
        for particle in particles:
            _validate_particle_perception_sigmas(particle.theta)
            new_particle = particle.copy()

            for param_name, value in new_particle.theta.items():
                min_val, max_val = self.param_bounds[param_name]

                if param_name == 'delay_steps':
                    noise_prob = min(noise_scale * 5, 0.5)
                    if self.rng.random() < noise_prob:
                        delta = self.rng.choice([-2, -1, 0, 1, 2])
                        new_value = int(value + delta)
                        new_value = np.clip(new_value, int(min_val), int(max_val))
                        new_particle.theta[param_name] = new_value
                else:
                    noise_std = noise_scale * (max_val - min_val)
                    noise = self.rng.normal(0, noise_std)
                    new_value = value + noise
                    new_value = np.clip(new_value, min_val, max_val)
                    new_particle.theta[param_name] = new_value

            try:
                _validate_particle_perception_sigmas(new_particle.theta)
            except ValueError:
                # A constrained proposal outside the paper domain is rejected,
                # preserving the previous valid particle without projection.
                new_particle = particle.copy()
            evolved_particles.append(new_particle)

        return evolved_particles

    def update_weights(self, particles: List[Particle],
                      log_likelihoods: np.ndarray) -> List[Particle]:
        """Update particle weights from one log-likelihood per particle."""
        if not particles:
            raise ValueError("cannot update weights for an empty particle set")
        log_likelihoods = np.asarray(log_likelihoods, dtype=float)
        if log_likelihoods.shape != (len(particles),):
            raise ValueError(
                "log_likelihoods must contain exactly one value per particle"
            )
        if not np.isfinite(log_likelihoods).all():
            raise ValueError("log_likelihoods must contain only finite values")

        for i, particle in enumerate(particles):
            particle.log_weight += log_likelihoods[i]

        log_weights = np.array([p.log_weight for p in particles])
        if not np.isfinite(log_weights).all():
            raise ValueError("updated particle log weights must remain finite")
        max_log_weight = np.max(log_weights)
        log_weights = log_weights - max_log_weight

        weights = np.exp(log_weights)
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise ValueError("particle weights cannot be normalized to a finite sum")
        weights = weights / weight_sum
        if not np.isfinite(weights).all():
            raise ValueError("normalized particle weights must remain finite")

        for i, particle in enumerate(particles):
            particle.weight = weights[i]
            particle.log_weight = np.log(weights[i] + 1e-10)

        return particles

    def compute_ess(self, particles: List[Particle]) -> float:
        """Compute the effective sample size (ESS)."""
        weights = np.array([p.weight for p in particles])
        ess = 1.0 / np.sum(weights ** 2)
        self.effective_sample_size = ess
        return ess

    def resample(self, particles: List[Particle],
                method: str = 'systematic') -> List[Particle]:
        """Resample particles."""
        weights = np.array([p.weight for p in particles])

        if method == 'multinomial':
            indices = self._multinomial_resample(weights)
        elif method == 'systematic':
            indices = self._systematic_resample(weights)
        elif method == 'residual':
            indices = self._residual_resample(weights)
        else:
            raise ValueError(f"unsupported resampling method: {method}")

        new_particles = []
        for idx in indices:
            new_particle = particles[idx].copy()
            new_particle.weight = 1.0 / self.num_particles
            new_particle.log_weight = np.log(1.0 / self.num_particles)
            new_particles.append(new_particle)

        return new_particles

    def _multinomial_resample(self, weights: np.ndarray) -> np.ndarray:
        """Multinomial resampling."""
        return self.rng.choice(
            len(weights),
            size=self.num_particles,
            replace=True,
            p=weights
        )

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """Systematic resampling."""
        N = self.num_particles
        positions = (self.rng.random() + np.arange(N)) / N

        cumulative_sum = np.cumsum(weights)
        indices = np.zeros(N, dtype=int)
        i, j = 0, 0

        while i < N:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1

        return indices

    def _residual_resample(self, weights: np.ndarray) -> np.ndarray:
        """Residual resampling."""
        N = self.num_particles
        indices = []

        num_copies = (N * weights).astype(int)
        for i, n in enumerate(num_copies):
            indices.extend([i] * n)

        residual = N - len(indices)
        if residual > 0:
            residual_weights = N * weights - num_copies
            residual_weights /= np.sum(residual_weights)
            indices.extend(
                self.rng.choice(len(weights), residual, p=residual_weights)
            )

        return np.array(indices)

    def get_posterior_statistics(self, particles: List[Particle]) -> Dict[str, Dict[str, float]]:
        """Compute posterior summary statistics."""
        param_names = list(particles[0].theta.keys())
        weights = np.array([p.weight for p in particles])

        stats = {}

        mean_params = {}
        for param_name in param_names:
            values = np.array([p.theta[param_name] for p in particles])
            mean_params[param_name] = np.sum(weights * values)
        stats['mean'] = mean_params

        map_idx = np.argmax(weights)
        stats['map'] = particles[map_idx].theta.copy()

        std_params = {}
        for param_name in param_names:
            values = np.array([p.theta[param_name] for p in particles])
            mean = mean_params[param_name]
            variance = np.sum(weights * (values - mean) ** 2)
            std_params[param_name] = np.sqrt(variance)
        stats['std'] = std_params

        ci_params = {}
        for param_name in param_names:
            values = np.array([p.theta[param_name] for p in particles])
            sorted_indices = np.argsort(values)
            sorted_values = values[sorted_indices]
            sorted_weights = weights[sorted_indices]
            cumsum = np.cumsum(sorted_weights)

            ci_lower = sorted_values[np.searchsorted(cumsum, 0.025)]
            ci_upper = sorted_values[np.searchsorted(cumsum, 0.975)]
            ci_params[param_name] = (ci_lower, ci_upper)
        stats['ci_95'] = ci_params

        stats['ess'] = self.effective_sample_size

        stats['weight_mean'] = float(np.mean(weights))
        stats['weight_max'] = float(np.max(weights))
        stats['weight_min'] = float(np.min(weights))
        stats['weight_std'] = float(np.std(weights))
        stats['weight_median'] = float(np.median(weights))

        normalized_weights = weights / np.sum(weights)
        normalized_weights = normalized_weights + 1e-10
        entropy = -np.sum(normalized_weights * np.log(normalized_weights))
        stats['weight_entropy'] = float(entropy)

        stats['max_weight_particle_idx'] = int(map_idx)

        return stats
