#!/usr/bin/env python3
"""Inject lidar noise in meter space before observation normalization.

The sensor wrapper never mutates true world or vehicle state and never
post-processes already normalized observations.
"""


import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import sys
import os
import time
import matplotlib.pyplot as plt


from metadrive.component.sensors.lidar import Lidar
from metadrive.component.sensors.distance_detector import DistanceDetector
from common.cognitive_input import validate_perception_sigmas

logger = logging.getLogger(__name__)

_PERCEPT_SIGMA0_DEFAULT = 0.1
_PERCEPT_SIGMA_MAX_DEFAULT = 0.8
_PERCEPT_FAR_DISTANCE_DEFAULT = 150.0
_PERCEPT_USE_AR1_DEFAULT = True
_PERCEPT_RHO_DEFAULT = 0.8
_PERCEPT_USE_LOWPASS_DEFAULT = False
_PERCEPT_ALPHA_DEFAULT = 0.7
_PERCEPT_USE_KF_DEFAULT = True
_PERCEPT_KF_DT_DEFAULT = 0.1
_PERCEPT_KF_Q_DEFAULT = 0.5
_PERCEPT_KF_SIGMA_A_DEFAULT = 3.0
_PERCEPT_KF_Q_SCALE_DEFAULT = 100.0
_PERCEPT_KF_R_FLOOR_DEFAULT = 1e-4
_PERCEPT_KF_INIT_STD_POS_DEFAULT = 5.0
_PERCEPT_KF_INIT_STD_VEL_DEFAULT = 10.0


class PerceptNoiseLidar(Lidar):
    """Apply perception noise and optional filtering while preserving the Lidar API."""

    def __init__(self, engine, noise_config=None, parent_module=None):
        super().__init__(engine)


        self.noise_config = noise_config or {}
        self.parent_module = parent_module
        self.random_seed = self.noise_config.get('random_seed')
        self.rng = np.random.default_rng(self.random_seed)


        self.sigma0 = self.noise_config.get('sigma0', _PERCEPT_SIGMA0_DEFAULT)
        self.sigma_max = self.noise_config.get('sigma_max', _PERCEPT_SIGMA_MAX_DEFAULT)


        self.far_distance = self.noise_config.get('far_distance', _PERCEPT_FAR_DISTANCE_DEFAULT)
        self.recompute_k_from_sigma_max()
        self.use_ar1 = self.noise_config.get('use_ar1', _PERCEPT_USE_AR1_DEFAULT)
        self.rho = self.noise_config.get('rho', _PERCEPT_RHO_DEFAULT)
        self.use_lowpass = self.noise_config.get('use_lowpass', _PERCEPT_USE_LOWPASS_DEFAULT)
        self.alpha = self.noise_config.get('alpha', _PERCEPT_ALPHA_DEFAULT)


        self.use_kf = self.noise_config.get('use_kf', _PERCEPT_USE_KF_DEFAULT)
        self.kf_dt = float(self.noise_config.get('kf_dt', _PERCEPT_KF_DT_DEFAULT))
        self.kf_q = float(self.noise_config.get('kf_q', _PERCEPT_KF_Q_DEFAULT))
        self.kf_sigma_a = float(self.noise_config.get('kf_sigma_a', _PERCEPT_KF_SIGMA_A_DEFAULT))
        self.kf_q_scale = float(self.noise_config.get('kf_q_scale', _PERCEPT_KF_Q_SCALE_DEFAULT))
        self.kf_r_floor = float(self.noise_config.get('kf_r_floor', _PERCEPT_KF_R_FLOOR_DEFAULT))
        self.kf_init_std_pos = float(self.noise_config.get('kf_init_std_pos', _PERCEPT_KF_INIT_STD_POS_DEFAULT))
        self.kf_init_std_vel = float(self.noise_config.get('kf_init_std_vel', _PERCEPT_KF_INIT_STD_VEL_DEFAULT))


        self.num_beams = 0
        self.ar1_states = None
        self.prev_distances = None
        self.initialized = False


        self.kf_state = None  # shape: (N, 2)
        self.kf_P = None      # shape: (N, 2, 2)


        self.last_original_distances = None
        self.last_noisy_distances = None
        self.last_noise_levels = None
        self.front_beam_index = None


        self.front_beam_history = {
            'original_distances': [],
            'noisy_distances': [],
            'kf_estimates': [],
            'timestamps': []
        }


        self._clip_log_counter = 0

        logger.debug(f"PerceptNoiseLidar initialized: sigma0={self.sigma0:.3f}, sigma_max={self.sigma_max:.3f}, "
                    f"k={self.k:.3f}")

    def recompute_k_from_sigma_max(self):
        sigma0, sigma_max = validate_perception_sigmas(self.sigma0, self.sigma_max)
        far_distance = float(self.far_distance)
        if not np.isfinite(far_distance) or far_distance <= 0:
            raise ValueError(f"far_distance must be > 0 (got {far_distance})")
        self.k = np.sqrt(sigma_max**2 - sigma0**2) / far_distance
        return self.k

    def _reset_kf_state(self):
        """Rebuild the complete finite initial state and covariance for every beam."""
        if self.kf_state is None or self.kf_P is None:
            return

        self.kf_state.fill(0.0)
        self.kf_state[:, 0] = 50.0

        self.kf_P.fill(0.0)
        self.kf_P[:, 0, 0] = self.kf_init_std_pos ** 2
        self.kf_P[:, 1, 1] = self.kf_init_std_vel ** 2

    def _initialize_states(self, num_beams: int):
        if self.num_beams != num_beams or not self.initialized:
            self.num_beams = num_beams
            self.ar1_states = np.zeros(num_beams, dtype=np.float32)
            self.prev_distances = np.ones(num_beams, dtype=np.float32) * 300.0
            self.initialized = True


            if self.use_kf:
                self.kf_state = np.empty((num_beams, 2), dtype=np.float32)
                self.kf_P = np.empty((num_beams, 2, 2), dtype=np.float32)
                self._reset_kf_state()

            logger.debug(f"PerceptNoiseLidar state initialized: {num_beams} beams")

    def _sigma_weber(self, d, sigma_min=None, k=None, sigma_max=None, d_sat=None):
        """Compute capped Weber-law noise on non-negative meter distances."""
        if sigma_min is None:
            sigma_min = self.sigma0
        if k is None:
            k = self.k
        if sigma_max is None:
            sigma_max = self.sigma_max


        d_eff = np.maximum(d, 0.0)


        sig = np.sqrt(sigma_min**2 + (k * d_eff)**2)


        return np.minimum(sig, sigma_max)

    def reset(self):
        if self.initialized:
            self.ar1_states.fill(0.0)
            self.prev_distances.fill(300.0)


            if self.use_kf and (self.kf_state is not None):
                self._reset_kf_state()

            logger.debug("PerceptNoiseLidar state reset")


        self.front_beam_history = {
            'original_distances': [],
            'noisy_distances': [],
            'kf_estimates': [],
            'timestamps': []
        }

    def perceive(self, base_vehicle, physics_world, num_lasers, distance,
                 height=None, detector_mask=None, show=False):

        self._initialize_states(num_lasers)


        if self.front_beam_index is None:
            self.front_beam_index = num_lasers // 2


        result, detected_objects = super().perceive(
            base_vehicle, physics_world, num_lasers, distance,
            height, detector_mask, show
        )


        normalized_distances = np.array(result, dtype=np.float32)


        actual_distances = normalized_distances * distance


        self.last_original_distances = actual_distances.copy()


        noisy_distances = self._apply_noise_models(actual_distances, distance)


        self.last_noisy_distances = noisy_distances.copy()


        self.last_noise_levels = np.abs(noisy_distances - actual_distances)


        if self.front_beam_index is not None:
            import time
            import logging
            logger = logging.getLogger(__name__)



            total_beams = len(actual_distances)
            front_beam_start = int(total_beams * 0.5 - total_beams * 0.25)
            front_beam_end = int(total_beams * 0.5 + total_beams * 0.25)


            fastest_changing_beam = self._find_fastest_changing_beam(
                actual_distances[front_beam_start:front_beam_end],
                front_beam_start
            )

            if fastest_changing_beam is not None:
                beam_idx = fastest_changing_beam
                self.front_beam_history['original_distances'].append(float(actual_distances[beam_idx]))
                self.front_beam_history['noisy_distances'].append(float(noisy_distances[beam_idx]))
                self.front_beam_history['timestamps'].append(time.time())
                logger.debug(f"Using the fastest-changing radar beam: beam {beam_idx}, distance={actual_distances[beam_idx]:.3f}m")
            else:
                # Falling back to the minimum distance
                valid_actual = actual_distances[actual_distances < distance]
                valid_noisy = noisy_distances[noisy_distances < distance]

                if len(valid_actual) > 0 and len(valid_noisy) > 0:
                    min_distance = float(np.min(valid_actual))
                    min_noisy_distance = float(np.min(valid_noisy))
                    self.front_beam_history['original_distances'].append(min_distance)
                    self.front_beam_history['noisy_distances'].append(min_noisy_distance)
                    self.front_beam_history['timestamps'].append(time.time())
                    logger.debug(f"Falling back to the minimum distance: {min_distance:.3f}m")
                else:

                    default_distance = distance * 0.8
                    self.front_beam_history['original_distances'].append(default_distance)
                    self.front_beam_history['noisy_distances'].append(default_distance)
                    self.front_beam_history['timestamps'].append(time.time())
                    logger.debug(f"No valid nearby detection; using the default distance: {default_distance:.3f}m")


            if self.use_kf and self.kf_state is not None:
                kf_estimate = float(self.kf_state[self.front_beam_index, 0])
                self.front_beam_history['kf_estimates'].append(kf_estimate)
            else:

                self.front_beam_history['kf_estimates'].append(float(noisy_distances[self.front_beam_index]))


        normalized_noisy = np.clip(noisy_distances / distance, 0.0, 1.0)


        if self.parent_module and hasattr(self.parent_module, '_update_radar_visualization'):
            try:

                env = getattr(self.parent_module, 'attached_env', None)
                if env:
                    self.parent_module._update_radar_visualization(env)
            except Exception as e:

                pass

        return normalized_noisy.tolist(), detected_objects

    def get_front_beam_data(self):
        if (self.last_original_distances is None or
            self.last_noisy_distances is None or
            self.last_noise_levels is None or
            self.front_beam_index is None):
            return {
                'original_distance': 0.0,
                'noisy_distance': 0.0,
                'noise_level': 0.0,
                'beam_index': -1
            }

        return {
            'original_distance': float(self.last_original_distances[self.front_beam_index]),
            'noisy_distance': float(self.last_noisy_distances[self.front_beam_index]),
            'noise_level': float(self.last_noise_levels[self.front_beam_index]),
            'beam_index': int(self.front_beam_index)
        }

    def get_closest_beam_data(self):
        if (self.last_original_distances is None or
            self.last_noisy_distances is None or
            self.last_noise_levels is None):
            return {
                'original_distance': 0.0,
                'noisy_distance': 0.0,
                'noise_level': 0.0,
                'beam_index': -1
            }


        valid_distances = self.last_original_distances.copy()

        max_distance = 299.5
        valid_mask = valid_distances < max_distance

        if not np.any(valid_mask):

            front_data = self.get_front_beam_data()
            front_data['beam_index'] = 120
            return front_data


        valid_distances_filtered = np.where(valid_mask, valid_distances, np.inf)
        closest_index = np.argmin(valid_distances_filtered)

        return {
            'original_distance': float(self.last_original_distances[closest_index]),
            'noisy_distance': float(self.last_noisy_distances[closest_index]),
            'noise_level': float(self.last_noise_levels[closest_index]),
            'beam_index': int(closest_index)
        }

    def get_front_beam_history(self):
        return {
            'original_distances': self.front_beam_history['original_distances'].copy(),
            'noisy_distances': self.front_beam_history['noisy_distances'].copy(),
            'kf_estimates': self.front_beam_history['kf_estimates'].copy(),
            'timestamps': self.front_beam_history['timestamps'].copy(),
            'length': len(self.front_beam_history['original_distances'])
        }

    def get_latest_processed_distances(self):
        if self.last_noisy_distances is not None:
            return self.last_noisy_distances.copy()
        return None

    def _find_fastest_changing_beam(self, beam_distances, start_index):
        if not hasattr(self, '_beam_history') or len(self._beam_history) < 2:

            self._beam_history = []
            return None


        current_distances = beam_distances
        previous_distances = self._beam_history[-1]


        min_len = min(len(current_distances), len(previous_distances))
        if min_len == 0:
            return None

        current_distances = current_distances[:min_len]
        previous_distances = previous_distances[:min_len]


        distance_changes = previous_distances - current_distances


        fastest_changing_idx = np.argmax(distance_changes)


        self._beam_history.append(current_distances.copy())


        if len(self._beam_history) > 10:
            self._beam_history.pop(0)

        return start_index + fastest_changing_idx


    def _kf_filter(self, z_distances: np.ndarray, sigma_array: np.ndarray, return_var: bool = False):
        if not self.use_kf or self.kf_state is None or self.kf_P is None:
            if return_var:
                return z_distances, np.zeros_like(z_distances)
            return z_distances

        dt = float(self.kf_dt)
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=np.float32)
        dt2, dt3, dt4 = dt*dt, dt**3, dt**4
        q = self.kf_sigma_a ** 2
        Q_base = np.array([[dt4/4.0, dt3/2.0],
                      [dt3/2.0, dt2      ]], dtype=np.float32) * q
        Q = float(self.kf_q_scale or 1.0) * Q_base
        H = np.array([[1.0, 0.0]], dtype=np.float32)  # z = [1,0]@[r,r_dot]

        out = np.empty_like(z_distances, dtype=np.float32)
        var_out = np.empty_like(z_distances, dtype=np.float32) if return_var else None

        for i in range(self.num_beams):

            x = self.kf_state[i]   # [r, r_dot]
            P = self.kf_P[i]       # 2x2
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q


            z = float(z_distances[i])
            sigma_i = max(float(sigma_array[i]), 1e-6)
            R = np.array([[max(sigma_i * sigma_i, self.kf_r_floor)]], dtype=np.float32)


            y = np.array([[z]]) - (H @ x_pred).reshape(1, 1)
            S = H @ P_pred @ H.T + R
            try:
                K = (P_pred @ H.T) @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = (P_pred @ H.T) @ np.linalg.pinv(S)

            x_new = x_pred + (K @ y).reshape(2)
            I_KH = np.eye(2, dtype=np.float32) - (K @ H)
            P_new = I_KH @ P_pred @ I_KH.T + K @ R @ K.T


            self.kf_state[i] = x_new
            self.kf_P[i] = 0.5 * (P_new + P_new.T)
            out[i] = x_new[0]

            if return_var:
                var_out[i] = P_new[0, 0]

        if return_var:
            return out, var_out
        return out

    def _apply_noise_models(self, distances: np.ndarray, max_range: float) -> np.ndarray:
        noisy_distances = distances.copy()


        if self.sigma0 <= 0 and self.sigma_max <= 0:
            self.prev_distances = noisy_distances.copy()
            return noisy_distances


        if self.sigma0 > 0 or self.k > 0:


            noise_mask = distances < max_range


            valid_distances = distances[noise_mask]
            sigma_array = self._sigma_weber(valid_distances)
            gaussian_noise = self.rng.normal(0, sigma_array)



            noise_limit = 3.0 * sigma_array
            original_noise = gaussian_noise.copy()
            gaussian_noise = np.clip(gaussian_noise, -noise_limit, noise_limit)


            clipped_count = np.sum(np.abs(original_noise) > noise_limit)
            if clipped_count > 0:
                self._clip_log_counter = getattr(self, '_clip_log_counter', 0) + 1

                if self._clip_log_counter % 50 == 0 or self._clip_log_counter <= 5:
                    max_original = np.max(np.abs(original_noise))
                    max_clipped = np.max(np.abs(gaussian_noise))
                    max_sigma = np.max(sigma_array)
                    logger.debug(f"Noise clipping event {self._clip_log_counter} times: {clipped_count}/{len(distances)} beams were clipped, "
                                f"maximum noise {max_original:.2f}→{max_clipped:.2f}m (3σ={3*max_sigma:.2f}m)")


            if self.use_ar1 and self.rho > 0:
                # n_t = ρ * n_{t-1} + sqrt(1-ρ^2) * ξ_t

                self.ar1_states[noise_mask] = (self.rho * self.ar1_states[noise_mask] +
                                              np.sqrt(1 - self.rho**2) * gaussian_noise)
                gaussian_noise = self.ar1_states[noise_mask]


                gaussian_noise = np.clip(gaussian_noise, -noise_limit, noise_limit)


            noisy_distances[noise_mask] += gaussian_noise


            if np.any(noise_mask):
                logger.debug(f"Noise injection: {np.sum(noise_mask)}/{len(distances)} beams with valid distance values, "
                           f"maximum noise : {np.max(np.abs(gaussian_noise)):.3f}m")


        if self.use_kf:

            sigma_meas = self._sigma_weber(distances)
            sigma_meas = np.maximum(sigma_meas, 1e-6)
            noisy_distances = self._kf_filter(noisy_distances, sigma_meas)


        elif self.use_lowpass and self.alpha > 0:
            # d_filt = α * d_noisy + (1-α) * d_prev
            noisy_distances = (self.alpha * noisy_distances +
                             (1 - self.alpha) * self.prev_distances)
            self.prev_distances = noisy_distances.copy()


        noisy_distances = np.clip(noisy_distances, 0.0, max_range)


        self.prev_distances = noisy_distances.copy()

        return noisy_distances



    @staticmethod
    def _ensure_figdir(save_dir: str) -> str:
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def plot_noise_comparison(self, d_true: np.ndarray, d_noisy: np.ndarray, savepath: Optional[str] = None):
        assert d_true.shape == d_noisy.shape, "Distance arrays must have matching shapes"

        noise = d_noisy - d_true
        x = np.arange(len(d_true))

        fig = plt.figure(figsize=(12, 5))


        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(x, d_true, 'b-', label="True Distance (m)", linewidth=2)
        ax1.plot(x, d_noisy, 'r--', marker="o", markersize=3, label="Noisy Distance (m)", alpha=0.7)
        ax1.set_xlabel("Beam Index")
        ax1.set_ylabel("Distance (m)")
        ax1.set_title("Before vs After Noise Injection")
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)


        ax2 = fig.add_subplot(1, 2, 2)
        n, bins, patches = ax2.hist(noise, bins=30, density=True, alpha=0.7, color='orange', edgecolor='black')


        if len(noise) > 1:
            mu = np.mean(noise)
            sigma = np.std(noise)
            x_gauss = np.linspace(bins[0], bins[-1], 100)
            y_gauss = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_gauss - mu) / sigma) ** 2)
            ax2.plot(x_gauss, y_gauss, 'r-', linewidth=2, label=f"Gaussian Fit (μ={mu:.3f}, σ={sigma:.3f})")
            ax2.legend()

        ax2.set_xlabel("Noise (m)")
        ax2.set_ylabel("Density")
        ax2.set_title("Distribution of Injected Noise")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            logger.info(f"Noise comparison plot saved to: {savepath}")

        plt.close(fig)

    def plot_kf_timeseries(self, gt: np.ndarray, meas: np.ndarray, kf: np.ndarray,
                          kf_std: Optional[np.ndarray] = None, savepath: Optional[str] = None):
        assert len(gt) == len(meas) == len(kf), "Time-series arrays must have matching lengths"

        t = np.arange(len(gt))

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)


        ax.plot(t, gt, 'g-', linewidth=2, label="Ground Truth (m)")
        ax.scatter(t, meas, s=20, alpha=0.6, color='red', label="Noisy Measurement (m)")
        ax.plot(t, kf, 'b-', linewidth=2, label="KF Estimate (m)")


        if kf_std is not None:
            ax.fill_between(t, kf - kf_std, kf + kf_std, alpha=0.3, color='blue', label="KF ±1σ")

        ax.set_xlabel("Time Step")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Kalman Filter: Prediction / Observation / Update")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()

        if savepath:
            fig.savefig(savepath, dpi=150, bbox_inches="tight")
            logger.info(f"Kalman-filter time-series plot saved to: {savepath}")

        plt.close(fig)

    def visualize_noise_and_kf_once(self, distances_true_m: np.ndarray, max_range: float, save_dir: str = "figs"):
        self._ensure_figdir(save_dir)
        timestamp = int(time.time())


        self._initialize_states(len(distances_true_m))


        d_noisy = self._apply_noise_models(distances_true_m.copy(), max_range)
        noise_path = os.path.join(save_dir, f"noise_compare_{timestamp}.png")
        self.plot_noise_comparison(distances_true_m, d_noisy, noise_path)


        T = 200
        mean_dist = np.mean(distances_true_m)


        t_seq = np.arange(T)
        gt_seq = mean_dist + 5 * np.sin(0.1 * t_seq)
        gt_seq = np.clip(gt_seq, 0.0, max_range)


        meas_seq = np.zeros(T)
        kf_seq = np.zeros(T)
        kf_var_seq = np.zeros(T)


        self._initialize_states(1)

        for i in range(T):

            d_true_single = np.array([gt_seq[i]])
            d_meas_single = self._apply_noise_models(d_true_single, max_range)
            meas_seq[i] = d_meas_single[0]


            if self.use_kf:
                sigma_array = np.array([self._sigma_weber(gt_seq[i])])
                kf_result, kf_var = self._kf_filter(d_meas_single, sigma_array, return_var=True)
                kf_seq[i] = kf_result[0]
                kf_var_seq[i] = kf_var[0]
            else:
                kf_seq[i] = meas_seq[i]
                kf_var_seq[i] = 0.0


        kf_std_seq = np.sqrt(kf_var_seq) if self.use_kf else None
        kf_path = os.path.join(save_dir, f"kf_timeseries_{timestamp}.png")
        self.plot_kf_timeseries(gt_seq, meas_seq, kf_seq, kf_std_seq, kf_path)

        logger.info(f"Visualization complete; images saved to: {save_dir}")

    @classmethod
    def get_visualization_dir_from_env(cls, env=None):
        if env and hasattr(env, 'visualization_output_dir'):
            base_dir = env.visualization_output_dir
        else:

            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = f"./outputs/fig_cog/cognitive_analysis_{timestamp}"


        cog_influence_dir = os.path.join(base_dir, "cog_influence")
        os.makedirs(cog_influence_dir, exist_ok=True)

        return cog_influence_dir





class CognitivePerceptionModule:

    def __init__(self, noise_config: Optional[Dict[str, Any]] = None):
        self.noise_config = noise_config or self._get_default_config()
        self.original_lidar = None
        self.noise_lidar = None
        self.attached_env = None


        self._last_true_x = 0.0
        self._last_true_y = 0.0
        self._last_noisy_x = 0.0
        self._last_noisy_y = 0.0
        self._last_filtered_x = 0.0
        self._last_filtered_y = 0.0
        self._last_effective_sigma = 0.0


        self.radar_visualization_enabled = False
        self.radar_beam_node = None
        self.radar_beam_drawer = None

        logger.debug("CognitivePerceptionModule initialization complete")

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'sigma0': _PERCEPT_SIGMA0_DEFAULT,
            'sigma_max': _PERCEPT_SIGMA_MAX_DEFAULT,
            'far_distance': _PERCEPT_FAR_DISTANCE_DEFAULT,
            'use_ar1': _PERCEPT_USE_AR1_DEFAULT,
            'rho': _PERCEPT_RHO_DEFAULT,
            'use_lowpass': _PERCEPT_USE_LOWPASS_DEFAULT,
            'alpha': _PERCEPT_ALPHA_DEFAULT,


            'use_kf': _PERCEPT_USE_KF_DEFAULT,
            'kf_dt': _PERCEPT_KF_DT_DEFAULT,
            'kf_q': _PERCEPT_KF_Q_DEFAULT,
            'kf_sigma_a': _PERCEPT_KF_SIGMA_A_DEFAULT,
            'kf_q_scale': _PERCEPT_KF_Q_SCALE_DEFAULT,
            'kf_r_floor': _PERCEPT_KF_R_FLOOR_DEFAULT,
            'kf_init_std_pos': _PERCEPT_KF_INIT_STD_POS_DEFAULT,
            'kf_init_std_vel': _PERCEPT_KF_INIT_STD_VEL_DEFAULT,
        }

    def attach_to_env(self, env):
        self.attached_env = env
        self.original_lidar = env.engine.sensors.get("lidar")


        self.noise_lidar = PerceptNoiseLidar(env.engine, self.noise_config, parent_module=self)



        env.engine.sensors["lidar"] = self.noise_lidar


        self._setup_radar_visualization(env)

        logger.debug("The noisy radar sensor was attached to the environment")
        return True

    def reset(self):
        if self.noise_lidar:
            self.noise_lidar.reset()


        self._last_true_x = 0.0
        self._last_true_y = 0.0
        self._last_noisy_x = 0.0
        self._last_noisy_y = 0.0
        self._last_filtered_x = 0.0
        self._last_filtered_y = 0.0
        self._last_effective_sigma = 0.0

    def get_front_beam_info(self):
        return self.noise_lidar.get_front_beam_data()

    def get_closest_beam_info(self):
        if self.noise_lidar:
            return self.noise_lidar.get_closest_beam_data()
        return {
            'original_distance': 0.0,
            'noisy_distance': 0.0,
            'noise_level': 0.0,
            'beam_index': -1
        }

    def get_processed_radar_distances(self):
        if self.noise_lidar and hasattr(self.noise_lidar, 'get_latest_processed_distances'):
            return self.noise_lidar.get_latest_processed_distances()
        return None

    def enable_radar_visualization(self, enable: bool = True):
        self.radar_visualization_enabled = enable
        if enable:
            logger.info("Radar-beam visualization enabled")
        else:
            logger.info("Radar-beam visualization disabled")
            self._cleanup_radar_visualization()

    def _setup_radar_visualization(self, env):
        if not self.radar_visualization_enabled:
            return

        try:

            engine = env.engine


            self.radar_beam_drawer = engine.make_line_drawer(parent_node=engine.render, thickness=3.0)

            logger.info("Radar-beam visualization setup complete")

        except Exception as e:
            logger.error(f"Radar-beam visualization setup failed: {e}")
            self.radar_visualization_enabled = False

    def _update_radar_visualization(self, env):
        if not self.radar_visualization_enabled or not self.radar_beam_drawer:
            return

        try:

            agent = env.agent
            position = agent.position  # [x, y]
            heading = agent.heading_theta


            import numpy as np
            radar_range = 300.0


            end_x = position[0] + radar_range * np.cos(heading)
            end_y = position[1] + radar_range * np.sin(heading)


            start_height = 1.5
            end_height = 1.5


            self.radar_beam_drawer.reset()


            line_points = [
                [position[0], position[1], start_height],
                [end_x, end_y, end_height]
            ]


            line_colors = [
                [1.0, 0.0, 0.0, 0.8],
            ]


            self.radar_beam_drawer.draw_lines([line_points], [line_colors])

        except Exception as e:
            logger.error(f"Radar-beam visualization update failed: {e}")

    def _cleanup_radar_visualization(self):
        try:
            if self.radar_beam_drawer:
                self.radar_beam_drawer.reset()

                if hasattr(self.radar_beam_drawer, 'removeNode'):
                    self.radar_beam_drawer.removeNode()
                self.radar_beam_drawer = None

            self.radar_beam_node = None
            logger.info("Radar-beam visualization resources cleaned up")

        except Exception as e:
            logger.error(f"Radar-beam visualization cleanup failed: {e}")


    def detach_from_env(self):
        if self.attached_env and self.original_lidar:
            try:
                self.attached_env.engine.sensors["lidar"] = self.original_lidar
                logger.info("Original radar sensor restored")
            except:
                logger.warning("Failed to restore the original radar sensor")


        self._cleanup_radar_visualization()

        self.attached_env = None
        self.noise_lidar = None



    def process_vehicle_state(self, agent, ego_state=None, is_ppo_mode=False):
        """Compatibility no-op: sensor-layer noise leaves vehicle state untouched."""
        pass

    def process_observation(self, obs, ego_state=None, is_ppo_mode=False):
        """Return observations unchanged because noise precedes normalization."""
        return obs

    @property
    def sigma(self):
        return self.noise_config.get('sigma0', 0.1)

    @property
    def enable_kalman(self):
        return self.noise_config.get('use_ar1', True) or self.noise_config.get('use_lowpass', False)

    def generate_visualization(self, save_dir, env=None, test_distances=None, max_range=300.0):



        use_real_data = False
        if test_distances is None:

            if (self.noise_lidar and
                hasattr(self.noise_lidar, 'get_front_beam_history')):
                history = self.noise_lidar.get_front_beam_history()
                if history['length'] > 0:

                    logger.info(f"Generating visualization from real radar data: {history['length']} timesteps of history")
                    use_real_data = True
                    real_data = history
                else:
                    logger.warning("Insufficient real radar data; using synthetic data instead")

            if not use_real_data:

                logger.info("Generating visualization from synthetic data")
                num_beams = 240
                angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
                test_distances = np.ones(num_beams) * max_range


                vehicle_angles = [0, np.pi/6, np.pi/4, np.pi/2, np.pi]
                for angle in vehicle_angles:
                    idx = np.argmin(np.abs(angles - angle))
                    test_distances[idx-2:idx+3] = 15.0 + 5*np.random.randn()


                for i in range(20, 40):
                    test_distances[i] = 25.0 + 3*np.random.randn()

                test_distances = np.clip(test_distances, 0.5, max_range)


        if self.noise_lidar is None:

            logger.error("PerceptNoiseLidar is unavailable, so visualization cannot be generated")
            raise RuntimeError("PerceptNoiseLidar is unavailable. Ensure the cognitive perception module is attached to the environment correctly")

        if not hasattr(self.noise_lidar, 'plot_noise_comparison'):
            logger.error("PerceptNoiseLidar does not expose the required visualization methods")
            raise RuntimeError("PerceptNoiseLidar is missing required visualization methods")

        processor = self.noise_lidar
        logger.info("Generating visualization with the runtime PerceptNoiseLidar instance")


        if not hasattr(processor, '_initialize_states'):
            logger.error("PerceptNoiseLidar is missing the _initialize_states method")
            raise RuntimeError("PerceptNoiseLidar is missing the _initialize_states method")


        timestamp = int(time.time())


        import os
        os.makedirs(save_dir, exist_ok=True)

        if use_real_data:



            orig_distances = np.array(real_data['original_distances'])
            noisy_distances = np.array(real_data['noisy_distances'])

            noise_path = os.path.join(save_dir, f"noise_compare_{timestamp}.png")
            processor.plot_noise_comparison(orig_distances, noisy_distances, noise_path)


            gt_seq = np.array(real_data['original_distances'])
            meas_seq = np.array(real_data['noisy_distances'])
            kf_seq = np.array(real_data['kf_estimates'])


            kf_std_seq = None
            if len(gt_seq) > 1:
                residuals = np.abs(kf_seq - gt_seq)
                avg_residual = np.mean(residuals)
                kf_std_seq = np.full_like(kf_seq, avg_residual)

            kf_path = os.path.join(save_dir, f"kf_timeseries_{timestamp}.png")
            processor.plot_kf_timeseries(gt_seq, meas_seq, kf_seq, kf_std_seq, kf_path)

        else:


            processor._initialize_states(len(test_distances))


            d_noisy = processor._apply_noise_models(test_distances.copy(), max_range)
            noise_path = os.path.join(save_dir, f"noise_compare_{timestamp}.png")
            processor.plot_noise_comparison(test_distances, d_noisy, noise_path)


            T = 200
            mean_dist = np.mean(test_distances)
            t_seq = np.arange(T)
            gt_seq = mean_dist + 5 * np.sin(0.1 * t_seq)
            gt_seq = np.clip(gt_seq, 0.0, max_range)

            meas_seq = np.zeros(T)
            kf_seq = np.zeros(T)
            kf_var_seq = np.zeros(T)


            processor._initialize_states(1)

            for i in range(T):
                d_true_single = np.array([gt_seq[i]])
                d_meas_single = processor._apply_noise_models(d_true_single, max_range)
                meas_seq[i] = d_meas_single[0]


                if hasattr(processor, 'use_kf') and processor.use_kf:
                    sigma_array = np.array([processor._sigma_weber(gt_seq[i])])
                    kf_result, kf_var = processor._kf_filter(d_meas_single, sigma_array, return_var=True)
                    kf_seq[i] = kf_result[0]
                    kf_var_seq[i] = kf_var[0]
                else:
                    kf_seq[i] = meas_seq[i]
                    kf_var_seq[i] = 0.0

            kf_std_seq = np.sqrt(kf_var_seq) if (hasattr(processor, 'use_kf') and processor.use_kf) else None
            kf_path = os.path.join(save_dir, f"kf_timeseries_{timestamp}.png")
            processor.plot_kf_timeseries(gt_seq, meas_seq, kf_seq, kf_std_seq, kf_path)


        if self.noise_lidar is not None:
            logger.info(f"Runtime noise configuration: sigma0={getattr(processor, 'sigma0', 'N/A'):.3f}, "
                       f"k={getattr(processor, 'k', 'N/A'):.3f}, "
                       f"use_kf={getattr(processor, 'use_kf', 'N/A')}")

        logger.info(f"Cognitive perception visualization finished; images saved to: {save_dir}")
        return save_dir


def create_cognitive_perception_module(config: Optional[Dict[str, Any]] = None) -> CognitivePerceptionModule:
    return CognitivePerceptionModule(config)
