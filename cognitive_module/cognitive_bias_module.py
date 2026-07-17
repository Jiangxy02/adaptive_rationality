#!/usr/bin/env python3


import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque
import matplotlib.pyplot as plt
import os

logger = logging.getLogger(__name__)

_BIAS_VISUAL_DETECTION_DISTANCE_DEFAULT = 300.0
_BIAS_INVERSE_TTA_COEF_DEFAULT = 1.0
_BIAS_TTA_THRESHOLD_DEFAULT = 1.0
_BIAS_HISTORY_LENGTH_DEFAULT = 100
_BIAS_EXTENDED_HISTORY_LENGTH_DEFAULT = 1000
_BIAS_VERBOSE_DEFAULT = False


class CognitiveBiasModule:
    """Apply TTA-based looming penalties and retain histories for analysis."""

    def __init__(self, bias_config: Optional[Dict[str, Any]] = None,
        cognitive_perception_module: Optional[Any] = None):
        self.radians_detect_dis = _BIAS_VISUAL_DETECTION_DISTANCE_DEFAULT


        config = bias_config or self._get_default_config()

        self.inverse_tta_coef = config.get('inverse_tta_coef', _BIAS_INVERSE_TTA_COEF_DEFAULT)
        self.tta_threshold = config.get('tta_threshold', _BIAS_TTA_THRESHOLD_DEFAULT)

        self.visual_detection_distance = config.get('visual_detection_distance', _BIAS_VISUAL_DETECTION_DISTANCE_DEFAULT)

        self.bias_history = deque(maxlen=config.get('history_length', _BIAS_HISTORY_LENGTH_DEFAULT))
        self.reward_history = deque(maxlen=config.get('history_length', _BIAS_HISTORY_LENGTH_DEFAULT))
        self.tta_history = deque(maxlen=config.get('history_length', _BIAS_HISTORY_LENGTH_DEFAULT))

        extended_history_length = config.get('extended_history_length', _BIAS_EXTENDED_HISTORY_LENGTH_DEFAULT)
        self.detection_history = deque(maxlen=extended_history_length)
        self.distance_history = deque(maxlen=extended_history_length)
        self.threat_count_history = deque(maxlen=extended_history_length)

        self._step_count = 0
        self._total_bias = 0.0
        self._active_steps = 0

        self.attached_env = None

        self.cognitive_perception_module = cognitive_perception_module

        self.verbose = config.get('verbose', _BIAS_VERBOSE_DEFAULT)
        self._warned_radar_issues = set()

        logger.debug(f"CognitiveBiasModule initialized (looming penalty mode): c={self.inverse_tta_coef}, "
                    f"tta_threshold={self.tta_threshold}")


    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'inverse_tta_coef': _BIAS_INVERSE_TTA_COEF_DEFAULT,
            'tta_threshold': _BIAS_TTA_THRESHOLD_DEFAULT,
            'history_length': _BIAS_HISTORY_LENGTH_DEFAULT,
            'extended_history_length': _BIAS_EXTENDED_HISTORY_LENGTH_DEFAULT,
            'verbose': _BIAS_VERBOSE_DEFAULT,

            'visual_detection_distance': _BIAS_VISUAL_DETECTION_DISTANCE_DEFAULT
        }

    def _warn_radar_once(self, key: str, message: str) -> None:
        """Warn once for unexpected radar extraction failures."""
        if key not in self._warned_radar_issues:
            logger.warning(message)
            self._warned_radar_issues.add(key)

    def attach_to_env(self, env):
        try:
            self.attached_env = env

            if not self._check_tta_support(env):
                logger.warning("The environment may not support inverse_tta")
                return False

            logger.debug("Cognitive bias module attached to the environment")
            return True

        except Exception as e:
            logger.error(f"Failed to attach the cognitive bias module: {e}")
            return False

    def set_cognitive_perception_module(self, cognitive_perception_module):
        self.cognitive_perception_module = cognitive_perception_module
        logger.debug("The cognitive perception module reference was set on the cognitive bias module")

    def _check_tta_support(self, env) -> bool:
        has_tta = (
            hasattr(env, 'current_inverse_tta') or
            (hasattr(env, 'agent') and hasattr(env.agent, 'inverse_tta')) or
            (hasattr(env, 'get_inverse_tta'))
        )

        if not has_tta:
            try:
                logger.debug("The environment does not expose inverse_tta directly; falling back to the info dictionary")
            except:
                pass

        return True

    def get_inverse_tta(self, env, info: Optional[Dict] = None) -> Optional[float]:
        if hasattr(env, 'current_inverse_tta'):
            return env.current_inverse_tta

        if hasattr(env, 'agent'):
            agent = env.agent
            if hasattr(agent, 'inverse_tta'):
                return agent.inverse_tta

            if hasattr(agent, 'compute_inverse_tta'):
                return agent.compute_inverse_tta()

        if hasattr(env, 'get_inverse_tta'):
            return env.get_inverse_tta()

        if info and 'inverse_tta' in info:
            return info['inverse_tta']

        if hasattr(env, 'agent'):
            tta_value = self._compute_inverse_tta_manually(env)
            if tta_value is not None:
                return tta_value

        return None

    def _calculate_relative_speed_from_radar(self, current_min_distance: float, data_source: str) -> Optional[float]:
        estimated_dt = 0.1

        if self.cognitive_perception_module is not None and data_source == "cognitive perception module":
            noise_lidar = getattr(self.cognitive_perception_module, 'noise_lidar', None)
            if noise_lidar is not None:
                front_beam_history = noise_lidar.get_front_beam_history()
                distances = front_beam_history['noisy_distances']
                if front_beam_history['length'] >= 2 and len(distances) >= 2:
                    relative_speed = (distances[-2] - distances[-1]) / estimated_dt
                    return max(0.0, relative_speed)

        if self.distance_history:
            previous_distance = self.distance_history[-1]
            if np.isfinite(previous_distance) and np.isfinite(current_min_distance):
                relative_speed = (
                    previous_distance - current_min_distance
                ) / estimated_dt
                return max(0.0, relative_speed)

        return None

    def _compute_inverse_tta_manually(self, env) -> Optional[float]:
        try:
            agent = env.agent
            agent_speed = agent.speed

            detection_angle_degrees = 30
            max_detection_distance = self.visual_detection_distance

            min_distance = float('inf')
            radar_data_found = False
            beam_count = 0
            valid_beams = 0
            data_source = "unknown"
            radar_errors = []


            if self.cognitive_perception_module is not None:
                try:
                    processed_distances = self.cognitive_perception_module.get_processed_radar_distances()
                    if processed_distances is not None and len(processed_distances) > 0:
                        radar_data_found = True
                        beam_count = len(processed_distances)
                        data_source = "cognitive perception module"

                        total_beams = len(processed_distances)
                        angle_per_beam = 360.0 / total_beams
                        front_beam_index = 0
                        beam_range = int(detection_angle_degrees / angle_per_beam)

                        start_index = (front_beam_index - beam_range) % total_beams
                        end_index = (front_beam_index + beam_range + 1) % total_beams

                        logger.debug(f"cognitive perception module radar configuration: total beams={total_beams}, beams per degree={total_beams/360:.1f}, "
                                    f"forward beam range=[{start_index}:{end_index}], beam count={beam_range*2+1}")

                        if start_index < end_index:
                            front_radar_data = processed_distances[start_index:end_index]
                        else:
                            front_radar_data = np.concatenate([
                                processed_distances[start_index:],
                                processed_distances[:end_index]
                            ])

                        valid_distances = front_radar_data[
                            (front_radar_data > 0) &
                            (front_radar_data < max_detection_distance) &
                            (front_radar_data != float('inf')) &
                            (~np.isnan(front_radar_data))
                        ]

                        valid_beams = len(valid_distances)

                        if len(valid_distances) > 0:
                            min_distance = float(np.min(valid_distances))

                            logger.debug(f"cognitive perception module radar data available: total beams={total_beams}, valid distance count={valid_beams}, "
                                        f"minimum distance={min_distance:.2f}m")
                        else:
                            logger.debug(f"cognitive perception module radar data contains no valid distances: total beams={total_beams}, checked distance count={len(front_radar_data)}")
                    else:
                        logger.debug("the cognitive perception module did not provide valid radar data")

                except Exception as e:
                    radar_errors.append(e)
                    self._warn_radar_once("perception_module_exception", f"Failed to read radar data from the cognitive perception module: {e}")

            if not radar_data_found:
                try:
                    obs = env.get_single_observation()
                    logger.debug(f"Fallback path-observation data type: {type(obs)}, keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")

                    if isinstance(obs, dict) and 'lidar' in obs:
                        lidar_obs = obs['lidar']
                        logger.debug(f"Fallback path-radar data type: {type(lidar_obs)}, shape: {lidar_obs.shape if hasattr(lidar_obs, 'shape') else 'N/A'}")
                        if isinstance(lidar_obs, np.ndarray) and len(lidar_obs) > 0:
                            radar_data_found = True
                            beam_count = len(lidar_obs)
                            data_source = "environment observation-dict"
                            front_radar_data = lidar_obs

                    elif hasattr(obs, 'cloud_points'):
                        logger.debug(f"LidarStateObservation attribute check: cloud_points={obs.cloud_points}, type={type(obs.cloud_points) if obs.cloud_points is not None else 'None'}")
                        if hasattr(obs, 'lidar_observe') and hasattr(env, 'agent'):
                            try:
                                lidar_data = obs.lidar_observe(env.agent)
                                logger.debug(f"Attempting lidar_observe: data type={type(lidar_data)}, length={len(lidar_data) if lidar_data else 0}")
                                if lidar_data and len(lidar_data) > 0:
                                    lidar_obs = np.array(lidar_data)
                                    logger.debug(f"Data returned by lidar_observe: shape={lidar_obs.shape}")
                                    radar_data_found = True
                                    beam_count = len(lidar_obs)
                                    data_source = "environment observation-LidarObserve"
                                    front_radar_data = lidar_obs
                            except Exception as e:
                                radar_errors.append(e)
                                self._warn_radar_once("lidar_observe_exception", f"lidar_observe call failed: {e}")

                        if not radar_data_found and obs.cloud_points is not None:
                            cloud_points = obs.cloud_points
                            if isinstance(cloud_points, (list, np.ndarray)) and len(cloud_points) > 0:
                                lidar_obs = np.array(cloud_points)
                                logger.debug(f"Fallback path-LidarStateObservation radar data: shape={lidar_obs.shape}, type={type(lidar_obs)}")
                                radar_data_found = True
                                beam_count = len(lidar_obs)
                                data_source = "environment observation-LidarState"
                                front_radar_data = lidar_obs
                            else:
                                logger.debug(f"LidarStateObservation.cloud_points is empty or invalid: {cloud_points}")

                        if not radar_data_found:
                            logger.debug(f"LidarStateObservation debug information:")
                            logger.debug(f"    hasattr cloud_points: {hasattr(obs, 'cloud_points')}")
                            logger.debug(f"    hasattr lidar_observe: {hasattr(obs, 'lidar_observe')}")
                            logger.debug(f"    hasattr state_obs: {hasattr(obs, 'state_obs')}")
                            logger.debug(f"    dir(obs): {[attr for attr in dir(obs) if not attr.startswith('_')]}")

                    if radar_data_found:
                        total_beams = len(front_radar_data)
                        angle_per_beam = 360.0 / total_beams
                        front_beam_index = 0
                        beam_range = int(detection_angle_degrees / angle_per_beam)

                        start_index = (front_beam_index - beam_range) % total_beams
                        end_index = (front_beam_index + beam_range + 1) % total_beams

                        logger.debug(f"environment observation radar configuration: total beams={total_beams}, beams per degree={total_beams/360:.1f}, "
                                    f"forward beam range=[{start_index}:{end_index}]")

                        if start_index < end_index:
                            front_radar_data = front_radar_data[start_index:end_index]
                        else:
                            front_radar_data = np.concatenate([
                                front_radar_data[start_index:],
                                front_radar_data[:end_index]
                            ])

                        if np.max(front_radar_data) <= 1.0:
                            front_radar_data = front_radar_data * self.radians_detect_dis
                            logger.debug(f"Normalized data detected and converted to distance values in meters")

                        valid_distances = front_radar_data[
                            (front_radar_data > 0) &
                            (front_radar_data < max_detection_distance) &
                            (front_radar_data != float('inf')) &
                            (~np.isnan(front_radar_data))
                        ]

                        valid_beams = len(valid_distances)

                        if len(valid_distances) > 0:
                            min_distance = float(np.min(valid_distances))

                            logger.debug(f"environment observation radar data available: total beams={total_beams}, valid distance count={valid_beams}, "
                                        f"minimum distance={min_distance:.2f}m, data source={data_source}")
                    else:
                        logger.debug("Unable to extract radar information from the observation data")

                except Exception as e:
                    radar_errors.append(e)
                    self._warn_radar_once("observation_fallback_exception", f"Fallback path failed while reading radar data from environment observations: {e}")

            if radar_data_found and min_distance < float('inf'):
                relative_speed = self._calculate_relative_speed_from_radar(min_distance, data_source)
                speed_source = "radar distance change rate"

                if relative_speed is None:
                    detection_info = {
                        'method': 'radar_beam',
                        'status': 'insufficient_history',
                        'data_source': data_source,
                        'total_beams': beam_count,
                        'valid_beams': valid_beams,
                        'min_distance': min_distance,
                        'agent_speed': agent_speed,
                        'relative_speed': 0.0,
                        'speed_source': 'insufficient_history',
                        'tta': float('inf'),
                        'basic_inverse_tta': 0.0,
                        'final_inverse_tta': 0.0,
                        'visual_aversion_factor': 1.0,
                        'detection_angle': detection_angle_degrees
                    }
                    self.detection_history.append(detection_info)
                    self.distance_history.append(min_distance)
                    self.threat_count_history.append(1)
                    logger.debug(
                        f"Radar TTA waiting for the next frame [data source: {data_source}]: "
                        f"recorded first-frame distance={min_distance:.2f}m"
                    )
                    return None

                logger.debug(f"Using relative speed: {relative_speed:.2f}m/s (source: {speed_source})")

                if min_distance < 0.01:
                    min_distance = 0.01

                if relative_speed == 0.0:
                    tta = float('inf')
                    basic_inverse_tta = 0.0
                else:
                    tta = min_distance / relative_speed
                    basic_inverse_tta = relative_speed / min_distance

                inverse_tta = basic_inverse_tta

                detection_info = {
                    'method': 'radar_beam',
                    'status': 'ready',
                    'data_source': data_source,
                    'total_beams': beam_count,
                    'valid_beams': valid_beams,
                    'min_distance': min_distance,
                    'agent_speed': agent_speed,
                    'relative_speed': relative_speed,
                    'speed_source': speed_source,
                    'tta': tta,
                    'basic_inverse_tta': basic_inverse_tta,
                    'final_inverse_tta': inverse_tta,
                    'visual_aversion_factor': 1.0,
                    'detection_angle': detection_angle_degrees
                }
                self.detection_history.append(detection_info)
                self.distance_history.append(min_distance)
                self.threat_count_history.append(1 if min_distance < float('inf') else 0)

                logger.debug(f"Radar TTA computed successfully [data source: {data_source}]: minimum distance={min_distance:.2f}m, "
                            f"relative speed={relative_speed:.2f}m/s ({speed_source}), "
                            f"TTA={tta:.2f}s, inverse_tta={inverse_tta:.3f}")

                return inverse_tta

            if not radar_data_found:
                if radar_errors:
                    raise RuntimeError(
                        "Radar data unavailable from all configured sources"
                    ) from radar_errors[-1]
                raise RuntimeError("Radar data unavailable from all configured sources")

            detection_info = {
                'method': 'radar_beam',
                'status': 'no_threat',
                'data_source': data_source,
                'total_beams': beam_count,
                'valid_beams': valid_beams,
                'min_distance': float('inf'),
                'agent_speed': agent_speed,
                'relative_speed': 0.0,
                'speed_source': 'N/A',
                'tta': float('inf'),
                'basic_inverse_tta': 0.0,
                'final_inverse_tta': 0.0,
                'visual_aversion_factor': 1.0,
                'detection_angle': detection_angle_degrees
            }
            self.detection_history.append(detection_info)
            self.distance_history.append(float('inf'))
            self.threat_count_history.append(0)

            logger.debug(f"Radar TTA found no threat [data source: {data_source}]: speed={agent_speed:.2f}m/s, "
                        f"radar readout=success, "
                        f"total beam count={beam_count}, valid beam count={valid_beams}")

            return 0.0

        except Exception:
            logger.exception("Radar TTA computation failed")
            raise

    def process_reward(self, original_reward: float, env=None, info: Optional[Dict] = None,
        is_ppo_mode: bool = False) -> Tuple[float, Dict[str, Any]]:
        """Apply a tanh-scaled looming penalty above the inverse-TTA threshold.

        Reward shaping is active only in PPO mode; otherwise the original reward
        and unchanged bias metadata are returned.
        """

        bias_info = {
            'original_reward': original_reward,
            'bias_applied': 0.0,
            'inverse_tta': None,
            'bias_active': False
        }

        if not is_ppo_mode:
            return original_reward, bias_info

        inverse_tta = self.get_inverse_tta(env, info)
        bias_info['inverse_tta'] = inverse_tta
        if inverse_tta is None:
            self.tta_history.append(0.0)
            self.bias_history.append(0.0)
            self.reward_history.append(original_reward)
            return original_reward, bias_info

        self.tta_history.append(inverse_tta)

        if inverse_tta > self.tta_threshold:
            looming_penalty = self.inverse_tta_coef * np.tanh(inverse_tta)
            adjusted_reward = original_reward - looming_penalty

            self.bias_history.append(looming_penalty)
            bias_info['bias_applied'] = looming_penalty
            bias_info['bias_active'] = True
            self._total_bias += looming_penalty
            self._active_steps += 1

        else:
            adjusted_reward = original_reward
            self.bias_history.append(0.0)

        self.reward_history.append(original_reward)

        self._step_count += 1

        if self.verbose and self._step_count % 50 == 0:
            avg_bias = np.mean(list(self.bias_history)) if self.bias_history else 0.0
            avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0.0
            avg_tta = np.mean(list(self.tta_history)) if self.tta_history else 0.0
            logger.info(f"[CognitiveBias] Step {self._step_count}: "
                       f"avg_bias={avg_bias:.4f}, avg_reward={avg_reward:.4f}, "
                       f"avg_tta={avg_tta:.4f}, active_rate={self._active_steps/self._step_count:.2%}")

        bias_info['adjusted_reward'] = adjusted_reward
        return adjusted_reward, bias_info


    def reset(self):
        self.bias_history.clear()
        self.reward_history.clear()
        self.tta_history.clear()
        self.detection_history.clear()
        self.distance_history.clear()
        self.threat_count_history.clear()

        self._step_count = 0
        self._total_bias = 0.0
        self._active_steps = 0

        logger.debug("CognitiveBiasModule reset complete")

    def detach_from_env(self):
        self.attached_env = None
        logger.debug("Cognitive bias module detached from the environment")

    def get_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_steps': self._step_count,
            'active_steps': self._active_steps,
            'activation_rate': self._active_steps / max(1, self._step_count),
            'total_bias': self._total_bias,
            'average_bias': self._total_bias / max(1, self._step_count)
        }

        if self.bias_history:
            stats['recent_avg_bias'] = np.mean(list(self.bias_history))
            stats['recent_max_bias'] = np.max(list(self.bias_history))

        if self.reward_history:
            stats['recent_avg_reward'] = np.mean(list(self.reward_history))

        if self.tta_history:
            stats['recent_avg_tta'] = np.mean(list(self.tta_history))
            stats['recent_max_tta'] = np.max(list(self.tta_history))

        return stats


    def generate_visualization(self, env=None, save_dir: Optional[str] = None):
        if save_dir is None:
            save_dir = self._get_visualization_dir(env)

        os.makedirs(save_dir, exist_ok=True)

        self._plot_reward_comparison(save_dir)
        self._plot_detection_analysis(save_dir)
        self._plot_visual_aversion_details(save_dir)

        logger.info(f"Cognitive bias visualization finished; images saved to: {save_dir}")
        return save_dir

    def _get_visualization_dir(self, env=None) -> str:
        if env and hasattr(env, 'visualization_output_dir'):
            base_dir = env.visualization_output_dir
        else:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = f"./outputs/fig_cog/cognitive_analysis_{timestamp}"

        bias_dir = os.path.join(base_dir, "cognitive_bias")
        os.makedirs(bias_dir, exist_ok=True)
        return bias_dir

    def _plot_bias_history(self, save_dir: str):
        if not self.bias_history:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        bias_data = list(self.bias_history)
        ax1.plot(bias_data, 'b-', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Bias Amount')
        ax1.set_title('Cognitive Bias Over Time')
        ax1.grid(True, alpha=0.3)

        ax2.hist(bias_data, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax2.set_xlabel('Bias Amount')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Cognitive Bias')
        ax2.grid(True, alpha=0.3)

        mean_bias = np.mean(bias_data)
        std_bias = np.std(bias_data)
        ax2.axvline(mean_bias, color='red', linestyle='--',
                   label=f'Mean: {mean_bias:.3f}')
        ax2.axvline(mean_bias + std_bias, color='red', linestyle=':',
                   label=f'±1σ: {std_bias:.3f}')
        ax2.axvline(mean_bias - std_bias, color='red', linestyle=':')
        ax2.legend()

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'bias_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_tta_distribution(self, save_dir: str):
        if not self.tta_history:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        tta_data = list(self.tta_history)
        ax1.plot(tta_data, 'g-', alpha=0.7)
        ax1.axhline(self.tta_threshold, color='red', linestyle='--',
                   label=f'Threshold: {self.tta_threshold}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Inverse TTA')
        ax1.set_title('Inverse TTA Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.hist(tta_data, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(self.tta_threshold, color='red', linestyle='--',
                   label=f'Threshold: {self.tta_threshold}')
        ax2.set_xlabel('Inverse TTA')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Inverse TTA')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'tta_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


    def _plot_reward_comparison(self, save_dir: str):
        if not self.reward_history or not self.bias_history:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        original_rewards = list(self.reward_history)
        biases = list(self.bias_history)
        adjusted_rewards = [r - b for r, b in zip(original_rewards, biases)]

        steps = range(len(original_rewards))

        ax.plot(steps, original_rewards, 'b-', alpha=0.7, label='Original Reward')
        ax.plot(steps, adjusted_rewards, 'r-', alpha=0.7, label='Adjusted Reward')

        ax.fill_between(steps, original_rewards, adjusted_rewards,
                        alpha=0.3, color='yellow', label='Bias Effect')

        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Comparison: Original vs Cognitive Bias Adjusted')
        ax.legend()
        ax.grid(True, alpha=0.3)

        textstr = f'Total Bias: {self._total_bias:.2f}\n'
        textstr += f'Activation Rate: {self._active_steps/max(1, self._step_count):.2%}\n'
        textstr += f'Avg Original: {np.mean(original_rewards):.3f}\n'
        textstr += f'Avg Adjusted: {np.mean(adjusted_rewards):.3f}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'reward_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_detection_analysis(self, save_dir: str):
        if not self.detection_history:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        steps = range(len(self.detection_history))
        threat_counts = list(self.threat_count_history)
        distances = [d if d != float('inf') else None for d in self.distance_history]
        detection_data = list(self.detection_history)

        ax1.plot(steps, threat_counts, 'r-', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Threat Count')
        ax1.set_title('Detected Threat Objects Over Time')
        ax1.grid(True, alpha=0.3)

        valid_distances = [d for d in distances if d is not None]
        valid_steps = [s for s, d in zip(steps, distances) if d is not None]
        if valid_distances:
            ax2.plot(valid_steps, valid_distances, 'b-', alpha=0.7, linewidth=1.5)
            ax2.axhline(self.visual_detection_distance, color='red', linestyle='--',
                       label=f'Detection Range: {self.visual_detection_distance}m')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Distance (m)')
            ax2.set_title('Closest Threat Distance Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        if threat_counts:
            unique_counts = sorted(set(threat_counts))
            count_freq = [threat_counts.count(c) for c in unique_counts]
            ax3.bar(unique_counts, count_freq, alpha=0.7, color='orange')
            ax3.set_xlabel('Number of Threats')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Distribution of Threat Counts')
            ax3.grid(True, alpha=0.3, axis='y')

        if valid_distances:
            ax4.hist(valid_distances, bins=20, alpha=0.7, color='green', edgecolor='black')
            ax4.axvline(self.visual_detection_distance, color='red', linestyle='--',
                       label=f'Detection Range: {self.visual_detection_distance}m')
            ax4.set_xlabel('Distance (m)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Threat Distances')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'detection_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_visual_aversion_details(self, save_dir: str):
        if not self.detection_history:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        steps = range(len(self.detection_history))
        detection_data = list(self.detection_history)

        basic_ttas = [d.get('basic_inverse_tta', 0.0) for d in detection_data]
        final_ttas = [d.get('final_inverse_tta', 0.0) for d in detection_data]
        aversion_factors = [d.get('visual_aversion_factor', 1.0) for d in detection_data]
        agent_speeds = [d.get('agent_speed', 0.0) for d in detection_data]

        ax1.plot(steps, basic_ttas, 'b-', alpha=0.7, label='Basic Inverse TTA', linewidth=1.5)
        ax1.plot(steps, final_ttas, 'r-', alpha=0.7, label='Final Inverse TTA (with aversion)', linewidth=1.5)
        ax1.axhline(self.tta_threshold, color='orange', linestyle='--',
                   label=f'Threshold: {self.tta_threshold}')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Inverse TTA')
        ax1.set_title('Basic vs Final Inverse TTA Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(steps, aversion_factors, 'g-', alpha=0.7, linewidth=1.5)
        ax2.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No Aversion (1.0)')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Visual Aversion Factor')
        ax2.set_title('Visual Aversion Enhancement Factor Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(steps, agent_speeds, 'm-', alpha=0.7, linewidth=1.5)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Speed (m/s)')
        ax3.set_title('Agent Speed Over Time')
        ax3.grid(True, alpha=0.3)

        enhancement_ratios = []
        for basic, final in zip(basic_ttas, final_ttas):
            if basic > 1e-6:
                enhancement_ratios.append(final / basic)
            else:
                enhancement_ratios.append(1.0)

        ax4.plot(steps, enhancement_ratios, 'orange', alpha=0.7, linewidth=1.5)
        ax4.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='No Enhancement')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Enhancement Ratio (Final/Basic)')
        ax4.set_title('Visual Aversion Enhancement Effect')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'visual_aversion_details.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_cognitive_bias_module(config: Optional[Dict[str, Any]] = None) -> CognitiveBiasModule:
    return CognitiveBiasModule(config)
