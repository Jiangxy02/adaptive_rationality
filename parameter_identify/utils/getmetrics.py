
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

import logging


logging.getLogger('cognitive_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_bias_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_perception_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_delay_module').setLevel(logging.WARNING)

from metadrive.obs import observation_base
from parameter_identify.utils.particle_manager import Particle
from parameter_identify.utils.likelihood_calculator import LikelihoodCalculator, TrajectoryPoint
from parameter_identify.utils.parameter_time_sync import ParameterIdentificationTimeSynchronizer

COMPLEX_SIM_AVAILABLE = True

class GetMetrics:
    def __init__(self,
                 sigma_diag,
                 use_geometric_mean_likelihood,
                 collision_info_csv_path: Optional[str] = None):

        self.likelihood_calculator = LikelihoodCalculator(
            sigma_diag=sigma_diag,
            use_yaw=False
        )

        self.time_synchronizer = ParameterIdentificationTimeSynchronizer(
            simulation_time_step=0.1
        )


        self.collision_info_dict = {}
        if collision_info_csv_path:
            self._load_collision_info(collision_info_csv_path)

    def _load_collision_info(self, csv_path: str):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                filename = str(row.get('filename', ''))
                if not filename:
                    continue


                import os
                base_filename = os.path.basename(filename)
                if base_filename.endswith('.csv'):
                    base_filename = base_filename[:-4]

                collision_detected = row.get('collision_detected', False)
                collision_step = row.get('collision_step', -1)
                traj_start_timestamp = row.get('traj_start_timestamp', None)

                if collision_detected and collision_step >= 0 and traj_start_timestamp is not None:


                    collision_time = float(traj_start_timestamp) + float(collision_step) * 0.1
                    self.collision_info_dict[base_filename] = {
                        'collision_time': collision_time,
                        'collision_step': int(collision_step),
                        'traj_start_timestamp': float(traj_start_timestamp)
                    }

            print(f"Loaded {len(self.collision_info_dict)} collision records")
        except Exception as e:
            print(f"Failed to load collision information: {e}")
            self.collision_info_dict = {}

    def _get_collision_time_for_file(self, filename: Optional[str]) -> Optional[float]:
        if not filename:
            return None

        import os
        base_filename = os.path.basename(filename)
        if base_filename.endswith('.csv'):
            base_filename = base_filename[:-4]

        collision_info = self.collision_info_dict.get(base_filename)
        if collision_info:
            return collision_info['collision_time']
        return None

    def _truncate_trajectory_before_collision(
        self,
        trajectory: List[TrajectoryPoint],
        collision_time: float
    ) -> List[TrajectoryPoint]:
        if not trajectory:
            return trajectory

        truncated = []
        for point in trajectory:
            if point.timestamp < collision_time:
                truncated.append(point)
            else:
                break

        return truncated


    def _get_original_trajectory_segment(self,
                                       observed_trajectory: List[TrajectoryPoint],
                                       start_timestamp: float,
                                       num_steps: int) -> Optional[List[TrajectoryPoint]]:


        start_idx = None
        for i, point in enumerate(observed_trajectory):
            if point.timestamp >= start_timestamp:
                start_idx = i
                break

        if start_idx is None:
            return None



        time_step = self.time_synchronizer.simulation_time_step
        end_timestamp = start_timestamp + num_steps * time_step

        future_segment = []
        for i in range(start_idx, len(observed_trajectory)):
            if observed_trajectory[i].timestamp <= end_timestamp:
                future_segment.append(observed_trajectory[i])
            else:
                break

        return future_segment if len(future_segment) > 0 else None


    def _compute_one_step_nll_from_trajectories(self,
                            particles: List[Particle],
                            predicted_trajectories: List[List[TrajectoryPoint]],
                            next_obs: TrajectoryPoint,
                            window_idx: int,
                            prediction_start_idx: int) -> float:

        try:
            if len(particles) != len(predicted_trajectories):
                raise ValueError(
                    f"window {window_idx}: particle and trajectory counts differ"
                )


            particle_predictions = []
            weights = []

            for i, (particle, pred_traj) in enumerate(zip(particles, predicted_trajectories)):
                if len(pred_traj) <= prediction_start_idx:
                    raise ValueError(
                        f"window {window_idx}: particle {i} has no prediction at "
                        f"future index {prediction_start_idx}"
                    )
                particle_predictions.append(pred_traj[prediction_start_idx])
                weights.append(particle.weight)


            weights = np.asarray(weights, dtype=float)
            if not np.isfinite(weights).all() or np.any(weights < 0.0):
                raise ValueError(f"window {window_idx}: particle weights are invalid")
            weight_sum = np.sum(weights)
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                raise ValueError(f"window {window_idx}: particle weights sum to zero")
            weights = weights / weight_sum


            observation_vector = next_obs.to_array(self.likelihood_calculator.use_yaw)


            log_prob_components = []
            for pred_point, weight in zip(particle_predictions, weights):
                if weight <= 0:
                    continue


                pred_vector = pred_point.to_array(self.likelihood_calculator.use_yaw)


                error = observation_vector - pred_vector


                # log N(y; μ, Σ) = -0.5 * (error^T * Σ^{-1} * error + log|2πΣ|)
                mahalanobis_dist = error.T @ self.likelihood_calculator.inv_sigma @ error
                log_det_2pi_sigma = len(error) * np.log(2 * np.pi) + np.sum(np.log(self.likelihood_calculator.sigma_diag ** 2))

                log_gaussian = -0.5 * (mahalanobis_dist + log_det_2pi_sigma)
                log_weighted_component = np.log(weight) + log_gaussian

                log_prob_components.append(log_weighted_component)


            if len(log_prob_components) == 0:
                raise ValueError(f"window {window_idx}: no positive particle weight")


            max_log_prob = np.max(log_prob_components)
            log_sum_exp = max_log_prob + np.log(np.sum(np.exp(np.array(log_prob_components) - max_log_prob)))


            nll = -log_sum_exp

            if not np.isfinite(nll):
                raise ValueError(f"window {window_idx}: one-step NLL is not finite")

            return float(nll)

        except Exception as exc:
            raise RuntimeError(
                f"window {window_idx} one-step NLL calculation failed"
            ) from exc

    def _compute_h_step_rmse_from_trajectories(self,
                           particles: List[Particle],
                           predicted_trajectories: List[List[TrajectoryPoint]],
                           future_observations: List[TrajectoryPoint],
                           h_steps: int,
                           window_idx: int,
                           prediction_start_idx: int) -> Dict[str, float]:

        try:
            if len(particles) != len(predicted_trajectories):
                raise ValueError(
                    f"window {window_idx}: particle and trajectory counts differ"
                )


            best_particle_idx = np.argmax([p.weight for p in particles])
            predicted_trajectory = predicted_trajectories[best_particle_idx]

            if len(predicted_trajectory) == 0:
                raise ValueError(f"window {window_idx}: MAP trajectory is empty")



            available_pred_steps = max(
                0, len(predicted_trajectory) - prediction_start_idx
            )
            available_obs_steps = len(future_observations)
            min_length = min(available_pred_steps, available_obs_steps, h_steps)

            if min_length == 0:
                raise ValueError(
                    f"window {window_idx}: no aligned future points for RMSE"
                )


            future_prediction = predicted_trajectory[
                prediction_start_idx:prediction_start_idx + min_length
            ]
            pred_positions = np.array([(p.px, p.py) for p in future_prediction])
            obs_positions = np.array([(p.px, p.py) for p in future_observations[:min_length]])

            pred_velocities = np.array([(p.vx, p.vy) for p in future_prediction])
            obs_velocities = np.array([(p.vx, p.vy) for p in future_observations[:min_length]])


            pos_errors = pred_positions - obs_positions
            vel_errors = pred_velocities - obs_velocities


            rmse_px = np.sqrt(np.mean(pos_errors[:, 0] ** 2))
            rmse_py = np.sqrt(np.mean(pos_errors[:, 1] ** 2))
            rmse_vx = np.sqrt(np.mean(vel_errors[:, 0] ** 2))
            rmse_vy = np.sqrt(np.mean(vel_errors[:, 1] ** 2))


            rmse_position = np.sqrt(np.mean(np.sum(pos_errors ** 2, axis=1)))
            rmse_velocity = np.sqrt(np.mean(np.sum(vel_errors ** 2, axis=1)))


            rmse_weighted = np.sqrt(0.7 * rmse_position ** 2 + 0.3 * rmse_velocity ** 2)

            return {
                'rmse_position': float(rmse_position),
                'rmse_velocity': float(rmse_velocity),
                'rmse_px': float(rmse_px),
                'rmse_py': float(rmse_py),
                'rmse_vx': float(rmse_vx),
                'rmse_vy': float(rmse_vy),
                'rmse_weighted': float(rmse_weighted),
                'prediction_steps': min_length,
                'available_steps': len(future_observations),
                'predicted_steps': available_pred_steps
            }

        except Exception as exc:
            raise RuntimeError(
                f"window {window_idx} H-step RMSE calculation failed"
            ) from exc

    def _perform_evaluation_metrics(self,
                                  particles: List[Particle],
                                  predicted_trajectories: List[List[TrajectoryPoint]],
                                  obs_window: List[TrajectoryPoint],
                                  observed_trajectory: List[TrajectoryPoint],
                                  window_idx: int,
                                  total_windows: int):





        if len(particles) != len(predicted_trajectories):
            raise ValueError(
                f"window {window_idx}: particle and trajectory counts differ"
            )
        if not obs_window:
            raise ValueError(f"window {window_idx}: observation window is empty")

        current_timestamp = obs_window[-1].timestamp
        time_step = self.time_synchronizer.simulation_time_step
        prediction_start_idx = len(obs_window)


        next_timestamp = current_timestamp + time_step
        next_obs = next(
            (
                point
                for point in observed_trajectory
                if abs(point.timestamp - next_timestamp) < time_step / 2
            ),
            None,
        )
        nll_value = None
        if next_obs is not None:
            nll_value = self._compute_one_step_nll_from_trajectories(
                particles,
                predicted_trajectories,
                next_obs,
                window_idx,
                prediction_start_idx=prediction_start_idx,
            )
            print(
                f"Window {window_idx + 1}/{total_windows}: "
                f"1-step NLL = {nll_value:.4f}"
            )


        h_steps = 5
        future_end = current_timestamp + h_steps * time_step
        future_observations = [
            point
            for point in observed_trajectory
            if current_timestamp < point.timestamp <= future_end + 1e-12
        ]
        available_prediction_steps = min(
            (
                max(0, len(trajectory) - prediction_start_idx)
                for trajectory in predicted_trajectories
            ),
            default=0,
        )


        rmse_results = None
        if future_observations and available_prediction_steps > 0:
            rmse_results = self._compute_h_step_rmse_from_trajectories(
                particles,
                predicted_trajectories,
                future_observations,
                h_steps,
                window_idx,
                prediction_start_idx=prediction_start_idx,
            )
            print(
                f"Window {window_idx + 1}/{total_windows}: "
                f"{rmse_results['prediction_steps']}-step RMSE = "
                f"{rmse_results['rmse_position']:.4f}m (position), "
                f"{rmse_results['rmse_velocity']:.4f}m/s (velocity)"
            )


        nll_history = []
        rmse_history = []
        if nll_value is not None:
            nll_history.append({
                'window_idx': window_idx,
                'timestamp': current_timestamp,
                'nll': nll_value,
                'best_particle_weight': max(p.weight for p in particles),
            })

        if rmse_results is not None:
            rmse_data = rmse_results.copy()
            rmse_data.update({
                'window_idx': window_idx,
                'timestamp': current_timestamp,
                'h_steps': h_steps,
            })
            rmse_history.append(rmse_data)

        return nll_history, rmse_history


    def _compute_gt_log_likelihood_from_mc_stats(
        self,
        mc_stats: Optional[Dict],
        gt_trajectory: Optional[List[TrajectoryPoint]],
        min_std: float = 0.01
    ) -> Optional[Dict[str, Any]]:

        if not mc_stats or not isinstance(mc_stats, dict):
            return None

        if not gt_trajectory or len(gt_trajectory) == 0:
            return None

        mean_xy = mc_stats.get('mean_xy')
        std_xy = mc_stats.get('std_xy')

        if not mean_xy or not std_xy:
            return None

        try:

            mean_xy_arr = np.asarray(mean_xy, dtype=np.float32)
            std_xy_arr = np.asarray(std_xy, dtype=np.float32)

            if mean_xy_arr.size == 0 or std_xy_arr.size == 0:
                return None


            gt_x = np.array([p.px for p in gt_trajectory], dtype=np.float32)
            gt_y = np.array([p.py for p in gt_trajectory], dtype=np.float32)
            gt_times = np.array([p.timestamp for p in gt_trajectory], dtype=np.float64)


            time_step = mc_stats.get('time_step', 0.1)







            positions = mc_stats.get('positions')
            if positions and len(positions) > 0:

                first_sample = positions[0]
                if first_sample and len(first_sample) > 0:


                    T_pred = len(mean_xy_arr)
                    T_gt = len(gt_trajectory)



                    if T_pred > 0 and T_gt > 0:



                        T = min(T_pred, T_gt)


                        mean_x = mean_xy_arr[:T, 0]
                        mean_y = mean_xy_arr[:T, 1]
                        std_x = np.maximum(std_xy_arr[:T, 0], min_std)
                        std_y = np.maximum(std_xy_arr[:T, 1], min_std)
                        gt_x_aligned = gt_x[:T]
                        gt_y_aligned = gt_y[:T]
                    else:
                        return None
                else:
                    return None
            else:

                T_pred = len(mean_xy_arr)
                T_gt = len(gt_trajectory)
                T = min(T_pred, T_gt)

                if T == 0:
                    return None


                mean_x = mean_xy_arr[:T, 0]
                mean_y = mean_xy_arr[:T, 1]
                std_x = np.maximum(std_xy_arr[:T, 0], min_std)
                std_y = np.maximum(std_xy_arr[:T, 1], min_std)
                gt_x_aligned = gt_x[:T]
                gt_y_aligned = gt_y[:T]


            valid_mask = ~(np.isnan(mean_x) | np.isnan(mean_y) |
                          np.isnan(std_x) | np.isnan(std_y) |
                          np.isnan(gt_x_aligned) | np.isnan(gt_y_aligned))

            if not np.any(valid_mask):
                return None

            mean_x = mean_x[valid_mask]
            mean_y = mean_y[valid_mask]
            std_x = std_x[valid_mask]
            std_y = std_y[valid_mask]
            gt_x_aligned = gt_x_aligned[valid_mask]
            gt_y_aligned = gt_y_aligned[valid_mask]

            T_valid = len(mean_x)


            # log N(x; μ, σ) = -0.5 * log(2πσ²) - (x-μ)²/(2σ²)
            #                = -0.5 * [log(2π) + 2*log(σ) + (x-μ)²/σ²]
            log_2pi = np.log(2 * np.pi)


            log_prob_x = -0.5 * (log_2pi + 2 * np.log(std_x) + ((gt_x_aligned - mean_x) / std_x) ** 2)


            log_prob_y = -0.5 * (log_2pi + 2 * np.log(std_y) + ((gt_y_aligned - mean_y) / std_y) ** 2)


            log_prob_per_step = log_prob_x + log_prob_y


            total_log_likelihood = np.sum(log_prob_per_step)


            avg_log_likelihood = total_log_likelihood / T_valid if T_valid > 0 else 0.0


            nll = -total_log_likelihood
            avg_nll = -avg_log_likelihood

            return {
                'log_likelihood': float(total_log_likelihood),
                'avg_log_likelihood': float(avg_log_likelihood),
                'nll': float(nll),
                'avg_nll': float(avg_nll),
                'valid': True,
                'num_steps': int(T_valid),
                'num_pred_steps': int(T_pred),
                'num_gt_steps': int(T_gt)
            }
        except Exception as e:

            return None

    def _compute_ade_between_trajectories(
        self,
        traj1: List[TrajectoryPoint],
        traj2: List[TrajectoryPoint],
        max_time_diff: float = 0.2
    ) -> Optional[float]:

        if not traj1 or not traj2:
            return None


        traj1_times = np.array([p.timestamp for p in traj1], dtype=np.float64)
        traj2_times = np.array([p.timestamp for p in traj2], dtype=np.float64)


        aligned_traj1_xy = []
        aligned_traj2_xy = []

        for traj1_point in traj1:
            traj1_time = traj1_point.timestamp


            time_diffs = np.abs(traj2_times - traj1_time)
            closest_idx = np.argmin(time_diffs)
            min_time_diff = time_diffs[closest_idx]


            if min_time_diff > max_time_diff:
                continue


            traj2_point = traj2[closest_idx]

            aligned_traj1_xy.append([traj1_point.px, traj1_point.py])
            aligned_traj2_xy.append([traj2_point.px, traj2_point.py])

        if len(aligned_traj1_xy) == 0:
            return None


        traj1_xy = np.array(aligned_traj1_xy, dtype=np.float32)
        traj2_xy = np.array(aligned_traj2_xy, dtype=np.float32)
        dists = np.linalg.norm(traj1_xy - traj2_xy, axis=1)
        ade = float(np.mean(dists))

        return ade if np.isfinite(ade) else None

    def _align_trajectories_by_timestamp(
        self,
        pred_trajectory: List[TrajectoryPoint],
        gt_trajectory: List[TrajectoryPoint],
        max_time_diff: float = 0.2
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        if not pred_trajectory or not gt_trajectory:
            return None


        pred_times = np.array([p.timestamp for p in pred_trajectory], dtype=np.float64)
        gt_times = np.array([p.timestamp for p in gt_trajectory], dtype=np.float64)


        aligned_pred_xy = []
        aligned_gt_xy = []
        aligned_times = []

        for pred_point in pred_trajectory:
            pred_time = pred_point.timestamp


            time_diffs = np.abs(gt_times - pred_time)
            closest_idx = np.argmin(time_diffs)
            min_time_diff = time_diffs[closest_idx]


            if min_time_diff > max_time_diff:
                continue


            gt_point = gt_trajectory[closest_idx]

            aligned_pred_xy.append([pred_point.px, pred_point.py])
            aligned_gt_xy.append([gt_point.px, gt_point.py])
            aligned_times.append(pred_time)

        if len(aligned_pred_xy) == 0:
            return None

        return (np.array(aligned_pred_xy, dtype=np.float32),
                np.array(aligned_gt_xy, dtype=np.float32),
                np.array(aligned_times, dtype=np.float64))

    def compute_minade_minfde_missrate_at_k(
        self,
        predicted_trajectories: List[List[TrajectoryPoint]],
        gt_trajectory: Optional[List[TrajectoryPoint]],
        k: int,
        miss_epsilon: float,
        data_filename: Optional[str] = None,
        simulation_collision_info: Optional[Dict] = None,
        gt_likelihood_info: Optional[Dict] = None,
        mc_stats: Optional[Dict] = None
    ) -> Optional[Dict[str, float]]:

        if gt_trajectory is None or len(gt_trajectory) < 2:
            return None

        if not predicted_trajectories:
            return None

        k = max(1, int(k))
        miss_epsilon = float(miss_epsilon)


        gt_collision_time = None
        if data_filename:
            gt_collision_time = self._get_collision_time_for_file(data_filename)


        sim_collision_time = None
        if simulation_collision_info and simulation_collision_info.get('collision_detected', False):
            sim_collision_time = simulation_collision_info.get('collision_time')


        truncate_time = None
        if gt_collision_time is not None and sim_collision_time is not None:
            truncate_time = min(gt_collision_time, sim_collision_time)
        elif gt_collision_time is not None:
            truncate_time = gt_collision_time
        elif sim_collision_time is not None:
            truncate_time = sim_collision_time


        gt_trajectory_truncated = gt_trajectory
        if truncate_time is not None:
            gt_trajectory_truncated = self._truncate_trajectory_before_collision(gt_trajectory, truncate_time)
            if len(gt_trajectory_truncated) < 2:
                return None

        gt_future = gt_trajectory_truncated[1:]
        if len(gt_future) == 0:
            return None

        ade_list: List[float] = []
        fde_list: List[float] = []
        used = 0

        valid_predicted_trajectories: List[List[TrajectoryPoint]] = []

        for traj in predicted_trajectories:
            if used >= k:
                break
            if traj is None or len(traj) < 2:
                continue


            traj_truncated = traj
            if truncate_time is not None:
                traj_truncated = self._truncate_trajectory_before_collision(traj, truncate_time)
                if len(traj_truncated) < 2:
                    continue

            pred_future = traj_truncated[1:]
            if len(pred_future) == 0:
                continue


            aligned_result = self._align_trajectories_by_timestamp(pred_future, gt_future)
            if aligned_result is None:
                continue

            pred_xy, gt_xy, aligned_times = aligned_result

            if len(pred_xy) == 0:
                continue


            dists = np.linalg.norm(pred_xy - gt_xy, axis=1)
            ade = float(np.mean(dists))



            fde = float(dists[-1])


            if fde > 100.0:

                if len(aligned_times) > 0:
                    last_pred_time = pred_future[-1].timestamp if pred_future else 0.0
                    last_aligned_time = aligned_times[-1]
                    time_diff = abs(last_pred_time - last_aligned_time)


                    if time_diff > 0.5:

                        gt_times = np.array([p.timestamp for p in gt_future], dtype=np.float64)
                        time_diffs = np.abs(gt_times - last_pred_time)
                        if len(time_diffs) > 0 and np.min(time_diffs) <= 0.2:
                            closest_gt_idx = np.argmin(time_diffs)
                            closest_gt_point = gt_future[closest_gt_idx]

                            last_pred_point = pred_future[-1]
                            fde = float(np.linalg.norm([
                                last_pred_point.px - closest_gt_point.px,
                                last_pred_point.py - closest_gt_point.py
                            ]))

            if np.isfinite(ade) and np.isfinite(fde):
                ade_list.append(ade)
                fde_list.append(fde)

                valid_predicted_trajectories.append(pred_future)
                used += 1

        if not ade_list:
            return None

        min_ade = float(np.min(ade_list))
        min_fde = float(np.min(fde_list))
        miss = 1.0 if (min_fde > miss_epsilon) else 0.0



        spread_pair = None
        K = len(valid_predicted_trajectories)
        if K >= 2:

            pair_ade_sum = 0.0
            pair_count = 0

            for i in range(K):
                for j in range(i + 1, K):
                    traj_i = valid_predicted_trajectories[i]
                    traj_j = valid_predicted_trajectories[j]


                    ade_ij = self._compute_ade_between_trajectories(traj_i, traj_j)
                    if ade_ij is not None and np.isfinite(ade_ij):
                        pair_ade_sum += ade_ij
                        pair_count += 1



            if pair_count > 0:

                expected_pairs = K * (K - 1) / 2
                if pair_count == expected_pairs:

                    spread_pair = pair_ade_sum / (K * (K - 1))
                else:


                    spread_pair = pair_ade_sum / pair_count
            else:
                spread_pair = float('nan')
        else:

            spread_pair = float('nan')

        result = {
            'minADE': min_ade,
            'minFDE': min_fde,
            'miss': miss,
            'k': float(k),
            'k_effective': float(len(ade_list)),
            'epsilon': float(miss_epsilon),
            'gt_steps': float(len(gt_future))
        }


        if spread_pair is not None:
            result['spread_pair'] = float(spread_pair) if np.isfinite(spread_pair) else float('nan')



        if gt_likelihood_info and isinstance(gt_likelihood_info, dict) and gt_likelihood_info.get('valid'):
            result['log_likelihood'] = float(gt_likelihood_info.get('log_likelihood', float('nan')))
            result['avg_log_likelihood'] = float(gt_likelihood_info.get('avg_log_likelihood', float('nan')))
            result['nll'] = float(gt_likelihood_info.get('nll', float('nan')))
            result['avg_nll'] = float(gt_likelihood_info.get('avg_nll', float('nan')))
        else:


            computed_likelihood = self._compute_gt_log_likelihood_from_mc_stats(
                mc_stats, gt_future
            )
            if computed_likelihood and computed_likelihood.get('valid'):
                result['log_likelihood'] = float(computed_likelihood.get('log_likelihood', float('nan')))
                result['avg_log_likelihood'] = float(computed_likelihood.get('avg_log_likelihood', float('nan')))
                result['nll'] = float(computed_likelihood.get('nll', float('nan')))
                result['avg_nll'] = float(computed_likelihood.get('avg_nll', float('nan')))
            else:
                result['log_likelihood'] = float('nan')
                result['avg_log_likelihood'] = float('nan')
                result['nll'] = float('nan')
                result['avg_nll'] = float('nan')


        total_variance = float('nan')
        mean_variance = float('nan')
        if mc_stats and isinstance(mc_stats, dict):
            positions_raw = mc_stats.get('positions')
            if positions_raw:
                try:

                    positions = np.asarray(positions_raw, dtype=np.float32)

                    if positions.size > 0 and len(positions.shape) == 3:
                        N_samples, T, _ = positions.shape
                        if N_samples >= 2 and T > 0:
                            variances = []
                            for t in range(T):

                                pos_t = positions[:, t, :]  # (N_samples, 2)


                                valid_mask = ~(np.isnan(pos_t[:, 0]) | np.isnan(pos_t[:, 1]))
                                pos_t_valid = pos_t[valid_mask]

                                if len(pos_t_valid) >= 2:


                                    cov_matrix = np.cov(pos_t_valid.T)  # (2, 2)


                                    variance_t = np.trace(cov_matrix)

                                    if np.isfinite(variance_t) and variance_t >= 0:
                                        variances.append(float(variance_t))

                            if variances:

                                mean_variance = float(np.mean(variances))

                                total_variance = mean_variance
                except Exception as e:

                    pass

        result['total_variance'] = total_variance
        result['mean_variance'] = mean_variance

        return result
