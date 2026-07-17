#!/usr/bin/env python3


import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


def validate_sigma_value(value: float) -> float:
    """Return one strictly positive, finite observation standard deviation."""
    try:
        sigma = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"observation sigma must be a positive finite number, got {value!r}"
        ) from exc
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            f"observation sigma must be a positive finite number, got {value!r}"
        )
    return sigma


def validate_sigma_diag(values, expected_size: Optional[int] = None) -> np.ndarray:
    """Validate the full observation-noise diagonal before matrix inversion."""
    try:
        sigma_diag = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("sigma_diag must contain positive finite numbers") from exc
    if sigma_diag.ndim != 1:
        raise ValueError("sigma_diag must be a one-dimensional sequence")
    if expected_size is not None and sigma_diag.size != expected_size:
        raise ValueError(
            f"sigma_diag must contain {expected_size} values, got {sigma_diag.size}"
        )
    if not np.isfinite(sigma_diag).all() or np.any(sigma_diag <= 0.0):
        raise ValueError(
            f"sigma_diag must contain only positive finite values, got {values!r}"
        )
    return sigma_diag


@dataclass
class TrajectoryPoint:
    px: float
    py: float
    vx: float
    vy: float
    timestamp: float
    yaw: Optional[float] = None

    def to_array(self, include_yaw: bool = False) -> np.ndarray:
        if include_yaw and self.yaw is not None:
            return np.array([self.px, self.py, self.vx, self.vy, self.yaw])
        else:
            return np.array([self.px, self.py, self.vx, self.vy])


class LikelihoodCalculator:

    def __init__(self,
                 sigma_diag: Optional[List[float]] = None,
                 use_yaw: bool = False,
                 robust_loss: Optional[str] = None,
                 robust_loss_param: float = 1.0):

        self.use_yaw = use_yaw


        if sigma_diag is None:
            if use_yaw:

                sigma_diag = [0.1, 0.1, 0.5, 0.5, 0.1]  # [m, m, m/s, m/s, rad]
            else:
                sigma_diag = [0.1, 0.1, 0.5, 0.5]  # [m, m, m/s, m/s]

        expected_size = 5 if use_yaw else 4
        self.sigma_diag = validate_sigma_diag(sigma_diag, expected_size=expected_size)
        self.inv_sigma = np.diag(1.0 / (self.sigma_diag ** 2))

        self.robust_loss = robust_loss
        self.robust_loss_param = robust_loss_param

    def compute_trajectory_likelihood(self,
                                    observed_traj: List[TrajectoryPoint],
                                    predicted_traj: List[TrajectoryPoint],
                                    use_geometric_mean: bool = False) -> Dict[str, float]:

        if len(observed_traj) != len(predicted_traj):
            raise ValueError(f"Trajectory length mismatch: observed {len(observed_traj)} vs predicted {len(predicted_traj)}")


        frame_errors = []
        frame_nlls = []

        for obs_point, pred_point in zip(observed_traj, predicted_traj):

            error = self._compute_error_vector(obs_point, pred_point)
            frame_errors.append(error)


            if self.robust_loss is None:

                frame_nll = 0.5 * error.T @ self.inv_sigma @ error
            else:

                frame_nll = self._compute_robust_loss(error)

            frame_nlls.append(frame_nll)


        if use_geometric_mean:


            total_nll = np.mean(frame_nlls)
            mean_nll = total_nll
        else:

            total_nll = np.sum(frame_nlls)
            mean_nll = np.mean(frame_nlls)


        errors_array = np.array(frame_errors)
        rmse_per_dim = np.sqrt(np.mean(errors_array ** 2, axis=0))


        results = {
            'nll': total_nll,
            'log_likelihood': -total_nll,
            'mean_nll_per_frame': mean_nll,
            'rmse_px': rmse_per_dim[0],
            'rmse_py': rmse_per_dim[1],
            'rmse_vx': rmse_per_dim[2],
            'rmse_vy': rmse_per_dim[3],
            'rmse_total_position': np.sqrt(rmse_per_dim[0]**2 + rmse_per_dim[1]**2),
            'rmse_total_velocity': np.sqrt(rmse_per_dim[2]**2 + rmse_per_dim[3]**2),
            'frame_nlls': frame_nlls,
            'frame_errors': frame_errors
        }

        if self.use_yaw and len(rmse_per_dim) > 4:
            results['rmse_yaw'] = rmse_per_dim[4]

        return results

    def compute_windowed_likelihood(self,
                                  observed_traj: List[TrajectoryPoint],
                                  predicted_traj: List[TrajectoryPoint],
                                  window_size: int,
                                  stride: int = 1) -> List[Dict[str, float]]:

        window_results = []

        for start_idx in range(0, len(observed_traj) - window_size + 1, stride):
            end_idx = start_idx + window_size


            obs_window = observed_traj[start_idx:end_idx]
            pred_window = predicted_traj[start_idx:end_idx]


            window_result = self.compute_trajectory_likelihood(obs_window, pred_window)
            window_result['window_start'] = start_idx
            window_result['window_end'] = end_idx

            window_results.append(window_result)

        return window_results

    def _compute_error_vector(self, obs: TrajectoryPoint, pred: TrajectoryPoint) -> np.ndarray:
        error = obs.to_array(self.use_yaw) - pred.to_array(self.use_yaw)


        if self.use_yaw and len(error) > 4:

            error[4] = np.arctan2(np.sin(error[4]), np.cos(error[4]))

        return error

    def _compute_robust_loss(self, error: np.ndarray) -> float:

        normalized_error = error / self.sigma_diag

        if self.robust_loss == 'huber':

            return self._huber_loss(normalized_error)
        elif self.robust_loss == 'tukey':

            return self._tukey_loss(normalized_error)
        else:
            raise ValueError(f"Unsupported robust loss type: {self.robust_loss}")

    def _huber_loss(self, z: np.ndarray) -> float:
        delta = self.robust_loss_param
        abs_z = np.abs(z)


        quadratic = 0.5 * z ** 2
        linear = delta * (abs_z - 0.5 * delta)


        losses = np.where(abs_z <= delta, quadratic, linear)

        return np.sum(losses)

    def _tukey_loss(self, z: np.ndarray) -> float:
        c = self.robust_loss_param
        abs_z = np.abs(z)


        losses = np.where(
            abs_z <= c,
            (c ** 2 / 6) * (1 - (1 - (z / c) ** 2) ** 3),
            c ** 2 / 6
        )

        return np.sum(losses)

    def update_covariance(self, new_sigma_diag: List[float]):
        self.sigma_diag = np.array(new_sigma_diag)
        self.inv_sigma = np.diag(1.0 / (self.sigma_diag ** 2))

    def compute_likelihood_gradient(self,
                                  observed_traj: List[TrajectoryPoint],
                                  predicted_traj: List[TrajectoryPoint],
                                  param_gradients: Dict[str, List[TrajectoryPoint]]) -> Dict[str, float]:


        errors = []
        for obs, pred in zip(observed_traj, predicted_traj):
            errors.append(self._compute_error_vector(obs, pred))
        errors = np.array(errors)


        gradients = {}

        for param_name, grad_traj in param_gradients.items():

            grad_array = []
            for grad_point in grad_traj:
                grad_array.append(grad_point.to_array(self.use_yaw))
            grad_array = np.array(grad_array)


            grad_nll = 0.0
            for t in range(len(errors)):
                grad_nll -= errors[t] @ self.inv_sigma @ grad_array[t]

            gradients[param_name] = grad_nll

        return gradients

    @staticmethod
    def trajectory_from_dict_list(dict_list: List[Dict[str, float]]) -> List[TrajectoryPoint]:
        trajectory = []
        for point_dict in dict_list:
            trajectory.append(TrajectoryPoint(
                px=point_dict['px'],
                py=point_dict['py'],
                vx=point_dict['vx'],
                vy=point_dict['vy'],
                timestamp=point_dict.get('timestamp', 0.0),
                yaw=point_dict.get('yaw', None)
            ))
        return trajectory

    @staticmethod
    def trajectory_to_dict_list(trajectory: List[TrajectoryPoint]) -> List[Dict[str, float]]:
        dict_list = []
        for point in trajectory:
            point_dict = {
                'px': point.px,
                'py': point.py,
                'vx': point.vx,
                'vy': point.vy,
                'timestamp': point.timestamp
            }
            if point.yaw is not None:
                point_dict['yaw'] = point.yaw
            dict_list.append(point_dict)
        return dict_list
