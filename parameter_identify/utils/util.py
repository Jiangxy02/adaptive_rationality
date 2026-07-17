#!/usr/bin/env python3
"""
Utility helpers.
"""


import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
try:
    import yaml
except ImportError:
    yaml = None
import pickle
import pandas as pd
import os
from datetime import datetime

from parameter_identify.utils.likelihood_calculator import TrajectoryPoint
from parameter_identify.utils.particle_manager import Particle


def json_default(value):
    """Encode supported scientific-domain values without losing structure."""
    if isinstance(value, TrajectoryPoint):
        return {
            'px': value.px,
            'py': value.py,
            'vx': value.vx,
            'vy': value.vy,
            'timestamp': value.timestamp,
            'yaw': value.yaw,
        }
    if isinstance(value, Particle):
        return {
            'theta': value.theta,
            'weight': value.weight,
            'log_weight': value.log_weight,
        }
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration file.
    """
    ext = os.path.splitext(config_path)[1].lower()

    with open(config_path, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError("PyYAML is not available in the locked paper environment; use JSON instead")
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save a configuration file.
    """
    ext = os.path.splitext(save_path)[1].lower()

    with open(save_path, 'w') as f:
        if ext == '.json':
            json.dump(config, f, indent=2, default=json_default)
        elif ext in ['.yaml', '.yml']:
            if yaml is None:
                raise ImportError("PyYAML is not available in the locked paper environment; use JSON instead")
            yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")


def create_env_config(data_path: str,
                     scenario_name: str = 'trajectory',
                     enable_render: bool = False,
                     ppo_checkpoint_path: str = None,
                     use_cognitive_modules: bool = False,
                     **kwargs) -> Dict[str, Any]:
    """
    Create the environment configuration.
    """
    if ppo_checkpoint_path is None:
        raise ValueError("ppo_checkpoint_path must be provided explicitly")

    # Standard MetaDrive config containing only recognized keys.
    config = {
        'use_render': enable_render,
        'manual_control': False,
        'physics_world_step_size': 0.02,
        'decision_repeat': 5,
        'horizon': 1000,
        'vehicle_config': {
            'enable_reverse': False,
            'spawn_lane_index': None,
        },
        'traffic_density': 0.1,
        'map': 'SSSSSSSSSS',  # Use the standard map format.
        'start_seed': 0,
        'random_traffic': True,
        'random_spawn_lane_index': False,
        'random_agent_model': False,
    }

    # Additional custom config that is filtered before reaching MetaDrive.
    custom_config = {
        'scenario_name': scenario_name,
        'data_path': data_path,
        'ppo_checkpoint_path': ppo_checkpoint_path,
        'ppo_device': 'cpu',
        'use_cognitive_modules': use_cognitive_modules,
        'enable_background_vehicles': True,  # Enable background vehicles.
        'background_vehicle_update_mode': 'position',
        'enable_realtime': False,
        'target_fps': 50.0
    }

    # Merge directly; downstream env config handling filters custom fields.
    config.update(custom_config)

    # Apply any extra overrides.
    config.update(kwargs)

    return config


def compute_trajectory_metrics(trajectory: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute summary metrics for a trajectory.
    """
    if not trajectory:
        return {}

    # Convert to NumPy arrays.
    positions = np.array([[p['px'], p['py']] for p in trajectory])
    velocities = np.array([[p['vx'], p['vy']] for p in trajectory])

    # Compute metrics.
    metrics = {
        'total_distance': compute_path_length(positions),
        'mean_speed': np.mean(np.linalg.norm(velocities, axis=1)),
        'max_speed': np.max(np.linalg.norm(velocities, axis=1)),
        'mean_acceleration': compute_mean_acceleration(velocities),
        'path_smoothness': compute_path_smoothness(positions),
        'speed_smoothness': compute_speed_smoothness(velocities),
    }

    # Include yaw smoothness when yaw is available.
    if 'yaw' in trajectory[0] and trajectory[0]['yaw'] is not None:
        yaws = np.array([p['yaw'] for p in trajectory])
        metrics['yaw_smoothness'] = compute_yaw_smoothness(yaws)

    return metrics


def compute_path_length(positions: np.ndarray) -> float:
    """Compute path length."""
    if len(positions) < 2:
        return 0.0

    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)


def compute_mean_acceleration(velocities: np.ndarray, dt: float = 0.02) -> float:
    """Compute mean acceleration."""
    if len(velocities) < 2:
        return 0.0

    accelerations = np.diff(velocities, axis=0) / dt
    acc_magnitudes = np.linalg.norm(accelerations, axis=1)
    return np.mean(acc_magnitudes)


def compute_path_smoothness(positions: np.ndarray) -> float:
    """Compute path smoothness as an approximate curvature-change measure."""
    if len(positions) < 3:
        return 0.0

    # Compute first- and second-order differences.
    first_diff = np.diff(positions, axis=0)
    second_diff = np.diff(first_diff, axis=0)

    # Estimate curvature.
    curvatures = []
    for i in range(len(second_diff)):
        v1 = first_diff[i]
        v2 = second_diff[i]
        speed = np.linalg.norm(v1)
        if speed > 0.01:  # Avoid division by zero.
            curvature = np.linalg.norm(v2) / (speed ** 2)
            curvatures.append(curvature)

    if curvatures:
        return np.mean(curvatures)
    else:
        return 0.0


def compute_speed_smoothness(velocities: np.ndarray, dt: float = 0.02) -> float:
    """Compute speed smoothness."""
    if len(velocities) < 2:
        return 0.0

    speeds = np.linalg.norm(velocities, axis=1)
    speed_changes = np.abs(np.diff(speeds)) / dt
    return np.mean(speed_changes)


def compute_yaw_smoothness(yaws: np.ndarray, dt: float = 0.02) -> float:
    """Compute heading smoothness."""
    if len(yaws) < 2:
        return 0.0

    # Handle angle wrapping.
    yaw_diffs = np.diff(yaws)
    yaw_diffs = np.arctan2(np.sin(yaw_diffs), np.cos(yaw_diffs))

    yaw_rates = np.abs(yaw_diffs) / dt
    return np.mean(yaw_rates)


def synchronize_trajectories(traj1: List[Dict[str, float]],
                           traj2: List[Dict[str, float]],
                           method: str = 'truncate') -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Synchronize the lengths of two trajectories.
    """
    if method == 'truncate':
        # Truncate to the shorter length.
        min_len = min(len(traj1), len(traj2))
        return traj1[:min_len], traj2[:min_len]

    elif method == 'pad':
        # Pad to the longer length.
        if len(traj1) < len(traj2):
            # Pad traj1 using its last point.
            last_point = traj1[-1].copy()
            while len(traj1) < len(traj2):
                traj1.append(last_point.copy())
        else:
            # Pad traj2 using its last point.
            last_point = traj2[-1].copy()
            while len(traj2) < len(traj1):
                traj2.append(last_point.copy())
        return traj1, traj2

    elif method == 'interpolate':
        raise NotImplementedError("Interpolation-based synchronization is not implemented")

    else:
        raise ValueError(f"Unsupported synchronization method: {method}")


def generate_experiment_name() -> str:
    """Generate an experiment name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"param_identification_{timestamp}"


def create_summary_statistics(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary statistics table for identification results.
    """
    # Extract the final estimate.
    final_estimate = results['final_estimate']

    # Build summary rows.
    summary_data = []

    for param in final_estimate['mean'].keys():
        row = {
            'Parameter': param,
            'Mean': final_estimate['mean'][param],
            'Std': final_estimate['std'][param],
            'MAP': final_estimate['map'][param],
            'CI_Lower': final_estimate['ci_95'][param][0],
            'CI_Upper': final_estimate['ci_95'][param][1],
            'CI_Width': final_estimate['ci_95'][param][1] - final_estimate['ci_95'][param][0]
        }
        summary_data.append(row)

    # Create the DataFrame.
    df = pd.DataFrame(summary_data)

    # Store ESS metadata.
    df.attrs['ESS'] = final_estimate['ess']

    return df


def export_results(results: Dict[str, Any],
                  export_dir: str,
                  formats: List[str] = ['json', 'pkl', 'csv']):
    """
    Export results in multiple formats.
    """
    os.makedirs(export_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON.
    if 'json' in formats:
        json_path = os.path.join(export_dir, f'results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=json_default)
        print(f"Saved JSON results to: {json_path}")

    # Pickle.
    if 'pkl' in formats:
        pkl_path = os.path.join(export_dir, f'results_{timestamp}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved Pickle results to: {pkl_path}")

    # CSV summary.
    if 'csv' in formats:
        csv_path = os.path.join(export_dir, f'summary_{timestamp}.csv')
        df = create_summary_statistics(results)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV summary to: {csv_path}")


def validate_trajectory_data(trajectory: List[Dict[str, float]]) -> bool:
    """
    Validate trajectory data.
    """
    if not trajectory:
        print("Trajectory is empty")
        return False

    # Check required fields.
    required_fields = ['px', 'py', 'vx', 'vy']
    for i, point in enumerate(trajectory):
        for field in required_fields:
            if field not in point:
                print(f"Point {i} is missing field: {field}")
                return False
            if not isinstance(point[field], (int, float)):
                print(f"Point {i} has invalid type for field '{field}'")
                return False

    # Check numeric ranges.
    for i, point in enumerate(trajectory):
        speed = np.sqrt(point['vx']**2 + point['vy']**2)
        if speed > 50:  # 50 m/s = 180 km/h
            print(f"Point {i} has an unusually high speed: {speed:.2f} m/s")

    return True
