#!/usr/bin/env python3


import numpy as np
# Use a non-interactive matplotlib backend to avoid tkinter conflicts in multiprocessing.
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot.
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from matplotlib.patches import Ellipse

from parameter_identify.utils.likelihood_calculator import TrajectoryPoint


class TrajectoryVisualizationManager:


    def __init__(self, output_dir: str):

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_prediction_summary_visualization(self,
                                              prediction_history: List[Dict],
                                              simulation_manager,
                                              pred_win: int) -> Optional[str]:

        if not prediction_history:
            print("No prediction history is available for visualization")
            return None

        try:
            # Set modern style and color scheme
            plt.style.use('default')
            fig, ax = plt.subplots(1, 1, figsize=(18, 14))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('#f8f9fa')

            # 1. Plot reference trajectory as background
            if hasattr(simulation_manager, 'original_data') and simulation_manager.original_data is not None:
                original_data = simulation_manager.original_data
                original_timestamps = simulation_manager.original_timestamps

                # Filter trajectory range to show only parameter identification relevant period
                if len(prediction_history) > 0:
                    # Get time range of all prediction moments
                    min_pred_time = min(pred['prediction_start_time'] for pred in prediction_history)
                    max_pred_time = max(pred['prediction_start_time'] for pred in prediction_history)

                    # Start from the first prediction moment and skip the leading
                    # window that has no corresponding prediction trajectory.
                    display_start = min_pred_time

                    # End time: last prediction moment plus the prediction horizon.
                    display_end = max_pred_time + pred_win * 0.1  # pred_win steps × 0.1s/step

                    # Filter original data to display range
                    mask = (original_timestamps >= display_start) & (original_timestamps <= display_end)
                    if mask.any():
                        filtered_data = original_data[mask]
                        original_px = filtered_data['position_x'].values
                        original_py = filtered_data['position_y'].values
                        label_text = f'Ground Truth (Prediction Period: {display_start:.1f}s - {display_end:.1f}s)'
                    else:
                        original_px = original_data['position_x'].values
                        original_py = original_data['position_y'].values
                        label_text = 'Ground Truth (Full)'

                    print(f"Parameter identification visualization time range: {display_start:.1f}s - {display_end:.1f}s")
                    print(f"   Ground truth trajectory points: {len(original_px)}")
                    print(f"   Starting from first prediction moment, excluding pre-prediction window_size period")

                else:
                    # If no prediction history, show full trajectory
                    original_px = original_data['position_x'].values
                    original_py = original_data['position_y'].values
                    label_text = 'Ground Truth Trajectory'

                ax.plot(original_px, original_py, color='#2c3e50', linewidth=3, alpha=0.8,
                       label=label_text, zorder=1, linestyle='-')

            # 2. Plot prediction trajectories and corresponding ground truth segments
            # Use a more sophisticated color palette
            colors = plt.cm.Set3(np.linspace(0.1, 0.9, len(prediction_history)))
            prediction_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

            # Add simplified legend markers.
            prediction_legend_added = False
            gt_segment_legend_added = False

            for i, pred_data in enumerate(prediction_history):
                color = colors[i]
                window_idx = pred_data['window_idx']
                start_time = pred_data['prediction_start_time']
                predicted_traj = pred_data['predicted_trajectory']
                original_segment = pred_data['original_trajectory_segment']
                marker = prediction_markers[i % len(prediction_markers)]

                # 2.1 Plot predicted trajectory (branching from identification moment)
                if predicted_traj and len(predicted_traj) > 0:
                    pred_x = [p.px for p in predicted_traj]
                    pred_y = [p.py for p in predicted_traj]

                    # Add the legend only for the first predicted trajectory.
                    label_pred = 'Predicted Trajectories' if not prediction_legend_added else None
                    if not prediction_legend_added:
                        prediction_legend_added = True

                    ax.plot(pred_x, pred_y, '--', color=color, linewidth=2.5, alpha=0.9,
                           label=label_pred, zorder=3)

                    # Mark prediction start point
                    ax.scatter(pred_x[0], pred_y[0], color=color, s=120, marker=marker,
                              edgecolor='white', linewidth=2, zorder=5, alpha=0.9)

                    # Mark prediction end point
                    ax.scatter(pred_x[-1], pred_y[-1], color=color, s=80, marker='s',
                              edgecolor='white', linewidth=2, zorder=5, alpha=0.7)

                # 2.2 Plot corresponding ground truth segment (if exists)
                if original_segment and len(original_segment) > 0:
                    orig_x = [p.px for p in original_segment]
                    orig_y = [p.py for p in original_segment]

                    # Add the legend only for the first ground-truth segment.
                    label_gt = 'Ground Truth Segments' if not gt_segment_legend_added else None
                    if not gt_segment_legend_added:
                        gt_segment_legend_added = True

                    ax.plot(orig_x, orig_y, '-', color=color, linewidth=3.5, alpha=0.7,
                           label=label_gt, zorder=2)

            # 3. Style and annotate the plot
            ax.set_xlabel('Position X (m)', fontsize=14, fontweight='bold', color='#2c3e50')
            ax.set_ylabel('Position Y (m)', fontsize=14, fontweight='bold', color='#2c3e50')

            # Set title for parameter identification period visualization
            title = (f'Parameter Identification & Prediction Visualization\n'
                    f'Prediction Window: {pred_win} steps | Identification Instances: {len(prediction_history)}')

            ax.set_title(title, fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

            # Enhanced grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#bdc3c7')
            ax.set_axisbelow(True)

            # Place a compact legend below the x-axis title.
            legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
                              ncol=3, fontsize=10, frameon=True,
                              fancybox=True, shadow=False,
                              facecolor='white', edgecolor='#bdc3c7', framealpha=0.9)
            legend.get_frame().set_linewidth(1)

            ax.set_aspect('equal', adjustable='box')

            plt.tight_layout()
            # Leave extra room so the legend fits below the x-axis title.
            plt.subplots_adjust(bottom=0.18)

            # 5. Save high-quality figure as vector PDF
            prediction_plot_file = os.path.join(self.output_dir, 'prediction_summary_visualization.pdf')
            plt.savefig(prediction_plot_file, bbox_inches='tight',
                       facecolor='white', edgecolor='none', format='pdf')
            plt.close()

            print(f"Prediction trajectory visualization saved to: {prediction_plot_file}")
            return prediction_plot_file

        except Exception as e:
            print(f"Failed to create prediction summary visualization: {str(e)}")
            return None

    def create_comprehensive_visualization(self,
                                         obs_window: List[TrajectoryPoint],
                                         predicted_trajectories: List[List[TrajectoryPoint]],
                                         log_likelihoods: np.ndarray,
                                         prediction_data: Dict[str, Any],
                                         window_idx: int,
                                         window_start: int,
                                         current_timestamp: float,
                                         if_collision: bool,
                                         trajectory_visualizer,
                                         background_vehicles_data: Optional[List[Dict]] = None,
                                         pred_win: int = 10,
                                         predict_with_all_particles: bool = False,
                                         simulation_manager=None) -> Optional[str]:

        try:
            # Extract prediction data.
            future_predicted_trajectory = prediction_data.get('predicted_trajectory')
            future_real_trajectory = prediction_data.get('original_trajectory_segment')
            prediction_start_time = prediction_data.get('prediction_start_time')

            # Extract all-particle prediction data.
            all_future_predicted_trajectories = prediction_data.get('all_predicted_trajectories')
            all_future_weights = prediction_data.get('all_weights')

            if background_vehicles_data is None:
                print("Background vehicle data is unavailable; continuing without background vehicles")

            # Choose visualization settings according to prediction mode.
            if predict_with_all_particles and all_future_predicted_trajectories is not None:
                title_suffix = f"(Pred Win: {pred_win}, All Particles: {len(all_future_predicted_trajectories)})"
                print(
                    "Visualizing trajectories predicted by all particles "
                    f"({len(all_future_predicted_trajectories)} total)"
                )
            else:
                title_suffix = f"(Pred Win: {pred_win}, Best Particle)"
                all_future_predicted_trajectories = None
                all_future_weights = None
                print("Visualizing the best-particle predicted trajectory")

            # Delegate to the comprehensive visualizer.
            filepath = trajectory_visualizer.visualize_comprehensive_trajectories(
                observed_trajectory=obs_window,
                predicted_trajectories=predicted_trajectories,
                log_likelihoods=log_likelihoods,
                future_predicted_trajectory=future_predicted_trajectory,
                future_real_trajectory=future_real_trajectory,
                window_idx=window_idx,
                window_start=window_start,
                prediction_start_time=prediction_start_time,
                title_suffix=title_suffix,
                background_vehicles_data=background_vehicles_data,
                all_future_predicted_trajectories=all_future_predicted_trajectories,
                all_future_weights=all_future_weights,
                all_future_collision_infos=prediction_data.get('all_collision_infos'),
                if_collision=if_collision
            )

            if filepath:
                print(f"Saved comprehensive trajectory visualization: {filepath}")
                distribution_path = self.plot_prediction_distribution(
                    prediction_data, window_idx,
                    simulation_manager=simulation_manager
                )
                if distribution_path:
                    prediction_data['mc_distribution_plot'] = distribution_path
                    print(f"Saved Monte Carlo distribution visualization: {distribution_path}")
                return filepath
            else:
                print("Failed to save the comprehensive trajectory visualization")
                return None

        except Exception as e:
            print(f"Failed to create the comprehensive trajectory visualization: {str(e)}")
            return None

    def compute_gt_log_likelihood(self,
                                   mean_xy: np.ndarray,
                                   std_xy: np.ndarray,
                                   gt_trajectory: List['TrajectoryPoint'],
                                   min_std: float = 0.01) -> Dict[str, Any]:

        if mean_xy.size == 0 or std_xy.size == 0 or not gt_trajectory:
            return {'log_likelihood': None, 'valid': False, 'reason': 'Empty data'}

        # Extract ground-truth coordinates.
        gt_x = np.array([p.px for p in gt_trajectory], dtype=np.float32)
        gt_y = np.array([p.py for p in gt_trajectory], dtype=np.float32)

        # Align the number of time steps by taking the shorter length.
        T_pred = len(mean_xy)
        T_gt = len(gt_trajectory)
        T = min(T_pred, T_gt)

        if T == 0:
            return {'log_likelihood': None, 'valid': False, 'reason': 'No aligned steps'}

        # Slice aligned data.
        mean_x = mean_xy[:T, 0]
        mean_y = mean_xy[:T, 1]
        std_x = np.maximum(std_xy[:T, 0], min_std)  # Enforce a minimum standard deviation.
        std_y = np.maximum(std_xy[:T, 1], min_std)
        gt_x = gt_x[:T]
        gt_y = gt_y[:T]

        # Filter out NaN values.
        valid_mask = ~(np.isnan(mean_x) | np.isnan(mean_y) |
                      np.isnan(std_x) | np.isnan(std_y) |
                      np.isnan(gt_x) | np.isnan(gt_y))

        if not np.any(valid_mask):
            return {'log_likelihood': None, 'valid': False, 'reason': 'All NaN values'}

        mean_x = mean_x[valid_mask]
        mean_y = mean_y[valid_mask]
        std_x = std_x[valid_mask]
        std_y = std_y[valid_mask]
        gt_x = gt_x[valid_mask]
        gt_y = gt_y[valid_mask]

        T_valid = len(mean_x)

        # Compute Gaussian log-likelihood.
        # log N(x; μ, σ) = -0.5 * log(2πσ²) - (x-μ)²/(2σ²)
        #                = -0.5 * [log(2π) + 2*log(σ) + (x-μ)²/σ²]
        log_2pi = np.log(2 * np.pi)

        # X-dimension log-likelihood.
        log_prob_x = -0.5 * (log_2pi + 2 * np.log(std_x) + ((gt_x - mean_x) / std_x) ** 2)

        # Y-dimension log-likelihood.
        log_prob_y = -0.5 * (log_2pi + 2 * np.log(std_y) + ((gt_y - mean_y) / std_y) ** 2)

        # Total log-likelihood per step.
        log_prob_per_step = log_prob_x + log_prob_y

        # Total log-likelihood.
        total_log_likelihood = np.sum(log_prob_per_step)

        # Average per-step log-likelihood for easier comparison across lengths.
        avg_log_likelihood = total_log_likelihood / T_valid

        # Compute negative log-likelihood.
        nll = -total_log_likelihood
        avg_nll = -avg_log_likelihood

        # Error statistics per step.
        error_x = gt_x - mean_x
        error_y = gt_y - mean_y
        mahalanobis_dist = np.sqrt((error_x / std_x) ** 2 + (error_y / std_y) ** 2)

        return {
            'log_likelihood': float(total_log_likelihood),
            'avg_log_likelihood': float(avg_log_likelihood),
            'nll': float(nll),
            'avg_nll': float(avg_nll),
            'valid': True,
            'num_steps': int(T_valid),
            'num_pred_steps': int(T_pred),
            'num_gt_steps': int(T_gt),
            'log_prob_per_step': log_prob_per_step.tolist(),
            'mean_error_x': float(np.mean(np.abs(error_x))),
            'mean_error_y': float(np.mean(np.abs(error_y))),
            'mean_mahalanobis': float(np.mean(mahalanobis_dist)),
            'std_mahalanobis': float(np.std(mahalanobis_dist)),
        }

    def get_background_vehicle_trajectories(self,
                                            simulation_manager,
                                            start_timestamp: float,
                                            end_timestamp: float) -> Optional[List[Dict]]:

        if not hasattr(simulation_manager, 'full_data') or simulation_manager.full_data is None:
            print("simulation_manager.full_data is unavailable")
            return None

        full_data = simulation_manager.full_data

        print(f"Querying background vehicles for time range [{start_timestamp:.3f}, {end_timestamp:.3f}]")
        print(f"   CSV time range: [{full_data['timestamp'].min():.3f}, {full_data['timestamp'].max():.3f}]")

        # Exclude the ego vehicle (vehicle_id == -1).
        background_data = full_data[full_data['vehicle_id'] != -1]

        if background_data.empty:
            print("No background-vehicle data is available")
            return None

        print(
            f"   Background vehicles: {background_data['vehicle_id'].nunique()}, "
            f"total records: {len(background_data)}"
        )

        # Filter data within the requested time range using a small tolerance.
        time_margin = 0.5  # Add a 0.5-second tolerance.
        time_mask = (background_data['timestamp'] >= start_timestamp - time_margin) & \
                    (background_data['timestamp'] <= end_timestamp + time_margin)
        segment_data = background_data[time_mask]

        if segment_data.empty:
            print("No background-vehicle data falls within the requested time range")
            return None

        print(f"   Background-vehicle records in range: {len(segment_data)}")

        # Group by vehicle ID.
        background_trajectories = []
        for vehicle_id in segment_data['vehicle_id'].unique():
            vehicle_data = segment_data[segment_data['vehicle_id'] == vehicle_id].sort_values('timestamp')

            if len(vehicle_data) >= 1:
                trajectory = [
                    (row['position_x'], row['position_y'], row['timestamp'])
                    for _, row in vehicle_data.iterrows()
                ]
                background_trajectories.append({
                    'vehicle_id': int(vehicle_id),
                    'trajectory': trajectory
                })

        if background_trajectories:
            print(f"   Retrieved trajectories for {len(background_trajectories)} background vehicles")

        return background_trajectories if background_trajectories else None

    def check_3sigma_intersection(self,
                                  mean_xy: np.ndarray,
                                  std_xy: np.ndarray,
                                  bg_trajectory: List[tuple],
                                  sigma_multiplier: float = 3.0,
                                  vehicle_buffer: float = 1.5,
                                  prediction_buffer: float = 1.5) -> List[Dict]:

        intersections = []

        if mean_xy.size == 0 or std_xy.size == 0 or not bg_trajectory:
            return intersections

        T = len(mean_xy)

        # Total buffer = background-vehicle buffer + prediction buffer.
        total_buffer = vehicle_buffer + prediction_buffer

        for bg_x, bg_y, bg_ts in bg_trajectory:
            for t in range(T):
                if np.isnan(mean_xy[t]).any() or np.isnan(std_xy[t]).any():
                    continue

                mx, my = mean_xy[t]
                sx, sy = std_xy[t]

                # Enforce a minimum standard deviation.
                sx = max(sx, 0.1)
                sy = max(sy, 0.1)

                # Expanded range = 3σ range + total buffer.
                x_range = sigma_multiplier * sx + total_buffer
                y_range = sigma_multiplier * sy + total_buffer

                # Check whether the point is inside the expanded range.
                in_x_range = abs(bg_x - mx) <= x_range
                in_y_range = abs(bg_y - my) <= y_range

                if in_x_range and in_y_range:
                    # Euclidean distance.
                    euclidean_dist = np.sqrt((bg_x - mx) ** 2 + (bg_y - my) ** 2)

                    # Normalized distance relative to the original 3σ range.
                    norm_dist = np.sqrt(((bg_x - mx) / sx) ** 2 + ((bg_y - my) / sy) ** 2)

                    # Effective distance after accounting for buffer.
                    effective_dist_x = abs(bg_x - mx) - sigma_multiplier * sx
                    effective_dist_y = abs(bg_y - my) - sigma_multiplier * sy

                    # Flag points inside the core 3σ range, excluding the buffer.
                    in_core_range = (effective_dist_x <= 0) and (effective_dist_y <= 0)

                    intersections.append({
                        'pred_step': t,
                        'bg_point': (bg_x, bg_y, bg_ts),
                        'pred_mean': (mx, my),
                        'pred_std': (sx, sy),
                        'normalized_distance': norm_dist,
                        'euclidean_distance': euclidean_dist,
                        'in_core_3sigma': in_core_range,
                        'buffer_margin_x': total_buffer - max(0, effective_dist_x),
                        'buffer_margin_y': total_buffer - max(0, effective_dist_y)
                    })

        return intersections

    def plot_prediction_distribution(self, prediction_data: Dict[str, Any], window_idx: int,
                                     title_suffix: str = "",
                                     simulation_manager=None) -> Optional[str]:
        mc_stats = prediction_data.get('mc_stats')
        if not mc_stats:
            print("Current prediction is missing Monte Carlo statistics; skipping distribution visualization")
            return None

        positions_raw = mc_stats.get('positions')
        if not positions_raw:
            print("Monte Carlo position data is empty; skipping distribution visualization")
            return None

        try:
            positions = np.asarray(positions_raw, dtype=np.float32)
        except ValueError:
            print("Monte Carlo position data has an invalid format; skipping distribution visualization")
            return None

        if positions.size == 0:
            print("Monte Carlo position array is empty; skipping distribution visualization")
            return None

        mean_xy = np.asarray(mc_stats.get('mean_xy', []), dtype=np.float32)
        std_xy = np.asarray(mc_stats.get('std_xy', []), dtype=np.float32)
        time_step = float(mc_stats.get('time_step', 0.1))

        # Compute the log-likelihood of the ground-truth trajectory.
        original_trajectory_segment = prediction_data.get('original_trajectory_segment')
        gt_likelihood_info = self.compute_gt_log_likelihood(mean_xy, std_xy, original_trajectory_segment)

        # Store log-likelihood details for downstream use.
        prediction_data['gt_likelihood_info'] = gt_likelihood_info

        # Fetch background trajectories and detect intersections.
        bg_trajectories = None
        intersecting_vehicles = {}  # {vehicle_id: {'trajectory': [...], 'intersections': [...]}}
        all_bg_trajectories = {}  # Store all background trajectories for plotting.

        if simulation_manager is not None:
            prediction_start_time = prediction_data.get('prediction_start_time', 0.0)
            pred_win = len(mean_xy) if mean_xy.size > 0 else 10
            prediction_end_time = prediction_start_time + pred_win * time_step

            print(f"\nWindow {window_idx}: checking background-vehicle intersections...")

            bg_trajectories = self.get_background_vehicle_trajectories(
                simulation_manager, prediction_start_time, prediction_end_time
            )

            if bg_trajectories:
                # Print the prediction-distribution range.
                if mean_xy.size > 0 and std_xy.size > 0:
                    valid_mask = ~np.isnan(mean_xy).any(axis=1) & ~np.isnan(std_xy).any(axis=1)
                    if np.any(valid_mask):
                        mean_valid = mean_xy[valid_mask]
                        std_valid = std_xy[valid_mask]
                        print(f"   Predicted trajectory range: X=[{mean_valid[:,0].min():.1f}, {mean_valid[:,0].max():.1f}], "
                              f"Y=[{mean_valid[:,1].min():.1f}, {mean_valid[:,1].max():.1f}]")
                        print(f"   Predicted std range: sx=[{std_valid[:,0].min():.3f}, {std_valid[:,0].max():.3f}], "
                              f"sy=[{std_valid[:,1].min():.3f}, {std_valid[:,1].max():.3f}]")

                for bg_vehicle in bg_trajectories:
                    vehicle_id = bg_vehicle['vehicle_id']
                    trajectory = bg_vehicle['trajectory']

                    # Store all background trajectories.
                    all_bg_trajectories[vehicle_id] = trajectory

                    # Print background-vehicle bounds.
                    if trajectory:
                        bg_xs = [p[0] for p in trajectory]
                        bg_ys = [p[1] for p in trajectory]
                        print(
                            f"   Background vehicle {vehicle_id}: "
                            f"X=[{min(bg_xs):.1f}, {max(bg_xs):.1f}], "
                            f"Y=[{min(bg_ys):.1f}, {max(bg_ys):.1f}], "
                            f"points={len(trajectory)}"
                        )

                    # Detect intersections with the 3σ region.
                    intersections = self.check_3sigma_intersection(mean_xy, std_xy, trajectory)

                    if intersections:
                        intersecting_vehicles[vehicle_id] = {
                            'trajectory': trajectory,
                            'intersections': intersections
                        }

                if intersecting_vehicles:
                    print(
                        f"Window {window_idx}: detected {len(intersecting_vehicles)} background vehicles "
                        "intersecting the 3σ+buffer region"
                    )
                    for vid, info in intersecting_vehicles.items():
                        core_count = sum(1 for x in info['intersections'] if x.get('in_core_3sigma', False))
                        buffer_count = len(info['intersections']) - core_count
                        min_dist = min(x['euclidean_distance'] for x in info['intersections'])
                        print(
                            f"   - Vehicle {vid}: {len(info['intersections'])} intersections "
                            f"(core 3σ: {core_count}, buffer: {buffer_count}, "
                            f"closest distance: {min_dist:.2f}m)"
                        )
                else:
                    print("   No background vehicle intersects the 3σ+buffer region (buffer=1.5m+1.5m=3.0m)")
        else:
            print(f"Window {window_idx}: simulation_manager is None; background-vehicle data cannot be retrieved")

        # Print log-likelihood diagnostics.
        if gt_likelihood_info.get('valid'):
            print(f"Window {window_idx}: GT Log-Likelihood = {gt_likelihood_info['log_likelihood']:.4f}, "
                  f"Avg NLL = {gt_likelihood_info['avg_nll']:.4f}, "
                  f"Steps = {gt_likelihood_info['num_steps']}")

        sample_count = positions.shape[0]
        step_count = positions.shape[1] if positions.ndim == 3 else 0
        if step_count == 0:
            print("Monte Carlo trajectory length is zero; skipping distribution visualization")
            return None

        cluster_labels = prediction_data.get('mc_cluster_labels')
        if cluster_labels is None and mc_stats.get('cluster_info'):
            cluster_labels = mc_stats['cluster_info'].get('labels')

        if cluster_labels is None:
            cluster_array = np.zeros(sample_count, dtype=int)
            label_names = {0: 'Samples'}
        else:
            cluster_array = np.array(
                [lbl if lbl is not None else -1 for lbl in cluster_labels], dtype=int
            )
            unique_labels = sorted(set(cluster_array.tolist()))
            label_names = {lbl: (f'Cluster {lbl}' if lbl >= 0 else 'Unclustered') for lbl in unique_labels}

        time_axis = np.arange(step_count) * time_step

        fig = plt.figure(figsize=(12, 50))
        ax_xy = fig.add_subplot(111)

        # Track whether axis limits have already been set.
        axis_limits_set = False

        # Scatter-density view.
        flattened = positions.reshape(-1, 2)
        valid_mask = ~np.isnan(flattened).any(axis=1)
        valid_points = flattened[valid_mask]
        if len(valid_points) > 0:
            ax_xy.hexbin(valid_points[:, 0], valid_points[:, 1], gridsize=30, cmap='Blues', alpha=0.3)

        # Plot each sample trajectory.
        num_clusters = len([lbl for lbl in set(cluster_array.tolist()) if lbl >= 0])
        cmap = plt.cm.get_cmap('tab10', max(num_clusters, 1))
        handled_labels = set()
        for idx in range(sample_count):
            traj = positions[idx]
            mask = ~np.isnan(traj).any(axis=1)
            if not mask.any():
                continue
            coords = traj[mask]
            label_value = cluster_array[idx] if idx < len(cluster_array) else -1
            if label_value >= 0:
                color = cmap(label_value % cmap.N)
            else:
                color = '#7f8c8d'
            legend_label = None
            if label_value not in handled_labels:
                legend_label = label_names.get(label_value, 'Samples')
                handled_labels.add(label_value)
            ax_xy.plot(coords[:, 0], coords[:, 1], linewidth=1.2, alpha=0.6,
                       color=color, label=legend_label)
            ax_xy.scatter(coords[-1, 0], coords[-1, 1], s=35, color=color, alpha=0.85)

        # Plot the ground-truth trajectory.
        original_trajectory_segment = prediction_data.get('original_trajectory_segment')
        if original_trajectory_segment and len(original_trajectory_segment) > 0:
            gt_x = [p.px for p in original_trajectory_segment]
            gt_y = [p.py for p in original_trajectory_segment]
            ax_xy.plot(gt_x, gt_y, color='red', linewidth=3.0, alpha=0.9,
                      label='Ground Truth', linestyle='-', zorder=10)
            # Mark the ground-truth start and end points.
            ax_xy.scatter(gt_x[0], gt_y[0], s=120, color='red', marker='o',
                         edgecolor='white', linewidth=2.5, zorder=11, label='GT Start')
            ax_xy.scatter(gt_x[-1], gt_y[-1], s=120, color='red', marker='s',
                         edgecolor='white', linewidth=2.5, zorder=11, label='GT End')

        # Plot the mean trajectory.
        if mean_xy.size > 0:
            mean_mask = ~np.isnan(mean_xy).any(axis=1)
            mean_coords = mean_xy[mean_mask]
            if len(mean_coords) > 0:
                ax_xy.plot(mean_coords[:, 0], mean_coords[:, 1], color='black', linewidth=2.0,
                           label='Mean Trajectory')
                ax_xy.scatter(mean_coords[0, 0], mean_coords[0, 1], s=80, color='black',
                              edgecolor='white', linewidth=1.5, label='Mean Start')
                ax_xy.scatter(mean_coords[-1, 0], mean_coords[-1, 1], s=80, color='black',
                              marker='s', edgecolor='white', linewidth=1.5, label='Mean End')

            # Plot the confidence region.
            if std_xy.size > 0:
                x_mean = mean_xy[:, 0]
                y_mean = mean_xy[:, 1]
                x_std = std_xy[:, 0]
                y_std = std_xy[:, 1]

                # Filter valid data points.
                valid_mask = ~(np.isnan(x_mean) | np.isnan(y_mean) | np.isnan(x_std) | np.isnan(y_std))
                if np.any(valid_mask):
                    x_valid = x_mean[valid_mask]
                    y_valid = y_mean[valid_mask]
                    x_std_valid = x_std[valid_mask]
                    y_std_valid = y_std[valid_mask]

                    # Plot the ±3σ confidence region.
                    ax_xy.fill_between(x_valid,
                                      y_valid - 3*y_std_valid,
                                      y_valid + 3*y_std_valid,
                                      color='gray', alpha=0.1, label='±3σ Confidence')

                    # Choose axis limits that leave enough room for all paths.
                    x_min, x_max = np.min(x_valid), np.max(x_valid)
                    y_min, y_max = np.min(y_valid), np.max(y_valid)

                    # Include ground-truth bounds when available.
                    if original_trajectory_segment and len(original_trajectory_segment) > 0:
                        gt_x = [p.px for p in original_trajectory_segment]
                        gt_y = [p.py for p in original_trajectory_segment]
                        if len(gt_x) > 0 and len(gt_y) > 0:
                            x_min = min(x_min, np.min(gt_x))
                            x_max = max(x_max, np.max(gt_x))
                            y_min = min(y_min, np.min(gt_y))
                            y_max = max(y_max, np.max(gt_y))

                    # Add margins so every trajectory and confidence region is visible.
                    x_range = x_max - x_min
                    y_range = y_max - y_min

                    x_margin_ratio = 0.1  # X-axis margin ratio.
                    y_margin_ratio = 3    # Y-axis margin ratio.
                    min_margin = 3.0      # Minimum margin.

                    x_margin = max(x_range * x_margin_ratio, min_margin)
                    y_margin = max(y_range * y_margin_ratio, min_margin)

                    ax_xy.set_xlim(x_min - x_margin, x_max + x_margin)
                    ax_xy.set_ylim(y_min - y_margin, y_max + y_margin)
                    axis_limits_set = True

            # Retain the original ellipse visualization.
            if std_xy.size > 0:
                stride = max(1, step_count // 12)
                for step in range(0, step_count, stride):
                    if step >= len(mean_xy):
                        break
                    if np.isnan(mean_xy[step]).any() or np.isnan(std_xy[step]).any():
                        continue
                    width = 2 * std_xy[step, 0]
                    height = 2 * std_xy[step, 1]
                    if width <= 0 or height <= 0:
                        continue
                    ellipse = Ellipse(
                        (mean_xy[step, 0], mean_xy[step, 1]),
                        width=width,
                        height=height,
                        angle=0,
                        edgecolor='black',
                        facecolor='none',
                        linestyle='--',
                        linewidth=0.8,
                        alpha=0.5
                    )
                    ax_xy.add_patch(ellipse)

        # If confidence data did not set axis limits, derive them from the paths.
        if not axis_limits_set:
            all_x = []
            all_y = []

            # Collect all predicted-trajectory points.
            for idx in range(sample_count):
                traj = positions[idx]
                mask = ~np.isnan(traj).any(axis=1)
                if mask.any():
                    all_x.extend(traj[mask, 0].tolist())
                    all_y.extend(traj[mask, 1].tolist())

            # Collect mean-trajectory points.
            if mean_xy.size > 0:
                mean_mask = ~np.isnan(mean_xy).any(axis=1)
                if mean_mask.any():
                    all_x.extend(mean_xy[mean_mask, 0].tolist())
                    all_y.extend(mean_xy[mean_mask, 1].tolist())

            # Collect ground-truth points.
            if original_trajectory_segment and len(original_trajectory_segment) > 0:
                gt_x = [p.px for p in original_trajectory_segment]
                gt_y = [p.py for p in original_trajectory_segment]
                all_x.extend(gt_x)
                all_y.extend(gt_y)

            if len(all_x) > 0 and len(all_y) > 0:
                x_min, x_max = np.min(all_x), np.max(all_x)
                y_min, y_max = np.min(all_y), np.max(all_y)

                x_range = x_max - x_min
                y_range = y_max - y_min

                x_margin_ratio = 0.1
                y_margin_ratio = 3
                min_margin = 3.0

                x_margin = max(x_range * x_margin_ratio, min_margin)
                y_margin = max(y_range * y_margin_ratio, min_margin)

                ax_xy.set_xlim(x_min - x_margin, x_max + x_margin)
                ax_xy.set_ylim(y_min - y_margin, y_max + y_margin)

        # Plot all background-vehicle trajectories.
        if all_bg_trajectories:
            all_vehicle_ids = list(all_bg_trajectories.keys())
            bg_colors = plt.cm.get_cmap('tab20', max(len(all_vehicle_ids), 1))

            for i, vehicle_id in enumerate(all_vehicle_ids):
                trajectory = all_bg_trajectories[vehicle_id]
                is_intersecting = vehicle_id in intersecting_vehicles
                color = bg_colors(i % bg_colors.N)

                # Use a bolder style for intersecting vehicles.
                if is_intersecting:
                    linewidth = 3.0
                    alpha = 1.0
                    linestyle = '-'
                    marker_size = 120
                else:
                    linewidth = 2.0
                    alpha = 0.85
                    linestyle = '--'
                    marker_size = 80

                # Plot background trajectories.
                if len(trajectory) >= 2:
                    bg_x = [p[0] for p in trajectory]
                    bg_y = [p[1] for p in trajectory]
                    label = f'BG V{vehicle_id}' + ('' if is_intersecting else '')
                    ax_xy.plot(bg_x, bg_y, color=color, linewidth=linewidth, alpha=alpha,
                              linestyle=linestyle, label=label, zorder=15 if is_intersecting else 5)
                    # Mark start and end points.
                    ax_xy.scatter(bg_x[0], bg_y[0], s=marker_size, color=color, marker='^',
                                 edgecolor='white', linewidth=1.5, zorder=16 if is_intersecting else 6, alpha=alpha)
                    ax_xy.scatter(bg_x[-1], bg_y[-1], s=marker_size, color=color, marker='v',
                                 edgecolor='white', linewidth=1.5, zorder=16 if is_intersecting else 6, alpha=alpha)
                elif len(trajectory) == 1:
                    label = f'BG V{vehicle_id}' + ('' if is_intersecting else '')
                    ax_xy.scatter(trajectory[0][0], trajectory[0][1], s=marker_size, color=color,
                                 marker='o', edgecolor='white', linewidth=2, alpha=alpha,
                                 label=label, zorder=15 if is_intersecting else 5)

        # Mark intersection points.
        if intersecting_vehicles:
            for vehicle_id, vehicle_info in intersecting_vehicles.items():
                intersections = vehicle_info['intersections']

                # Distinguish core 3σ intersections from buffer-only intersections.
                for j, intersection in enumerate(intersections):
                    ix, iy, _ = intersection['bg_point']
                    pred_step = intersection['pred_step']
                    norm_dist = intersection['normalized_distance']
                    eucl_dist = intersection['euclidean_distance']
                    in_core = intersection.get('in_core_3sigma', False)

                    # Use red for core 3σ intersections and orange for buffer-only cases.
                    if in_core:
                        marker_color = 'red'
                        marker_edge = 'yellow'
                        risk_label = 'core'
                    else:
                        marker_color = 'orange'
                        marker_edge = 'white'
                        risk_label = 'buffer'

                    ax_xy.scatter(ix, iy, s=150, color=marker_color, marker='X',
                                 edgecolor=marker_edge, linewidth=2, zorder=20)

                    # Annotate only a few intersections to avoid clutter.
                    if j < 3:
                        ax_xy.annotate(
                            f'V{vehicle_id} {risk_label}\nstep:{pred_step}\n{eucl_dist:.1f}m',
                            xy=(ix, iy), xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=7, color=marker_color,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                            arrowprops=dict(arrowstyle='->', color=marker_color, lw=1),
                            zorder=21
                        )

            # Store intersection details in prediction_data.
            prediction_data['intersecting_vehicles'] = {
                vid: {
                    'num_intersections': len(info['intersections']),
                    'min_normalized_distance': min(x['normalized_distance'] for x in info['intersections']),
                    'intersections': info['intersections']
                }
                for vid, info in intersecting_vehicles.items()
            }

        # Build the title with log-likelihood and intersection summaries.
        title_str = f'Monte Carlo Prediction Distribution (Window {window_idx + 1})'
        if gt_likelihood_info.get('valid'):
            title_str += f'\nGT Log-Likelihood: {gt_likelihood_info["log_likelihood"]:.2f} | Avg NLL: {gt_likelihood_info["avg_nll"]:.2f}'
        if intersecting_vehicles:
            vehicle_ids_str = ', '.join([str(vid) for vid in intersecting_vehicles.keys()])
            title_str += f'\n3σ Intersection: Vehicles [{vehicle_ids_str}]'
        if title_suffix:
            title_str += f' {title_suffix}'

        ax_xy.set_title(title_str, fontsize=14, fontweight='bold')
        ax_xy.set_xlabel('Position X (m)')
        ax_xy.set_ylabel('Position Y (m)')
        ax_xy.grid(True, alpha=0.3, linestyle='--')
        ax_xy.set_aspect('equal', adjustable='box')
        ax_xy.legend(loc='upper right', fontsize=9, frameon=True, framealpha=0.85)

        # Add a log-likelihood info box.
        if gt_likelihood_info.get('valid'):
            info_text = (
                f"GT Log-Likelihood Stats:\n"
                f"  Total LL: {gt_likelihood_info['log_likelihood']:.4f}\n"
                f"  Avg LL/step: {gt_likelihood_info['avg_log_likelihood']:.4f}\n"
                f"  Avg NLL: {gt_likelihood_info['avg_nll']:.4f}\n"
                f"  Steps: {gt_likelihood_info['num_steps']}\n"
                f"  Mean |err_x|: {gt_likelihood_info['mean_error_x']:.3f}m\n"
                f"  Mean |err_y|: {gt_likelihood_info['mean_error_y']:.3f}m\n"
                f"  Mean Mahal.: {gt_likelihood_info['mean_mahalanobis']:.3f}"
            )
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax_xy.text(0.02, 0.98, info_text, transform=ax_xy.transAxes, fontsize=8,
                      verticalalignment='top', fontfamily='monospace', bbox=props)

        plt.tight_layout()

        distribution_dir = os.path.join(self.output_dir, 'prediction_distribution')
        os.makedirs(distribution_dir, exist_ok=True)
        filename = f'prediction_distribution_window_{window_idx:04d}.pdf'
        filepath = os.path.join(distribution_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', format='pdf')
        plt.close(fig)

        return filepath

    def get_background_vehicles_data(self,
                                   simulation_manager,
                                   current_timestamp: float) -> Optional[List[Dict]]:

        if not hasattr(simulation_manager, 'full_data') or simulation_manager.full_data is None:
            return None

        full_data = simulation_manager.full_data

        # Exclude the ego vehicle (vehicle_id == -1).
        background_data = full_data[full_data['vehicle_id'] != -1]

        if background_data.empty:
            return None

        # Group by vehicle ID.
        background_vehicles = []
        for vehicle_id in background_data['vehicle_id'].unique():
            vehicle_data = background_data[background_data['vehicle_id'] == vehicle_id]

            # Compute absolute time differences.
            time_diffs = abs(vehicle_data['timestamp'] - current_timestamp)

            # Find the closest timestamp.
            closest_idx = time_diffs.idxmin()
            closest_row = vehicle_data.loc[closest_idx]

            # Require a reasonably close timestamp, such as within 0.5 seconds.
            if time_diffs.loc[closest_idx] <= 0.5:
                background_vehicles.append({
                    'vehicle_id': int(vehicle_id),
                    'positions': [(
                        closest_row['position_x'],
                        closest_row['position_y'],
                        closest_row['timestamp']
                    )]
                })

        return background_vehicles if background_vehicles else None

    def create_prediction_statistics_report(self, prediction_history: List[Dict], pred_win: int):
        if not prediction_history:
            return

        report_file = os.path.join(self.output_dir, 'prediction_statistics_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAJECTORY PREDICTION STATISTICS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prediction Window Size: {pred_win} steps\n")
            f.write(f"Total Prediction Instances: {len(prediction_history)}\n\n")

            # Statistics for each prediction moment
            f.write("## PREDICTION INSTANCE DETAILS\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"{'Window':<8} {'Start Time':<12} {'Pred Points':<12} {'GT Points':<12} "
                f"{'Best Weight':<12} {'MinADE@K':<10} {'MinFDE@K':<10} {'Miss@K':<8} "
                f"{'AvgLL':<10} {'TotalVar':<10}\n"
            )
            f.write("-" * 80 + "\n")

            total_predictions = 0
            total_real_segments = 0
            # Aggregate multi-trajectory metrics if available
            ade_vals: List[float] = []
            fde_vals: List[float] = []
            miss_vals: List[float] = []
            log_likelihood_vals: List[float] = []
            avg_log_likelihood_vals: List[float] = []
            variance_vals: List[float] = []
            metric_k = None
            metric_eps = None

            for pred_data in prediction_history:
                window_idx = pred_data['window_idx']
                start_time = pred_data['prediction_start_time']
                pred_len = len(pred_data['predicted_trajectory']) if pred_data['predicted_trajectory'] else 0
                real_len = len(pred_data['original_trajectory_segment']) if pred_data['original_trajectory_segment'] else 0
                weight = pred_data.get('best_particle_weight')
                weight_val = float(weight) if weight is not None else float('nan')

                m = pred_data.get('multi_traj_metrics')
                minade_str = "-"
                minfde_str = "-"
                miss_str = "-"
                avgll_str = "-"
                totalvar_str = "-"
                if isinstance(m, dict):
                    minade = m.get('minADE')
                    minfde = m.get('minFDE')
                    miss = m.get('miss')
                    avg_log_likelihood = m.get('avg_log_likelihood')
                    total_variance = m.get('total_variance')

                    if minade is not None and np.isfinite(minade):
                        minade_str = f"{float(minade):.3f}"
                        ade_vals.append(float(minade))
                    if minfde is not None and np.isfinite(minfde):
                        minfde_str = f"{float(minfde):.3f}"
                        fde_vals.append(float(minfde))
                    if miss is not None and np.isfinite(miss):
                        miss_str = f"{float(miss):.0f}"
                        miss_vals.append(float(miss))
                    if avg_log_likelihood is not None and np.isfinite(avg_log_likelihood):
                        avgll_str = f"{float(avg_log_likelihood):.3f}"
                        avg_log_likelihood_vals.append(float(avg_log_likelihood))
                        log_likelihood_vals.append(float(m.get('log_likelihood', float('nan'))))
                    if total_variance is not None and np.isfinite(total_variance):
                        totalvar_str = f"{float(total_variance):.4f}"
                        variance_vals.append(float(total_variance))
                    if metric_k is None and m.get('k') is not None:
                        metric_k = int(float(m.get('k')))
                    if metric_eps is None and m.get('epsilon') is not None:
                        metric_eps = float(m.get('epsilon'))

                f.write(
                    f"W{window_idx+1:<7} {start_time:<12.3f} {pred_len:<12} {real_len:<12} "
                    f"{weight_val:<12.6f} {minade_str:<10} {minfde_str:<10} {miss_str:<8} "
                    f"{avgll_str:<10} {totalvar_str:<10}\n"
                )

                if pred_len > 0:
                    total_predictions += 1
                if real_len > 0:
                    total_real_segments += 1

            f.write("-" * 80 + "\n")
            f.write(f"\nSuccessful Predictions: {total_predictions}/{len(prediction_history)}\n")
            f.write(f"Predictions with Ground Truth: {total_real_segments}/{len(prediction_history)}\n")

            if ade_vals or fde_vals or miss_vals or log_likelihood_vals or variance_vals:
                f.write("\n## MULTI-TRAJECTORY METRICS SUMMARY\n")
                f.write("-" * 80 + "\n")
                if metric_k is not None:
                    f.write(f"K: {metric_k}\n")
                if metric_eps is not None:
                    f.write(f"Miss epsilon (m): {metric_eps:.3f}\n")
                if ade_vals:
                    f.write(f"Mean MinADE@K: {float(np.mean(ade_vals)):.4f}\n")
                if fde_vals:
                    f.write(f"Mean MinFDE@K: {float(np.mean(fde_vals)):.4f}\n")
                if miss_vals:
                    f.write(f"MissRate@K: {float(np.mean(miss_vals)):.4f}\n")

                # Add log-likelihood statistics.
                if avg_log_likelihood_vals:
                    f.write(f"\n## LOG-LIKELIHOOD METRICS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Mean Avg Log-Likelihood: {float(np.mean(avg_log_likelihood_vals)):.4f}\n")
                    if log_likelihood_vals:
                        valid_ll = [ll for ll in log_likelihood_vals if np.isfinite(ll)]
                        if valid_ll:
                            f.write(f"Mean Total Log-Likelihood: {float(np.mean(valid_ll)):.4f}\n")

                # Add variance statistics.
                if variance_vals:
                    f.write(f"\n## VARIANCE METRICS\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"Mean Total Variance: {float(np.mean(variance_vals)):.4f}\n")
                    f.write(f"Std Total Variance: {float(np.std(variance_vals)):.4f}\n")
                    f.write(f"Min Total Variance: {float(np.min(variance_vals)):.4f}\n")
                    f.write(f"Max Total Variance: {float(np.max(variance_vals)):.4f}\n")

            # Optimal parameter statistics
            f.write(f"\n## OPTIMAL PARAMETER STATISTICS\n")
            f.write("-" * 80 + "\n")

            # Collect all optimal parameters
            param_values = {}
            for pred_data in prediction_history:
                for param, value in pred_data['best_theta'].items():
                    if param not in param_values:
                        param_values[param] = []
                    param_values[param].append(value)

            for param, values in param_values.items():
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                f.write(f"{param:<30}: Mean={mean_val:8.4f}, Std={std_val:8.4f}, "
                       f"Range=[{min_val:8.4f}, {max_val:8.4f}]\n")

        print(f"Prediction statistics report saved to: {report_file}")

    def export_multi_traj_metrics_to_csv(self, prediction_history: List[Dict], pred_win: int):

        if not prediction_history:
            print("No prediction history is available for CSV export")
            return

        import pandas as pd

        csv_rows = []

        for pred_data in prediction_history:
            window_idx = pred_data.get('window_idx', -1)
            start_time = pred_data.get('prediction_start_time', 0.0)
            pred_len = len(pred_data.get('predicted_trajectory', []))
            gt_len = len(pred_data.get('original_trajectory_segment', []))
            weight = pred_data.get('best_particle_weight')

            # Extract multi-trajectory metrics.
            metrics = pred_data.get('multi_traj_metrics')
            if metrics is None:
                # Record basic information even when metrics are missing.
                row = {
                    'window_idx': window_idx,
                    'prediction_start_time': start_time,
                    'predicted_points': pred_len,
                    'gt_points': gt_len,
                    'best_particle_weight': weight if weight is not None else float('nan'),
                    'minADE_at_K': float('nan'),
                    'minFDE_at_K': float('nan'),
                    'miss_rate_at_K': float('nan'),
                    'spread_pair': float('nan'),
                    'k': float('nan'),
                    'k_effective': float('nan'),
                    'miss_epsilon': float('nan'),
                    'gt_steps': float('nan'),
                    'metric_source': 'N/A'
                }
            else:
                row = {
                    'window_idx': window_idx,
                    'prediction_start_time': start_time,
                    'predicted_points': pred_len,
                    'gt_points': gt_len,
                    'best_particle_weight': weight if weight is not None else float('nan'),
                    'minADE_at_K': float(metrics.get('minADE', float('nan'))) if metrics.get('minADE') is not None else float('nan'),
                    'minFDE_at_K': float(metrics.get('minFDE', float('nan'))) if metrics.get('minFDE') is not None else float('nan'),
                    'miss_rate_at_K': float(metrics.get('miss', float('nan'))) if metrics.get('miss') is not None else float('nan'),
                    'spread_pair': float(metrics.get('spread_pair', float('nan'))) if metrics.get('spread_pair') is not None else float('nan'),
                    'log_likelihood': float(metrics.get('log_likelihood', float('nan'))) if metrics.get('log_likelihood') is not None else float('nan'),
                    'avg_log_likelihood': float(metrics.get('avg_log_likelihood', float('nan'))) if metrics.get('avg_log_likelihood') is not None else float('nan'),
                    'nll': float(metrics.get('nll', float('nan'))) if metrics.get('nll') is not None else float('nan'),
                    'avg_nll': float(metrics.get('avg_nll', float('nan'))) if metrics.get('avg_nll') is not None else float('nan'),
                    'total_variance': float(metrics.get('total_variance', float('nan'))) if metrics.get('total_variance') is not None else float('nan'),
                    'mean_variance': float(metrics.get('mean_variance', float('nan'))) if metrics.get('mean_variance') is not None else float('nan'),
                    'k': float(metrics.get('k', float('nan'))) if metrics.get('k') is not None else float('nan'),
                    'k_effective': float(metrics.get('k_effective', float('nan'))) if metrics.get('k_effective') is not None else float('nan'),
                    'miss_epsilon': float(metrics.get('epsilon', float('nan'))) if metrics.get('epsilon') is not None else float('nan'),
                    'gt_steps': float(metrics.get('gt_steps', float('nan'))) if metrics.get('gt_steps') is not None else float('nan'),
                    'metric_source': str(metrics.get('source', 'unknown'))
                }

            csv_rows.append(row)

        # Convert to a DataFrame and save.
        df = pd.DataFrame(csv_rows)

        # Sort by window index.
        df = df.sort_values('window_idx')

        # Save the CSV file.
        csv_path = os.path.join(self.output_dir, 'multi_traj_metrics.csv')
        df.to_csv(csv_path, index=False, float_format='%.6f')

        print(f"Saved multi-trajectory metrics CSV: {csv_path}")
        print(f"   Total records: {len(csv_rows)}")

        # Summarize valid metric counts.
        valid_metrics = df[df['minADE_at_K'].notna()]
        if len(valid_metrics) > 0:
            print(f"   Valid metric records: {len(valid_metrics)}/{len(csv_rows)}")
            print(f"   Mean MinADE@K: {valid_metrics['minADE_at_K'].mean():.4f} m")
            print(f"   Mean MinFDE@K: {valid_metrics['minFDE_at_K'].mean():.4f} m")
            valid_spread = df[df['spread_pair'].notna()]
            if len(valid_spread) > 0:
                print(f"   Mean Spread_pair: {valid_spread['spread_pair'].mean():.4f} m")
            print(f"   Mean MissRate@K: {valid_metrics['miss_rate_at_K'].mean():.4f}")

        return csv_path
