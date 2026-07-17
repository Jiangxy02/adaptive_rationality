#!/usr/bin/env python3
"""
Trajectory-visualization utilities used during parameter identification.
"""


import numpy as np
# Use a non-interactive matplotlib backend to avoid tkinter conflicts in multiprocessing.
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot.
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, is_color_like
import os
from typing import List, Optional, Dict
from parameter_identify.utils.likelihood_calculator import TrajectoryPoint


class TrajectoryVisualizer:
    """Trajectory visualizer."""

    def __init__(self,
                 output_dir: str,
                 enable_visualization: bool = True,
                 visualization_interval: int = 1,
                 figure_size: tuple = (12, 8),
                 dpi: int = 150):
        """
        Initialize the trajectory visualizer.
        """
        self.output_dir = output_dir
        self.enable_visualization = enable_visualization
        self.visualization_interval = visualization_interval
        self.figure_size = figure_size
        self.dpi = dpi

        # Window counter.
        self.current_window_count = 0

        # Create the visualization output directory.
        self.viz_dir = os.path.join(output_dir, 'trajectory_visualization')
        os.makedirs(self.viz_dir, exist_ok=True)

    def should_visualize(self) -> bool:
        """
        Return whether the current window should be visualized.
        """
        if not self.enable_visualization:
            return False

        self.current_window_count += 1
        return self.current_window_count % self.visualization_interval == 0

    def visualize_trajectories(self,
                             observed_trajectory: List[TrajectoryPoint],
                             predicted_trajectories: List[List[TrajectoryPoint]],
                             log_likelihoods: np.ndarray,
                             window_idx: int,
                             window_start: int,
                             title_suffix: str = "") -> Optional[str]:
        """
        Visualize the observed trajectory and predicted particle trajectories.
        """
        try:
            fig, ax = plt.subplots(1, 1, figsize=self.figure_size)

            self._plot_observed_trajectory(ax, observed_trajectory)
            self._plot_predicted_trajectories(ax, predicted_trajectories, log_likelihoods)

            self._setup_plot_appearance(ax, window_idx, window_start, title_suffix)

            self._add_statistics_text(ax, predicted_trajectories, log_likelihoods)

            plt.tight_layout()

            filepath = self._save_figure(window_idx)

            plt.close()

            return filepath

        except Exception as e:
            print(f"Trajectory visualization error: {e}")
            return None

    def _plot_observed_trajectory(self, ax, observed_trajectory: List[TrajectoryPoint]):
        """Plot the observed trajectory."""
        obs_x = [p.px for p in observed_trajectory]
        obs_y = [p.py for p in observed_trajectory]

        ax.plot(obs_x, obs_y, 'r-', linewidth=3, label='Observed Trajectory', zorder=10)
        ax.scatter(obs_x[0], obs_y[0], color='red', s=100, marker='o',
                  label='Start Point', zorder=11)
        ax.scatter(obs_x[-1], obs_y[-1], color='red', s=100, marker='s',
                  label='End Point', zorder=11)

    def _plot_predicted_trajectories(self, ax, predicted_trajectories: List[List[TrajectoryPoint]],
                                   log_likelihoods: np.ndarray):
        """Plot predicted trajectories."""
        likelihoods = np.exp(log_likelihoods)
        norm = Normalize(vmin=np.min(likelihoods), vmax=np.max(likelihoods))

        colors = plt.cm.tab20(np.linspace(0, 1, len(predicted_trajectories)))

        for i, pred_traj in enumerate(predicted_trajectories):
            if len(pred_traj) == 0:
                continue

            pred_x = [p.px for p in pred_traj]
            pred_y = [p.py for p in pred_traj]

            likelihood = likelihoods[i]

            color = colors[i % len(colors)]
            alpha = 1

            particle_label = f'Particle {i+1} (L={likelihood:.2e})'

            ax.plot(pred_x, pred_y, color=color, alpha=alpha, linewidth=2,
                   label=particle_label, linestyle='-')

            ax.scatter(pred_x[0], pred_y[0], color=color, s=30, marker='o', alpha=alpha)
            ax.scatter(pred_x[-1], pred_y[-1], color=color, s=30, marker='s', alpha=alpha)

    def _should_show_likelihood_label(self, particle_idx: int, likelihood: float,
                                    all_likelihoods: np.ndarray) -> bool:
        """Return whether to show a likelihood label for a particle."""
        return particle_idx < 10 or likelihood > np.percentile(all_likelihoods, 90)

    def _add_likelihood_annotation(self, ax, x: float, y: float,
                                 likelihood: float, color):
        """Add a likelihood annotation."""
        ax.annotate(f'{likelihood:.2e}',
                   (x, y),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7,
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor=color, alpha=0.5))

    def _setup_plot_appearance(self, ax, window_idx: int, window_start: int,
                              title_suffix: str = ""):
        """Configure plot appearance."""
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)

        title = f'Trajectory Visualization - Window {window_idx} (Start Index: {window_start})'
        if title_suffix:
            title += f' {title_suffix}'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                          fontsize=9, frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(0.9)

        ax.axis('equal')

    def _add_statistics_text(self, ax, predicted_trajectories: List[List[TrajectoryPoint]],
                           log_likelihoods: np.ndarray):
        """Add a statistics text box."""
        likelihoods = np.exp(log_likelihoods)

        stats_text = (f'Particles: {len(predicted_trajectories)}\n'
                     f'Likelihood Range: [{np.min(likelihoods):.2e}, {np.max(likelihoods):.2e}]\n'
                     f'Likelihood Mean: {np.mean(likelihoods):.2e}\n'
                     f'Likelihood Std: {np.std(likelihoods):.2e}')

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

    def _add_colorbar(self, ax, log_likelihoods: np.ndarray):
        """Add a colorbar."""
        likelihoods = np.exp(log_likelihoods)
        norm = Normalize(vmin=np.min(likelihoods), vmax=np.max(likelihoods))
        colormap = cm.viridis

        sm = cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Likelihood', fontsize=12)

    def _save_figure(self, window_idx: int) -> str:
        """Save a figure file."""
        filename = f'trajectories_window_{window_idx:04d}.pdf'
        filepath = os.path.join(self.viz_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', format='pdf')
        print(f"Trajectory visualization saved: {filepath}")
        return filepath

    def create_summary_visualization(self,
                                   estimation_history: List[dict],
                                   param_names: Optional[List[str]] = None) -> Optional[str]:
        """
        Create a summary visualization for parameter estimation.
        """
        if not self.enable_visualization or not estimation_history:
            return None

        try:
            if param_names is None:
                param_names = list(estimation_history[0]['mean'].keys())

            fig, axes = plt.subplots(len(param_names), 2,
                                   figsize=(12, 4*len(param_names)))
            if len(param_names) == 1:
                axes = axes.reshape(1, -1)

            windows = [h['window_idx'] for h in estimation_history]

            for i, param in enumerate(param_names):
                self._plot_parameter_evolution(axes[i], windows, param, estimation_history)

            plt.tight_layout()

            filepath = os.path.join(self.viz_dir, 'parameter_evolution_summary.pdf')
            plt.savefig(filepath, bbox_inches='tight', format='pdf')
            plt.close()

            print(f"Parameter evolution summary saved: {filepath}")
            return filepath

        except Exception as e:
            print(f"Summary visualization error: {e}")
            return None

    def _plot_parameter_evolution(self, axes_row, windows: List[int],
                                param: str, estimation_history: List[dict]):
        """Plot the evolution of a single parameter."""
        means = [h['mean'][param] for h in estimation_history]
        stds = [h['std'][param] for h in estimation_history]
        maps = [h['map'][param] for h in estimation_history]
        ci_lowers = [h['ci_95'][param][0] for h in estimation_history]
        ci_uppers = [h['ci_95'][param][1] for h in estimation_history]

        ax1 = axes_row[0]
        ax1.plot(windows, means, 'b-', label='Mean', linewidth=2)
        ax1.plot(windows, maps, 'r--', label='MAP', linewidth=1)
        ax1.fill_between(windows, ci_lowers, ci_uppers, alpha=0.3, color='blue')
        ax1.set_xlabel('Window Index')
        ax1.set_ylabel(param)
        ax1.set_title(f'{param} - Estimation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes_row[1]
        ax2.plot(windows, stds, 'g-', linewidth=2)
        ax2.set_xlabel('Window Index')
        ax2.set_ylabel('Std Dev')
        ax2.set_title(f'{param} - Uncertainty')
        ax2.grid(True, alpha=0.3)

    def visualize_comprehensive_trajectories(self,
                                           observed_trajectory: List[TrajectoryPoint],
                                           predicted_trajectories: List[List[TrajectoryPoint]],
                                           log_likelihoods: np.ndarray,
                           future_predicted_trajectory: Optional[List[TrajectoryPoint]],
                           future_real_trajectory: Optional[List[TrajectoryPoint]],
                           window_idx: int,
                           window_start: int,
                           prediction_start_time: Optional[float] = None,
                           title_suffix: str = "",
                           background_vehicles_data: Optional[List[Dict]] = None,
                           all_future_predicted_trajectories: Optional[List[List[TrajectoryPoint]]] = None,
                           all_future_weights: Optional[List[float]] = None,
                           all_future_collision_infos: Optional[List[Dict]] = None,
                           if_collision: bool = False) -> Optional[str]:
        """
        Create a comprehensive visualization that includes identification,
        future prediction, and ground-truth future segments.
        """

        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#f8f9fa')

        # 1. Plot the observation window.
        self._plot_observed_trajectory_comprehensive(ax, observed_trajectory, label_prefix="ID Window")

        # 2. Plot identification trajectories.
        self._plot_identification_trajectories(ax, predicted_trajectories, log_likelihoods)

        # 3. Plot the ground-truth future segment.
        if future_real_trajectory and len(future_real_trajectory) > 0:
            self._plot_future_real_trajectory(ax, future_real_trajectory)

        # 4. Plot all-particle future predictions if enabled.
        if all_future_predicted_trajectories is not None and all_future_weights is not None:
            self._plot_all_particles_future_trajectories(ax, all_future_predicted_trajectories, all_future_weights, all_future_collision_infos)

        # 5. Plot the best-particle future prediction.
        if future_predicted_trajectory and len(future_predicted_trajectory) > 0:
            self._plot_future_predicted_trajectory(ax, future_predicted_trajectory,if_collision)

        # 6. Plot background vehicles and collect plot bounds.
        trajectory_bounds = self._plot_background_vehicles(ax, background_vehicles_data, observed_trajectory,
                                    future_predicted_trajectory, future_real_trajectory)

        # Use the trajectory bounds for the visible range.
        ax.set_xlim(trajectory_bounds['x_min'], trajectory_bounds['x_max'])
        ax.set_ylim(trajectory_bounds['y_min'], trajectory_bounds['y_max'])

        # 7. Draw time markers.
        self._plot_time_markers(ax, observed_trajectory, future_predicted_trajectory,
                                future_real_trajectory, prediction_start_time)

        # 8. Finalize plot appearance.
        self._setup_comprehensive_plot_appearance(ax, window_idx, window_start, title_suffix)

        plt.tight_layout()

        filepath = self._save_comprehensive_figure(window_idx)

        plt.close()

        return filepath



    def _plot_observed_trajectory_comprehensive(self, ax, observed_trajectory: List[TrajectoryPoint],
                                              label_prefix: str = "Observed"):
        """Plot the observed trajectory for the identification window."""
        obs_x = [p.px for p in observed_trajectory]
        obs_y = [p.py for p in observed_trajectory]

        ax.plot(obs_x, obs_y, color='#2E86AB', linewidth=4, label=f'{label_prefix} Trajectory (k-r to k)',
               zorder=10, alpha=0.9)

        ax.scatter(obs_x[0], obs_y[0], color='#2E86AB', s=150, marker='o',
                  label=f'{label_prefix} Start', zorder=12, edgecolor='white', linewidth=2)
        ax.scatter(obs_x[-1], obs_y[-1], color='#2E86AB', s=150, marker='s',
                  label=f'{label_prefix} End (Prediction Start)', zorder=12, edgecolor='white', linewidth=2)

    def _plot_identification_trajectories(self, ax, predicted_trajectories: List[List[TrajectoryPoint]],
                                        log_likelihoods: np.ndarray):
        """Plot identification trajectories for the best particles."""
        if len(predicted_trajectories) == 0:
            return

        # Show only a few best particles to keep the figure readable.
        num_to_show = min(5, len(predicted_trajectories))
        best_indices = np.argsort(log_likelihoods)[-num_to_show:]

        # Use gray tones for identification trajectories.
        colors = ['#95A5A6', '#7F8C8D', '#BDC3C7', '#85929E', '#AEB6BF']

        for i, idx in enumerate(best_indices):
            pred_traj = predicted_trajectories[idx]
            if len(pred_traj) == 0:
                continue

            pred_x = [p.px for p in pred_traj]
            pred_y = [p.py for p in pred_traj]

            likelihood = np.exp(log_likelihoods[idx])
            color = colors[i % len(colors)]

            label = 'ID Prediction (Best Particles)' if i == 0 else None

            ax.plot(pred_x, pred_y, color=color, alpha=0.6, linewidth=1.5,
                   label=label, linestyle='--', zorder=5)

    def _plot_future_predicted_trajectory(self, ax, future_predicted_trajectory: List[TrajectoryPoint], if_collision: bool=False):
        """Plot the future predicted trajectory."""
        pred_x = [p.px for p in future_predicted_trajectory]
        pred_y = [p.py for p in future_predicted_trajectory]

        ax.plot(pred_x, pred_y, color='#F39C12', linewidth=3, label='Future Prediction (Best Particle)',
               zorder=8, alpha=0.9, linestyle='-')

        ax.scatter(pred_x[-1], pred_y[-1], color='#F39C12', s=120, marker='D',
                  label='Prediction End', zorder=10, edgecolor='white', linewidth=2)
        if if_collision:
            ax.scatter(pred_x[-1], pred_y[-1], color='red', s=120, marker='x',
                    label='Collision', zorder=11, edgecolor='white', linewidth=2)


    def _plot_future_real_trajectory(self, ax, future_real_trajectory: List[TrajectoryPoint]):
        """Plot the ground-truth future trajectory segment."""
        real_x = [p.px for p in future_real_trajectory]
        real_y = [p.py for p in future_real_trajectory]

        ax.plot(real_x, real_y, color='#27AE60', linewidth=3, label='Ground Truth Future',
               zorder=7, alpha=0.9, linestyle='-')

        ax.scatter(real_x[-1], real_y[-1], color='#27AE60', s=120, marker='*',
                  label='Ground Truth End', zorder=11, edgecolor='white', linewidth=2)

    def _plot_all_particles_future_trajectories(self, ax, all_future_predicted_trajectories: List[List[TrajectoryPoint]],
                                              all_future_weights: List[float],
                                              all_collision_infos: Optional[List[Dict]] = None):
        """Plot all-particle future trajectories and mark collision endpoints."""
        if len(all_future_predicted_trajectories) == 0:
            return

        sorted_indices = np.argsort(all_future_weights)[::-1]

        num_particles = len(all_future_predicted_trajectories)
        max_particles_to_show = min(20, num_particles)

        base_color = np.array([155, 89, 182])  # Purple (#9B59B6).

        successful_trajectories = 0
        for i, idx in enumerate(sorted_indices[:max_particles_to_show]):
            trajectory = all_future_predicted_trajectories[idx]
            weight = all_future_weights[idx]

            if len(trajectory) == 0:
                continue

            pred_x = [p.px for p in trajectory]
            pred_y = [p.py for p in trajectory]

            alpha = 0.2 + 0.6 * (weight / max(all_future_weights)) if max(all_future_weights) > 0 else 0.3
            alpha = max(0.15, min(0.8, alpha))

            color_factor = 0.5 + 0.5 * (1 - i / max_particles_to_show)
            color = base_color * color_factor / 255.0

            line_width = 1.5 if i < 5 else 1.0
            ax.plot(pred_x, pred_y, color=color, linewidth=line_width,
                   alpha=alpha, zorder=5, linestyle='-')

            if (all_collision_infos and idx < len(all_collision_infos) and
                all_collision_infos[idx] and all_collision_infos[idx].get('collision_detected', False)):
                crash_marker_color = '#E74C3C'
                ax.scatter(pred_x[-1], pred_y[-1], color=crash_marker_color, s=80,
                          marker='X', zorder=10, alpha=min(1.0, alpha + 0.3))

                if not hasattr(self, '_collision_legend_added'):
                    ax.scatter([], [], color=crash_marker_color, s=80, marker='X',
                              label='Collision End', zorder=10)
                    self._collision_legend_added = True

            successful_trajectories += 1

        if successful_trajectories > 0:
            legend_color = base_color * 0.8 / 255.0
            ax.plot([], [], color=legend_color, linewidth=2, alpha=0.6,
                   label=f'All Particles Future ({successful_trajectories}/{num_particles} shown)')

    def _plot_time_markers(self, ax, observed_trajectory: List[TrajectoryPoint],
                          future_predicted_trajectory: Optional[List[TrajectoryPoint]],
                          future_real_trajectory: Optional[List[TrajectoryPoint]],
                          prediction_start_time: Optional[float]):
        """Plot time-separation markers."""
        if len(observed_trajectory) == 0:
            return

        start_x, start_y = observed_trajectory[-1].px, observed_trajectory[-1].py

        if future_predicted_trajectory or future_real_trajectory:
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0] if ax.get_xlim() != (0.0, 1.0) else 50
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0] if ax.get_ylim() != (0.0, 1.0) else 50
            line_length = min(x_range, y_range) * 0.1

            ax.axvline(x=start_x, color='red', linestyle=':', alpha=0.7, linewidth=2,
                      label='Prediction Start Line', zorder=3)

    def _plot_background_vehicles(self, ax, background_vehicles_data: List[Dict],
                                observed_trajectory: List[TrajectoryPoint],
                                future_predicted_trajectory: Optional[List[TrajectoryPoint]],
                                future_real_trajectory: Optional[List[TrajectoryPoint]]):
        """
        Plot background vehicles and return trajectory bounds.
        """
        if not background_vehicles_data or len(background_vehicles_data) == 0:
            return

        trajectory_bounds = self._calculate_trajectory_bounds(
            observed_trajectory, future_predicted_trajectory, future_real_trajectory
        )


        if trajectory_bounds is None:
            return

        filtered_vehicles, xy_bounds = self._filter_vehicles_by_boundary(
            background_vehicles_data, trajectory_bounds
        )

        if len(filtered_vehicles) == 0:
            return

        vehicle_colors = ['#8E44AD', '#E67E22', '#16A085', '#E74C3C', '#3498DB']

        for i, vehicle_data in enumerate(filtered_vehicles):
            color = vehicle_colors[i % len(vehicle_colors)]


            if 'positions' in vehicle_data and len(vehicle_data['positions']) > 0:
                px, py, timestamp = vehicle_data['positions'][0]

                ax.scatter(px, py,
                          color=color, s=60, marker='^', alpha=0.8,
                          label='Background Vehicles' if i == 0 else None,
                          zorder=4, edgecolor='white', linewidth=1)

                if 'vehicle_id' in vehicle_data:
                    ax.annotate(f"V{vehicle_data['vehicle_id']}",
                              (px, py),
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=8, alpha=0.7, color=color,
                              fontweight='bold')

        return xy_bounds

    def _calculate_trajectory_bounds(self, observed_trajectory: List[TrajectoryPoint],
                                   future_predicted_trajectory: Optional[List[TrajectoryPoint]],
                                   future_real_trajectory: Optional[List[TrajectoryPoint]]) -> Optional[Dict]:
        """
        Compute the bounds of all trajectories.
        """
        all_x = []
        all_y = []

        if observed_trajectory:
            all_x.extend([p.px for p in observed_trajectory])
            all_y.extend([p.py for p in observed_trajectory])

        if future_predicted_trajectory:
            all_x.extend([p.px for p in future_predicted_trajectory])
            all_y.extend([p.py for p in future_predicted_trajectory])

        if future_real_trajectory:
            all_x.extend([p.px for p in future_real_trajectory])
            all_y.extend([p.py for p in future_real_trajectory])

        if len(all_x) == 0:
            return None

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        return {
            'x_min': x_min,
            'x_max': x_max ,
            'y_min': y_min ,
            'y_max': y_max
        }

    def _filter_vehicles_by_boundary(self, background_vehicles_data: List[Dict],
                                   trajectory_bounds: Dict) -> List[Dict]:
        """
        Filter background vehicles by trajectory bounds.
        """
        # Extend the bounds by fixed buffers.
        x_buffer = 20.0
        y_buffer = 10.0

        filter_x_min = trajectory_bounds['x_min'] - x_buffer
        filter_x_max = trajectory_bounds['x_max'] + x_buffer
        filter_y_min = trajectory_bounds['y_min'] - y_buffer
        filter_y_max = trajectory_bounds['y_max'] + y_buffer

        filtered_vehicles = []

        for vehicle_data in background_vehicles_data:
            if 'positions' in vehicle_data and len(vehicle_data['positions']) > 0:
                px, py, timestamp = vehicle_data['positions'][0]

                if (filter_x_min <= px <= filter_x_max and
                    filter_y_min <= py <= filter_y_max):
                    filtered_vehicles.append(vehicle_data)

        xy_bounds = {
            'x_min': filter_x_min,
            'x_max': filter_x_max,
            'y_min': filter_y_min,
            'y_max': filter_y_max
        }
        return filtered_vehicles, xy_bounds

    def _setup_comprehensive_plot_appearance(self, ax, window_idx: int, window_start: int,
                                          title_suffix: str = ""):
        """Configure appearance for the comprehensive plot."""
        ax.set_xlabel('X Position (m)', fontsize=14, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Y Position (m)', fontsize=14, fontweight='bold', color='#2c3e50')

        title = (f'Comprehensive Trajectory Visualization - Window {window_idx}\n'
                f'Parameter Identification + Future Prediction')
        if title_suffix:
            title += f' {title_suffix}'

        ax.set_title(title, fontsize=16, fontweight='bold', color='#2c3e50', pad=20)

        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#bdc3c7')
        ax.set_axisbelow(True)

        legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),
                          fontsize=11, frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#bdc3c7')

        ax.set_aspect('equal', adjustable='box')

    def _add_comprehensive_statistics_text(self, ax, predicted_trajectories: List[List[TrajectoryPoint]],
                                         log_likelihoods: np.ndarray,
                                         future_predicted_trajectory: Optional[List[TrajectoryPoint]],
                                         future_real_trajectory: Optional[List[TrajectoryPoint]]):
        """Add a comprehensive statistics text box."""
        likelihoods = np.exp(log_likelihoods)

        stats_lines = [
            f'ID Particles: {len(predicted_trajectories)}',
            f'Likelihood Range: [{np.min(likelihoods):.2e}, {np.max(likelihoods):.2e}]',
            f'Best Likelihood: {np.max(likelihoods):.2e}',
        ]

        if future_predicted_trajectory:
            stats_lines.append(f'Future Pred Steps: {len(future_predicted_trajectory)}')

        if future_real_trajectory:
            stats_lines.append(f'Ground Truth Steps: {len(future_real_trajectory)}')

        if future_predicted_trajectory and future_real_trajectory:
            min_len = min(len(future_predicted_trajectory), len(future_real_trajectory))
            if min_len > 0:
                pred_positions = np.array([(p.px, p.py) for p in future_predicted_trajectory[:min_len]])
                real_positions = np.array([(p.px, p.py) for p in future_real_trajectory[:min_len]])
                errors = np.linalg.norm(pred_positions - real_positions, axis=1)
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                stats_lines.extend([
                    f'Mean Pred Error: {mean_error:.3f}m',
                    f'Max Pred Error: {max_error:.3f}m'
                ])

        stats_text = '\n'.join(stats_lines)

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9,
                        edgecolor='#bdc3c7', linewidth=1))

    def _save_comprehensive_figure(self, window_idx: int) -> str:
        """Save the comprehensive figure."""
        filename = f'trajectories_window_all_{window_idx:04d}.pdf'
        filepath = os.path.join(self.viz_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', format='pdf',
                   facecolor='white', edgecolor='none')
        print(f"Comprehensive trajectory visualization saved: {filepath}")
        return filepath

    def visualize_nll_and_rmse(self,
                             nll_history: List[Dict],
                             rmse_history: List[Dict],
                             window_size: int = 10,
                             h_steps: int = 5) -> Optional[str]:
        """
        Create rolling 1-step NLL and H-step RMSE visualizations.
        """
        if not self.enable_visualization:
            return None

        if not nll_history and not rmse_history:
            print("No NLL or RMSE data available for visualization")
            return None

        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
            fig.patch.set_facecolor('white')

            # 1. Plot rolling 1-step NLL.
            if nll_history:
                ax_nll = axes[0]
                self._plot_nll_curve(ax_nll, nll_history, window_size)
            else:
                axes[0].text(0.5, 0.5, 'No NLL data available',
                           transform=axes[0].transAxes, ha='center', va='center',
                           fontsize=14, alpha=0.6)
                axes[0].set_title('1-step NLL (No Data)', fontsize=14, fontweight='bold')

            # 2. Plot H-step RMSE.
            if rmse_history:
                ax_rmse = axes[1]
                self._plot_rmse_curves(ax_rmse, rmse_history, h_steps)
            else:
                axes[1].text(0.5, 0.5, f'No {h_steps}-step RMSE data available',
                           transform=axes[1].transAxes, ha='center', va='center',
                           fontsize=14, alpha=0.6)
                axes[1].set_title(f'{h_steps}-step RMSE (No Data)', fontsize=14, fontweight='bold')

            plt.tight_layout(pad=3.0)

            filepath = os.path.join(self.viz_dir, 'nll_rmse_evaluation.pdf')
            plt.savefig(filepath, bbox_inches='tight', format='pdf',
                       facecolor='white', edgecolor='none')
            plt.close()

            print(f"Saved NLL and RMSE evaluation plots: {filepath}")
            return filepath

        except Exception as e:
            print(f"NLL/RMSE visualization error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _plot_nll_curve(self, ax, nll_history: List[Dict], window_size: int):
        """Plot the rolling 1-step NLL curve."""
        if not nll_history:
            return

        windows = [item['window_idx'] for item in nll_history]
        timestamps = [item['timestamp'] for item in nll_history]
        nll_values = [item['nll'] for item in nll_history]

        # Filter invalid values.
        valid_indices = [i for i, nll in enumerate(nll_values)
                        if np.isfinite(nll) and nll < 1000]

        if len(valid_indices) == 0:
            ax.text(0.5, 0.5, 'All NLL values are invalid',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, alpha=0.6)
            ax.set_title('1-step NLL (Invalid Data)', fontsize=14, fontweight='bold')
            return

        valid_windows = [windows[i] for i in valid_indices]
        valid_nll = [nll_values[i] for i in valid_indices]

        ax.plot(valid_windows, valid_nll, 'b-', alpha=0.3, linewidth=1,
               label='Raw NLL', zorder=1)

        # Compute and plot a moving average.
        if len(valid_nll) >= window_size:
            smoothed_nll = self._moving_average(valid_nll, window_size)
            smoothed_windows = valid_windows[window_size-1:]

            ax.plot(smoothed_windows, smoothed_nll, 'r-', linewidth=3,
                   label=f'{window_size}-point Moving Average', zorder=3)

        ax.set_xlabel('Window Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Negative Log-Likelihood', fontsize=12, fontweight='bold')
        ax.set_title('Rolling 1-step NLL (Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        mean_nll = np.mean(valid_nll)
        std_nll = np.std(valid_nll)
        min_nll = np.min(valid_nll)

        stats_text = (f'Mean: {mean_nll:.3f}\n'
                     f'Std: {std_nll:.3f}\n'
                     f'Min: {min_nll:.3f}\n'
                     f'Valid Points: {len(valid_nll)}/{len(nll_history)}')

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    def _plot_rmse_curves(self, ax, rmse_history: List[Dict], h_steps: int):
        """Plot H-step RMSE curves."""
        if not rmse_history:
            return

        windows = [item['window_idx'] for item in rmse_history]
        rmse_position = [item.get('rmse_position', float('inf')) for item in rmse_history]
        rmse_velocity = [item.get('rmse_velocity', float('inf')) for item in rmse_history]
        rmse_px = [item.get('rmse_px', float('inf')) for item in rmse_history]
        rmse_py = [item.get('rmse_py', float('inf')) for item in rmse_history]
        rmse_weighted = [item.get('rmse_weighted', float('inf')) for item in rmse_history]

        def filter_valid_rmse(values):
            return [v if np.isfinite(v) and v < 100 else np.nan for v in values]

        rmse_position = filter_valid_rmse(rmse_position)
        rmse_velocity = filter_valid_rmse(rmse_velocity)
        rmse_weighted = filter_valid_rmse(rmse_weighted)

        ax.plot(windows, rmse_position, 'g-', linewidth=2.5, label='Position RMSE', marker='o', markersize=4)
        ax.plot(windows, rmse_velocity, 'b-', linewidth=2.5, label='Velocity RMSE', marker='s', markersize=4)
        ax.plot(windows, rmse_weighted, 'r-', linewidth=2.5, label='Weighted RMSE', marker='^', markersize=4)

        ax.set_xlabel('Window Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_title(f'{h_steps}-step RMSE (Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        valid_pos = [v for v in rmse_position if not np.isnan(v)]
        valid_vel = [v for v in rmse_velocity if not np.isnan(v)]
        valid_weighted = [v for v in rmse_weighted if not np.isnan(v)]

        if valid_pos and valid_vel:
            stats_text = (f'Pos RMSE: {np.mean(valid_pos):.3f}±{np.std(valid_pos):.3f}m\n'
                         f'Vel RMSE: {np.mean(valid_vel):.3f}±{np.std(valid_vel):.3f}m/s\n'
                         f'Valid Points: {len(valid_pos)}/{len(rmse_history)}')
        else:
            stats_text = f'Valid Points: 0/{len(rmse_history)}'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    def _moving_average(self, data: List[float], window_size: int) -> List[float]:
        """Compute a moving average."""
        if len(data) < window_size:
            return data

        smoothed = []
        for i in range(window_size - 1, len(data)):
            window_data = data[i - window_size + 1:i + 1]
            smoothed.append(np.mean(window_data))

        return smoothed

    def get_visualization_info(self) -> dict:
        """Return visualization settings and state."""
        return {
            'enable_visualization': self.enable_visualization,
            'visualization_interval': self.visualization_interval,
            'current_window_count': self.current_window_count,
            'output_directory': self.viz_dir if self.enable_visualization else None,
            'figure_size': self.figure_size,
            'dpi': self.dpi
        }


def create_trajectory_visualizer(output_dir: str,
                               enable_visualization: bool = False,
                               visualization_interval: int = 1,
                               **kwargs) -> TrajectoryVisualizer:
    """
    Factory function for creating a trajectory visualizer.
    """
    return TrajectoryVisualizer(
        output_dir=output_dir,
        enable_visualization=enable_visualization,
        visualization_interval=visualization_interval,
        **kwargs
    )
