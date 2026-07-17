#!/usr/bin/env python3
"""Speed-control visualization mixin split from ppo_checkpoint_simulation_with_cog.py."""


import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict


class SpeedControlVizMixin:

    def generate_speed_control_visualization(self, episode_data: Dict, save_dir: str = None) -> str:
        """
        Generate speed-control reward visualization charts

        Args:
            episode_data: episode statistics
            save_dir: output directory

        Returns:
            saved chart path
        """
        if not self.enable_speed_control_visualization or self.speed_control_viz_data is None:
            print("Speed-control visualization is disabled")
            return None

        if not self.speed_control_viz_data['step_count']:
            print("No speed-control visualization data available")
            return None

        # Create the output directory
        if save_dir is None:
            save_dir = "fig_cog/speed_control_visualization"
        else:
            save_dir = os.path.join("fig_cog", "speed_control_visualization")

        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use English-capable fonts to avoid missing CJK glyphs.
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Create the speed-control visualization figure with a 2x2 layout.
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Speed Control Reward Visualization Analysis - {timestamp}', fontsize=16, fontweight='bold')

        steps = self.speed_control_viz_data['step_count']

        # Report collected data coverage
        print(f"Speed-control visualization data: {len(steps)} steps")

        # 1. Speed-control reward submodule comparison
        ax1 = axes[0, 0]
        if (self.speed_control_viz_data['speed_control_tracking'] and
            self.speed_control_viz_data['speed_control_soft_wall'] and
            self.speed_control_viz_data['speed_control_behavior_guidance']):

            tracking_rewards = self.speed_control_viz_data['speed_control_tracking']
            soft_wall_rewards = self.speed_control_viz_data['speed_control_soft_wall']
            behavior_rewards = self.speed_control_viz_data['speed_control_behavior_guidance']

            # Keep arrays aligned with the step axis.
            min_len = min(len(steps), len(tracking_rewards), len(soft_wall_rewards), len(behavior_rewards))
            if min_len > 0:
                steps_subset = steps[:min_len]
                tracking_subset = tracking_rewards[:min_len]
                soft_wall_subset = soft_wall_rewards[:min_len]
                behavior_subset = behavior_rewards[:min_len]

                ax1.plot(steps_subset, tracking_subset, 'g-', linewidth=2, label='Speed Tracking', marker='o', markersize=3)
                ax1.plot(steps_subset, soft_wall_subset, 'r-', linewidth=2, label='Soft Wall', marker='s', markersize=3)
                ax1.plot(steps_subset, behavior_subset, 'b-', linewidth=2, label='Behavior Guidance', marker='^', markersize=3)

                ax1.set_title('Speed Control Reward Sub-modules', fontweight='bold')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Reward Value')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Insufficient Speed Control Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Speed Control Reward Sub-modules', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Speed Control Module Disabled', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Speed Control Reward Sub-modules', fontweight='bold')

        # 2. Total speed-control reward and vehicle speed
        ax2 = axes[0, 1]
        if (self.speed_control_viz_data['speed_control_total'] and
            self.speed_control_viz_data['vehicle_speeds'] and
            self.speed_control_viz_data['speed_references']):

            total_rewards = self.speed_control_viz_data['speed_control_total']
            vehicle_speeds = self.speed_control_viz_data['vehicle_speeds']
            speed_refs = self.speed_control_viz_data['speed_references']

            # Keep arrays aligned with the step axis.
            min_len = min(len(steps), len(total_rewards), len(vehicle_speeds), len(speed_refs))
            if min_len > 0:
                steps_subset = steps[:min_len]
                total_subset = total_rewards[:min_len]
                speeds_subset = vehicle_speeds[:min_len]
                refs_subset = speed_refs[:min_len]

                # Create twin y-axes.
                ax2_twin = ax2.twinx()

                # Left y-axis: total speed-control reward
                line1 = ax2.plot(steps_subset, total_subset, 'purple', linewidth=2, label='Total SC Reward', marker='o', markersize=3)
                ax2.set_ylabel('Total Speed Control Reward', color='purple')
                ax2.tick_params(axis='y', labelcolor='purple')

                # Right y-axis: vehicle speed and reference speed
                line2 = ax2_twin.plot(steps_subset, speeds_subset, 'orange', linewidth=2, label='Vehicle Speed', marker='s', markersize=3)
                line3 = ax2_twin.plot(steps_subset, refs_subset, 'cyan', linewidth=2, label='Reference Speed', marker='^', markersize=3, linestyle='--')
                ax2_twin.set_ylabel('Speed (m/s)', color='orange')
                ax2_twin.tick_params(axis='y', labelcolor='orange')

                # Merge the legends
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, loc='upper left')

                ax2.set_title('Speed Control Total Reward & Vehicle Speed', fontweight='bold')
                ax2.set_xlabel('Steps')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Insufficient Speed Control Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Speed Control Total Reward & Vehicle Speed', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Speed Control Module Disabled', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Speed Control Total Reward & Vehicle Speed', fontweight='bold')

        # 3. Speed deviation analysis
        ax3 = axes[1, 0]
        if self.speed_control_viz_data['speed_deviations']:
            deviations = self.speed_control_viz_data['speed_deviations']

            # Keep arrays aligned with the step axis.
            min_len = min(len(steps), len(deviations))
            if min_len > 0:
                steps_subset = steps[:min_len]
                deviations_subset = deviations[:min_len]

                ax3.plot(steps_subset, deviations_subset, 'red', linewidth=2, marker='o', markersize=3, label='Speed Deviation')
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Reference Line')

                ax3.set_title('Speed Deviation Analysis', fontweight='bold')
                ax3.set_xlabel('Steps')
                ax3.set_ylabel('Speed Deviation (m/s)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient Deviation Data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Speed Deviation Analysis', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Deviation Data Available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Speed Deviation Analysis', fontweight='bold')

        # 4. Speed-control reward statistics
        ax4 = axes[1, 1]
        if self.speed_control_viz_data['speed_control_total']:
            total_rewards = self.speed_control_viz_data['speed_control_total']
            tracking_rewards = self.speed_control_viz_data['speed_control_tracking']
            soft_wall_rewards = self.speed_control_viz_data['speed_control_soft_wall']
            behavior_rewards = self.speed_control_viz_data['speed_control_behavior_guidance']

            # Compute summary statistics
            total_mean = np.mean(total_rewards)
            tracking_mean = np.mean(tracking_rewards)
            soft_wall_mean = np.mean(soft_wall_rewards)
            behavior_mean = np.mean(behavior_rewards)

            # Create the bar chart
            categories = ['Total', 'Tracking', 'Soft Wall', 'Behavior']
            values = [total_mean, tracking_mean, soft_wall_mean, behavior_mean]
            colors = ['purple', 'green', 'red', 'blue']

            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_title('Average Speed Control Rewards', fontweight='bold')
            ax4.set_ylabel('Average Reward Value')
            ax4.grid(True, alpha=0.3)

            # Annotate each bar with its value.
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'No Reward Data Available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Average Speed Control Rewards', fontweight='bold')

        plt.tight_layout()

        # Save the chart
        viz_filename = f"speed_control_visualization_{timestamp}.png"
        viz_path = os.path.join(save_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Generate the summary report
        self._generate_speed_control_report(episode_data, save_dir, timestamp)

        print(f"Saved speed-control visualization: {viz_path}")
        return viz_path

    def _generate_speed_control_report(self, episode_data: Dict, save_dir: str, timestamp: str):
        """Generate the speed-control reward summary report"""
        report_path = os.path.join(save_dir, f"speed_control_report_{timestamp}.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Speed-Control Reward Analysis Report\n\n")
            f.write(f"**Generated at**: {timestamp}\n\n")

            # Episode Summary
            f.write(f"## Episode Summary\n\n")
            f.write(f"- **Total steps**: {episode_data.get('episode_length', 0)}\n")
            f.write(f"- **Total reward**: {episode_data.get('total_reward', 0):.3f}\n")
            f.write(f"- **Reached destination**: {'Yes' if episode_data.get('success', False) else 'No'}\n")
            f.write(f"- **Average speed**: {episode_data.get('avg_speed', 0):.2f} m/s\n\n")

            # Speed-control reward statistics
            if (self.speed_control_viz_data['speed_control_total'] and
                any(abs(x) > 1e-6 for x in self.speed_control_viz_data['speed_control_total'])):

                total_rewards = np.array(self.speed_control_viz_data['speed_control_total'])
                tracking_rewards = np.array(self.speed_control_viz_data['speed_control_tracking'])
                soft_wall_rewards = np.array(self.speed_control_viz_data['speed_control_soft_wall'])
                behavior_rewards = np.array(self.speed_control_viz_data['speed_control_behavior_guidance'])

                f.write(f"## Speed-Control Reward Analysis\n\n")
                f.write(f"- **Total speed-control reward**:\n")
                f.write(f"  - Mean: {np.mean(total_rewards):.4f}\n")
                f.write(f"  - Std: {np.std(total_rewards):.4f}\n")
                f.write(f"  - Min: {np.min(total_rewards):.4f}\n")
                f.write(f"  - Max: {np.max(total_rewards):.4f}\n\n")

                f.write(f"- **Speed-tracking submodule**:\n")
                f.write(f"  - Average reward: {np.mean(tracking_rewards):.4f}\n")
                f.write(f"  - Std: {np.std(tracking_rewards):.4f}\n\n")

                f.write(f"- **Soft speed-wall submodule**:\n")
                f.write(f"  - Average reward: {np.mean(soft_wall_rewards):.4f}\n")
                f.write(f"  - Std: {np.std(soft_wall_rewards):.4f}\n\n")

                f.write(f"- **Behavior-guidance submodule**:\n")
                f.write(f"  - Average reward: {np.mean(behavior_rewards):.4f}\n")
                f.write(f"  - Std: {np.std(behavior_rewards):.4f}\n\n")

                # Speed statistics
                if self.speed_control_viz_data['vehicle_speeds']:
                    speeds = np.array(self.speed_control_viz_data['vehicle_speeds'])
                    refs = np.array(self.speed_control_viz_data['speed_references'])
                    deviations = np.array(self.speed_control_viz_data['speed_deviations'])

                    f.write(f"- **Speed statistics**:\n")
                    f.write(f"  - Average vehicle speed: {np.mean(speeds):.2f} m/s\n")
                    f.write(f"  - Average reference speed: {np.mean(refs):.2f} m/s\n")
                    f.write(f"  - Average speed deviation: {np.mean(deviations):.2f} m/s\n")
                    f.write(f"  - Speed deviation std: {np.std(deviations):.2f} m/s\n\n")

        print(f"Saved speed-control report: {report_path}")

    def clear_speed_control_visualization_data(self):
        """Clear speed-control visualization data"""
        if self.speed_control_viz_data:
            for key in self.speed_control_viz_data:
                self.speed_control_viz_data[key].clear()
            print("Speed-control visualization data cleared")

    def _fill_default_speed_control_data(self):
        """Fill default speed-control data"""
        if self.speed_control_viz_data:
            self.speed_control_viz_data['speed_control_total'].append(0.0)
            self.speed_control_viz_data['speed_control_tracking'].append(0.0)
            self.speed_control_viz_data['speed_control_soft_wall'].append(0.0)
            self.speed_control_viz_data['speed_control_behavior_guidance'].append(0.0)
            self.speed_control_viz_data['vehicle_speeds'].append(0.0)
            self.speed_control_viz_data['speed_references'].append(0.0)
            self.speed_control_viz_data['speed_deviations'].append(0.0)
