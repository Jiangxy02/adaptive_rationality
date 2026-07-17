#!/usr/bin/env python3
"""Visualization mixin split from ppo_checkpoint_simulation_with_cog.py."""


import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use a non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict


class CognitiveVizMixin:

    def generate_cognitive_visualization(self, episode_data: Dict, save_dir: str = None) -> str:
        """
        Generate cognitive-module visualization charts

        Args:
            episode_data: episode statistics
            save_dir: output directory

        Returns:
            saved chart path
        """
        if not self.enable_cognitive_visualization or self.cognitive_viz_data is None:
            print("Cognitive visualization is disabled")
            return None

        if not self.cognitive_viz_data['step_count']:
            print("No visualization data available")
            return None

        # Create the output directory
        if save_dir is None:
            save_dir = "cognitive_visualization"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use English-capable fonts to avoid missing CJK glyphs.
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

        # Create the combined visualization figure with a 3x2 layout.
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Cognitive Modules Visualization Analysis - {timestamp}', fontsize=16, fontweight='bold')

        steps = self.cognitive_viz_data['step_count']

        # Report collected data coverage
        print(f"Cognitive visualization data: {len(steps)} steps, rewards={len(self.cognitive_viz_data['original_rewards'])}, observations={len(self.cognitive_viz_data['original_observations'])}")

        # 1. Cognitive bias strength over time
        ax1 = axes[0, 0]
        if self.cognitive_viz_data['bias_strength']:
            bias_strength = self.cognitive_viz_data['bias_strength']
            bias_applied = self.cognitive_viz_data['bias_applied']

            # Keep bias arrays aligned with the step axis.
            min_len = min(len(steps), len(bias_strength), len(bias_applied))
            if min_len > 0:
                steps_subset = steps[:min_len]
                bias_strength_subset = bias_strength[:min_len]
                bias_applied_subset = bias_applied[:min_len]

                ax1.plot(steps_subset, bias_strength_subset, 'r-', linewidth=2, label='Bias Strength')
                ax1.fill_between(steps_subset, 0, bias_strength_subset, where=[x for x in bias_applied_subset],
                               alpha=0.3, color='red', label='Bias Applied')
                ax1.set_title('Cognitive Bias Strength', fontweight='bold')
                ax1.set_xlabel('Steps')
                ax1.set_ylabel('Bias Strength')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'Insufficient Bias Data', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Cognitive Bias Strength', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'Cognitive Bias Module Disabled', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Cognitive Bias Strength', fontweight='bold')

        # 2. Reward comparison
        ax2 = axes[0, 1]
        if self.cognitive_viz_data['original_rewards'] and self.cognitive_viz_data['modified_rewards']:
            original_rewards = self.cognitive_viz_data['original_rewards']
            modified_rewards = self.cognitive_viz_data['modified_rewards']

            # Keep reward arrays aligned with the step axis.
            min_len = min(len(steps), len(original_rewards), len(modified_rewards))
            if min_len > 0:
                steps_subset = steps[:min_len]
                original_rewards_subset = original_rewards[:min_len]
                modified_rewards_subset = modified_rewards[:min_len]

                # Measure the reward difference magnitude.
                reward_diff = [abs(orig - mod) for orig, mod in zip(original_rewards_subset, modified_rewards_subset)]
                max_diff = max(reward_diff) if reward_diff else 0

                ax2.plot(steps_subset, original_rewards_subset, 'g-', linewidth=3, label='Original Reward', alpha=0.8, marker='o', markersize=4)
                ax2.plot(steps_subset, modified_rewards_subset, 'b-', linewidth=3, label='Modified Reward', marker='s', markersize=4)

                # Fill the area only when the difference is materially visible.
                if max_diff > 1e-6:  # Show only when the difference exceeds the threshold.
                    ax2.fill_between(steps_subset, original_rewards_subset, modified_rewards_subset,
                                   alpha=0.3, color='orange', label='Bias Effect')

                ax2.set_title('Reward Signal Comparison', fontweight='bold')
                ax2.set_xlabel('Steps')
                ax2.set_ylabel('Reward Value')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Insufficient Reward Data', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Reward Signal Comparison', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Insufficient Reward Data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Reward Signal Comparison', fontweight='bold')

        # 3. Cognitive delay steps
        ax3 = axes[1, 0]
        if self.cognitive_viz_data['delay_steps']:
            delay_steps = self.cognitive_viz_data['delay_steps']
            delay_applied = self.cognitive_viz_data['delay_applied']

            # Keep delay arrays aligned with the step axis.
            min_len = min(len(steps), len(delay_steps), len(delay_applied))
            if min_len > 0:
                steps_subset = steps[:min_len]
                delay_steps_subset = delay_steps[:min_len]
                delay_applied_subset = delay_applied[:min_len]

                ax3.plot(steps_subset, delay_steps_subset, 'purple', linewidth=2, marker='o', markersize=3, label='Delay Steps')
                ax3.fill_between(steps_subset, 0, delay_steps_subset, where=[x for x in delay_applied_subset],
                               alpha=0.3, color='purple', label='Delay Applied')
                ax3.set_title('Cognitive Delay Steps', fontweight='bold')
                ax3.set_xlabel('Steps')
                ax3.set_ylabel('Delay Steps')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'Insufficient Delay Data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Cognitive Delay Steps', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'Cognitive Delay Module Disabled', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Cognitive Delay Steps', fontweight='bold')

        # 4. Action comparison (steering)
        ax4 = axes[1, 1]
        if self.cognitive_viz_data['original_actions'] and self.cognitive_viz_data['delayed_actions']:
            original_actions = np.array(self.cognitive_viz_data['original_actions'])
            delayed_actions = np.array(self.cognitive_viz_data['delayed_actions'])

            # Keep action arrays aligned with the step axis.
            min_len = min(len(steps), len(original_actions), len(delayed_actions))
            if min_len > 0:
                steps_subset = steps[:min_len]
                original_actions_subset = original_actions[:min_len]
                delayed_actions_subset = delayed_actions[:min_len]

                ax4.plot(steps_subset, original_actions_subset[:, 0], 'g-', linewidth=2, label='Original Steering', alpha=0.7)
                ax4.plot(steps_subset, delayed_actions_subset[:, 0], 'orange', linewidth=2, label='Delayed Steering')
                ax4.set_title('Steering Action Comparison', fontweight='bold')
                ax4.set_xlabel('Steps')
                ax4.set_ylabel('Steering Value')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Insufficient Action Data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Steering Action Comparison', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Insufficient Action Data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Steering Action Comparison', fontweight='bold')

        # 5. Perception noise level
        ax5 = axes[2, 0]
        if self.cognitive_viz_data['perception_noise']:
            perception_noise = self.cognitive_viz_data['perception_noise']
            perception_applied = self.cognitive_viz_data['perception_applied']

            # Keep perception-noise arrays aligned with the step axis.
            min_len = min(len(steps), len(perception_noise), len(perception_applied))
            if min_len > 0:
                steps_subset = steps[:min_len]
                perception_noise_subset = perception_noise[:min_len]
                perception_applied_subset = perception_applied[:min_len]

                ax5.plot(steps_subset, perception_noise_subset, 'cyan', linewidth=2, label='Front Beam Noise', marker='o', markersize=3)
                ax5.fill_between(steps_subset, 0, perception_noise_subset, where=[x for x in perception_applied_subset],
                               alpha=0.3, color='cyan', label='Noise Applied')
                ax5.set_title('Front Radar Beam Noise Level (Real-time)', fontweight='bold')
                ax5.set_xlabel('Steps')
                ax5.set_ylabel('Noise Magnitude (meters)')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
            else:
                ax5.text(0.5, 0.5, 'Insufficient Perception Data', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Front Radar Beam Noise Level (Real-time)', fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'Cognitive Perception Module Disabled', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Front Radar Beam Noise Level (Real-time)', fontweight='bold')

        # 6. Observation comparison
        ax6 = axes[2, 1]
        if self.cognitive_viz_data['original_observations'] and self.cognitive_viz_data['noisy_observations']:
            original_obs = self.cognitive_viz_data['original_observations']
            noisy_obs = self.cognitive_viz_data['noisy_observations']

            # Keep observation arrays aligned with the step axis.
            min_len = min(len(steps), len(original_obs), len(noisy_obs))
            if min_len > 0:
                steps_subset = steps[:min_len]
                original_obs_subset = original_obs[:min_len]
                noisy_obs_subset = noisy_obs[:min_len]

                # Measure the observation difference magnitude.
                obs_diff = [abs(orig - noise) for orig, noise in zip(original_obs_subset, noisy_obs_subset)]
                max_diff = max(obs_diff) if obs_diff else 0

                ax6.plot(steps_subset, original_obs_subset, 'g-', linewidth=3, label='Original Distance', alpha=0.8, marker='o', markersize=4)
                ax6.plot(steps_subset, noisy_obs_subset, 'red', linewidth=3, label='Noisy Distance', marker='s', markersize=4)

                # Fill the area only when the difference is materially visible.
                if max_diff > 1e-6:  # Show only when the difference exceeds the threshold.
                    ax6.fill_between(steps_subset, original_obs_subset, noisy_obs_subset,
                                   alpha=0.3, color='yellow', label='Noise Effect')

                ax6.set_title('Front Radar Distance: Before vs After Noise', fontweight='bold')
                ax6.set_xlabel('Steps')
                ax6.set_ylabel('Distance (meters)')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'Insufficient Observation Data', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Front Radar Distance: Before vs After Noise', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'Insufficient Observation Data', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Front Radar Distance: Before vs After Noise', fontweight='bold')



        plt.tight_layout()

        # Save the chart
        viz_filename = f"cognitive_visualization_{timestamp}.png"
        viz_path = os.path.join(save_dir, viz_filename)
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        # Generate the summary report
        self._generate_cognitive_report(episode_data, save_dir, timestamp)

        print(f"Saved cognitive visualization: {viz_path}")
        return viz_path

    def _generate_cognitive_report(self, episode_data: Dict, save_dir: str, timestamp: str):
        """Generate the cognitive-module summary report"""
        report_path = os.path.join(save_dir, f"cognitive_report_{timestamp}.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Cognitive Module Analysis Report\n\n")
            f.write(f"**Generated at**: {timestamp}\n\n")

            # Episode Summary
            f.write(f"## Episode Summary\n\n")
            f.write(f"- **Total steps**: {episode_data.get('episode_length', 0)}\n")
            f.write(f"- **Total reward**: {episode_data.get('total_reward', 0):.3f}\n")
            f.write(f"- **Reached destination**: {'Yes' if episode_data.get('success', False) else 'No'}\n")
            f.write(f"- **Average speed**: {episode_data.get('avg_speed', 0):.2f} m/s\n\n")

            # Cognitive bias statistics
            if self.cognitive_viz_data['bias_strength']:
                bias_data = self.cognitive_viz_data['bias_strength']
                bias_applied_count = sum(self.cognitive_viz_data['bias_applied'])

                f.write(f"## Cognitive Bias Module\n\n")
                f.write(f"- **Bias activations**: {bias_applied_count}\n")
                f.write(f"- **Average bias strength**: {np.mean(bias_data):.4f}\n")
                f.write(f"- **Maximum bias strength**: {np.max(bias_data):.4f}\n")
                f.write(f"- **Bias activation rate**: {bias_applied_count/len(bias_data)*100:.1f}%\n\n")

            # Cognitive delay statistics
            if self.cognitive_viz_data['delay_steps']:
                delay_data = self.cognitive_viz_data['delay_steps']
                delay_applied_count = sum(self.cognitive_viz_data['delay_applied'])

                f.write(f"## Cognitive Delay Module\n\n")
                f.write(f"- **Delay activations**: {delay_applied_count}\n")
                f.write(f"- **Average delay steps**: {np.mean(delay_data):.2f}\n")
                f.write(f"- **Maximum delay steps**: {np.max(delay_data)}\n")
                f.write(f"- **Delay activation rate**: {delay_applied_count/len(delay_data)*100:.1f}%\n\n")

            # Cognitive perception statistics
            if self.cognitive_viz_data['perception_noise']:
                noise_data = self.cognitive_viz_data['perception_noise']
                perception_applied_count = sum(self.cognitive_viz_data['perception_applied'])

                f.write(f"## Cognitive Perception Module\n\n")
                f.write(f"- **Noise activations**: {perception_applied_count}\n")
                f.write(f"- **Average noise level**: {np.mean(noise_data):.4f}\n")
                f.write(f"- **Maximum noise level**: {np.max(noise_data):.4f}\n")
                f.write(f"- **Noise activation rate**: {perception_applied_count/len(noise_data)*100:.1f}%\n\n")

            # Impact analysis
            if (self.cognitive_viz_data['original_rewards'] and
                self.cognitive_viz_data['modified_rewards']):
                orig_rewards = np.array(self.cognitive_viz_data['original_rewards'])
                mod_rewards = np.array(self.cognitive_viz_data['modified_rewards'])
                reward_diff = mod_rewards - orig_rewards

                f.write(f"## Cognitive Impact Analysis\n\n")
                f.write(f"- **Average reward change**: {np.mean(reward_diff):.4f}\n")
                f.write(f"- **Largest negative impact**: {np.min(reward_diff):.4f}\n")
                f.write(f"- **Largest positive impact**: {np.max(reward_diff):.4f}\n")
                f.write(f"- **Change in reward standard deviation**: {np.std(mod_rewards) - np.std(orig_rewards):.4f}\n\n")



        print(f"Saved cognitive report: {report_path}")

    def clear_cognitive_visualization_data(self):
        """Clear cognitive visualization data"""
        if self.cognitive_viz_data:
            for key in self.cognitive_viz_data:
                self.cognitive_viz_data[key].clear()
            print("Cognitive visualization data cleared")
