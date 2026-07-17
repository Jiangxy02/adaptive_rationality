

import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
from datetime import datetime

_DELAY_STEPS_DEFAULT = 0
_DELAY_ENABLE_SMOOTHING_DEFAULT = False
_DELAY_SMOOTHING_FACTOR_DEFAULT = 0.3
_DELAY_ENABLE_VISUALIZATION_DEFAULT = True


class CognitiveDelayModule:
    """Simulate per-environment driver latency, applied only in PPO mode."""

    def __init__(
        self,
        delay_steps=_DELAY_STEPS_DEFAULT,
        enable_smoothing=_DELAY_ENABLE_SMOOTHING_DEFAULT,
        smoothing_factor=_DELAY_SMOOTHING_FACTOR_DEFAULT,
        enable_visualization=_DELAY_ENABLE_VISUALIZATION_DEFAULT
    ):
        self.delay_steps = delay_steps
        self.enable_smoothing = enable_smoothing
        self.smoothing_factor = smoothing_factor
        self.enable_visualization = enable_visualization



        self._buffers = {}
        self._previous_actions = {}
        self._last_commanded_actions = {}
        self._reset_delay_state(0)


        if self.enable_visualization:
            self._reset_visualization_data()

    def _reset_delay_state(self, env_id):
        env_id = int(env_id)
        self._buffers[env_id] = deque(maxlen=max(self.delay_steps + 1, 1))
        self._previous_actions[env_id] = np.zeros(2, dtype=np.float32)
        self._last_commanded_actions[env_id] = np.zeros(2, dtype=np.float32)

    def _get_delay_state(self, env_id):
        env_id = int(env_id)
        if env_id not in self._buffers:
            self._reset_delay_state(env_id)
        return (
            self._buffers[env_id],
            self._previous_actions[env_id],
            self._last_commanded_actions[env_id],
        )

    @property
    def buffer(self):
        return self._buffers[0]

    @property
    def previous_action(self):
        return self._previous_actions[0]

    @property
    def _last_commanded_action(self):
        return self._last_commanded_actions[0]

    def _reset_visualization_data(self):
        self.visualization_data = {
            'step': [],
            'original_action_throttle': [],
            'original_action_steering': [],
            'smoothed_action_throttle': [],
            'smoothed_action_steering': [],
            'delayed_action_throttle': [],
            'delayed_action_steering': [],
            'is_ppo_mode': [],
            'buffer_size': [],
            'env_id': []
        }
        self._step_count = 0

    def process_action(self, action, is_ppo_mode=True, env_id=0):
        """Delay PPO actions after optional smoothing and a zero-action warm-up.

        Non-PPO actions pass through unchanged. PPO mode emits zeros until the
        per-environment buffer contains ``delay_steps + 1`` commands.
        """
        env_id = int(env_id)
        buffer, previous_action, _ = self._get_delay_state(env_id)

        if not is_ppo_mode:

            if self.enable_visualization:
                self._record_visualization_data(
                    original_action=action,
                    smoothed_action=action,
                    delayed_action=action,
                    is_ppo_mode=False,
                    env_id=env_id,
                    buffer_size=len(buffer),
                )
            return action


        action = np.array(action, dtype=np.float32)


        self._last_commanded_actions[env_id] = action.copy()


        if self.enable_smoothing:
            smoothed_action = (
                self.smoothing_factor * action +
                (1 - self.smoothing_factor) * previous_action
            )
            self._previous_actions[env_id] = smoothed_action.copy()
        else:
            smoothed_action = action


        buffer.append(smoothed_action.copy())


        if self.delay_steps == 0:

            delayed_action = smoothed_action
        elif len(buffer) < self.delay_steps + 1:

            delayed_action = np.zeros_like(smoothed_action)
        else:


            delayed_action = buffer[0]


        if self.enable_visualization:
            self._record_visualization_data(
                original_action=action,
                smoothed_action=smoothed_action,
                delayed_action=delayed_action,
                is_ppo_mode=True,
                env_id=env_id,
                buffer_size=len(buffer),
            )

        return delayed_action

    def _record_visualization_data(
        self,
        original_action,
        smoothed_action,
        delayed_action,
        is_ppo_mode,
        env_id,
        buffer_size,
    ):
        self.visualization_data['step'].append(self._step_count)
        self.visualization_data['original_action_throttle'].append(float(original_action[0]))
        self.visualization_data['original_action_steering'].append(float(original_action[1]))
        self.visualization_data['smoothed_action_throttle'].append(float(smoothed_action[0]))
        self.visualization_data['smoothed_action_steering'].append(float(smoothed_action[1]))
        self.visualization_data['delayed_action_throttle'].append(float(delayed_action[0]))
        self.visualization_data['delayed_action_steering'].append(float(delayed_action[1]))
        self.visualization_data['is_ppo_mode'].append(is_ppo_mode)
        self.visualization_data['buffer_size'].append(buffer_size)
        self.visualization_data['env_id'].append(env_id)

        self._step_count += 1

    def reset(self, env_id=None):
        if env_id is None:
            self._buffers.clear()
            self._previous_actions.clear()
            self._last_commanded_actions.clear()
            self._reset_delay_state(0)
            if self.enable_visualization:
                self._reset_visualization_data()
            return

        self._reset_delay_state(env_id)

    def get_status(self, env_id=0):
        env_id = int(env_id)
        buffer, previous_action, last_commanded_action = self._get_delay_state(env_id)
        status = {
            'delay_steps': self.delay_steps,
            'enable_smoothing': self.enable_smoothing,
            'smoothing_factor': self.smoothing_factor,
            'buffer_size': len(buffer),
            'last_commanded_action': last_commanded_action.tolist(),
            'previous_action': previous_action.tolist(),
            'enable_visualization': self.enable_visualization,
            'env_id': env_id,
            'tracked_envs': sorted(self._buffers),
        }

        if self.enable_visualization:
            status['recorded_steps'] = len(self.visualization_data['step'])

        return status

    def update_config(self, **kwargs):
        if 'delay_steps' in kwargs:
            self.delay_steps = kwargs['delay_steps']
            for env_id, buffer in list(self._buffers.items()):
                self._buffers[env_id] = deque(
                    buffer,
                    maxlen=max(self.delay_steps + 1, 1),
                )


        if 'enable_smoothing' in kwargs:
            self.enable_smoothing = kwargs['enable_smoothing']


        if 'smoothing_factor' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']


        if 'enable_visualization' in kwargs:
            self.enable_visualization = kwargs['enable_visualization']
            if self.enable_visualization and not hasattr(self, 'visualization_data'):
                self._reset_visualization_data()


    def generate_delay_visualization(self, output_dir=None, session_name=None):
        if not self.enable_visualization or not self.visualization_data['step']:

            return None


        if output_dir is None:

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = "./outputs/fig_cog"
            output_dir = os.path.join(base_dir, f"cognitive_analysis_{timestamp}", "cognitive_delay")

        if session_name is None:
            session_name = f"delay_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


        os.makedirs(output_dir, exist_ok=True)


        steps = np.array(self.visualization_data['step'])
        original_throttle = np.array(self.visualization_data['original_action_throttle'])
        original_steering = np.array(self.visualization_data['original_action_steering'])
        smoothed_throttle = np.array(self.visualization_data['smoothed_action_throttle'])
        smoothed_steering = np.array(self.visualization_data['smoothed_action_steering'])
        delayed_throttle = np.array(self.visualization_data['delayed_action_throttle'])
        delayed_steering = np.array(self.visualization_data['delayed_action_steering'])
        is_ppo_mode = np.array(self.visualization_data['is_ppo_mode'])


        plt.style.use('default')

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        try:
            fig.suptitle(f'Cognitive Delay Module Effect Analysis - {session_name}', fontsize=16, fontweight='bold')
        except:
            fig.suptitle(f'Cognitive Delay Module Analysis - {session_name}', fontsize=16, fontweight='bold')


        ax1 = axes[0, 0]
        ax1.plot(steps, original_throttle, 'b-', linewidth=2, label='Original Action', alpha=0.8)
        if self.enable_smoothing:
            ax1.plot(steps, smoothed_throttle, 'g--', linewidth=1.5, label='Smoothed Action', alpha=0.7)
        ax1.plot(steps, delayed_throttle, 'r-', linewidth=2, label=f'Delayed Action (Delay {self.delay_steps} Steps)', alpha=0.8)


        ppo_regions = np.where(is_ppo_mode)[0]
        if len(ppo_regions) > 0:
            ax1.fill_between(steps, ax1.get_ylim()[0], ax1.get_ylim()[1],
                           where=is_ppo_mode, alpha=0.1, color='yellow', label='PPO Mode')

        try:
            ax1.set_title('Throttle/Brake Control Action Time Series', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Simulation Steps')
            ax1.set_ylabel('Action Value')
        except:
            ax1.set_title('Throttle/Brake Action Time Series', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Simulation Steps')
            ax1.set_ylabel('Action Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)


        ax2 = axes[0, 1]
        ax2.plot(steps, original_steering, 'b-', linewidth=2, label='Original Action', alpha=0.8)
        if self.enable_smoothing:
            ax2.plot(steps, smoothed_steering, 'g--', linewidth=1.5, label='Smoothed Action', alpha=0.7)
        ax2.plot(steps, delayed_steering, 'r-', linewidth=2, label=f'Delayed Action (Delay {self.delay_steps} Steps)', alpha=0.8)


        if len(ppo_regions) > 0:
            ax2.fill_between(steps, ax2.get_ylim()[0], ax2.get_ylim()[1],
                           where=is_ppo_mode, alpha=0.1, color='yellow', label='PPO Mode')

        try:
            ax2.set_title('Steering Control Action Time Series', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Simulation Steps')
            ax2.set_ylabel('Action Value')
        except:
            ax2.set_title('Steering Action Time Series', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Simulation Steps')
            ax2.set_ylabel('Action Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)


        ax3 = axes[1, 0]

        throttle_delay_diff = np.abs(original_throttle - delayed_throttle)
        steering_delay_diff = np.abs(original_steering - delayed_steering)

        ax3.plot(steps, throttle_delay_diff, 'r-', linewidth=2, label='Throttle Delay Difference', alpha=0.8)
        ax3.plot(steps, steering_delay_diff, 'b-', linewidth=2, label='Steering Delay Difference', alpha=0.8)


        if len(ppo_regions) > 0:
            ax3.fill_between(steps, 0, ax3.get_ylim()[1],
                           where=is_ppo_mode, alpha=0.1, color='yellow', label='PPO Mode')

        try:
            ax3.set_title('Delay Effect Difference Analysis', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Simulation Steps')
            ax3.set_ylabel('|Original Action - Delayed Action|')
        except:
            ax3.set_title('Delay Effect Difference Analysis', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Simulation Steps')
            ax3.set_ylabel('|Original - Delayed Action|')
        ax3.legend()
        ax3.grid(True, alpha=0.3)


        ax4 = axes[1, 1]
        ax4.axis('off')


        total_steps = len(steps)
        ppo_steps = np.sum(is_ppo_mode)
        avg_throttle_delay = np.mean(throttle_delay_diff[is_ppo_mode]) if ppo_steps > 0 else 0
        avg_steering_delay = np.mean(steering_delay_diff[is_ppo_mode]) if ppo_steps > 0 else 0
        max_throttle_delay = np.max(throttle_delay_diff) if len(throttle_delay_diff) > 0 else 0
        max_steering_delay = np.max(steering_delay_diff) if len(steering_delay_diff) > 0 else 0


        stats_text = f"""
Delay Module Statistics:

Configuration Parameters:
• Delay Steps: {self.delay_steps}
• Smoothing: {'Enabled' if self.enable_smoothing else 'Disabled'}
• Smoothing Factor: {self.smoothing_factor:.2f}

Running Statistics:
• Total Simulation Steps: {total_steps}
• PPO Mode Steps: {ppo_steps} ({ppo_steps/total_steps*100:.1f}%)
• Non-PPO Mode Steps: {total_steps - ppo_steps}

Delay Effects (PPO Mode):
• Average Throttle Delay: {avg_throttle_delay:.4f}
• Average Steering Delay: {avg_steering_delay:.4f}
• Maximum Throttle Delay: {max_throttle_delay:.4f}
• Maximum Steering Delay: {max_steering_delay:.4f}
        """

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))


        plt.tight_layout()


        output_file = os.path.join(output_dir, f"{session_name}_delay_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()




        self._generate_delay_report(output_dir, session_name, {
            'total_steps': total_steps,
            'ppo_steps': ppo_steps,
            'avg_throttle_delay': avg_throttle_delay,
            'avg_steering_delay': avg_steering_delay,
            'max_throttle_delay': max_throttle_delay,
            'max_steering_delay': max_steering_delay
        })

        return output_file

    def _generate_delay_report(self, output_dir, session_name, stats):
        report_file = os.path.join(output_dir, f"{session_name}_delay_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Cognitive Delay Module Analysis Report\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Session name: {session_name}\n")
            f.write("=" * 50 + "\n\n")

            f.write("Module configuration:\n")
            f.write(f"  Delay steps: {self.delay_steps}\n")
            f.write(f"  Action smoothing: {'Enabled' if self.enable_smoothing else 'Disabled'}\n")
            f.write(f"  Smoothing factor: {self.smoothing_factor:.3f}\n")
            f.write(f"  Visualization recording: {'Enabled' if self.enable_visualization else 'Disabled'}\n\n")

            f.write("Run statistics:\n")
            f.write(f"  Total simulation steps: {stats['total_steps']}\n")
            f.write(f"  PPO mode steps: {stats['ppo_steps']} ({stats['ppo_steps']/stats['total_steps']*100:.1f}%)\n")
            f.write(f"  Non-PPO mode steps: {stats['total_steps'] - stats['ppo_steps']}\n\n")

            f.write("Delay effect analysis (PPO mode):\n")
            f.write(f"  Average throttle/brake delay difference: {stats['avg_throttle_delay']:.6f}\n")
            f.write(f"  Average steering delay difference: {stats['avg_steering_delay']:.6f}\n")
            f.write(f"  Maximum throttle/brake delay difference: {stats['max_throttle_delay']:.6f}\n")
            f.write(f"  Maximum steering delay difference: {stats['max_steering_delay']:.6f}\n\n")

            f.write("Notes:\n")
            f.write("  - Delay difference = |original action - delayed action|\n")
            f.write("  - Delay effects are active only in PPO expert mode\n")
            f.write("  - Smoothing reduces abrupt control changes\n")
            f.write("  - Delay simulates human driver reaction latency\n")



    def get_visualization_data(self):
        if not self.enable_visualization:
            return None
        return self.visualization_data.copy()
