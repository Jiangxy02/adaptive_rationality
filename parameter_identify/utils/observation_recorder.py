#!/usr/bin/env python3


import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import os

class ObservationRecorder:


    def __init__(self, output_dir="observation_logs", session_name=None):

        self.output_dir = output_dir
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name = session_name


        os.makedirs(output_dir, exist_ok=True)


        self.step_data = []
        self.summary_stats = {}


        self.csv_path = os.path.join(output_dir, f"{session_name}_observations.csv")
        self.json_path = os.path.join(output_dir, f"{session_name}_observations.json")
        self.analysis_path = os.path.join(output_dir, f"{session_name}_analysis.txt")

        print(f"Observation recorder initialized")
        print(f"   Session name: {session_name}")
        print(f"   Output directory: {output_dir}")
        print(f"   CSV file: {self.csv_path}")
        print(f"   JSON file: {self.json_path}")
        print(f"   Analysis report: {self.analysis_path}")

    def record_step(self, env, action, action_info, obs, reward, info, step_count):

        try:

            step_record = {
                'step': step_count,
                'timestamp': datetime.now().isoformat(),
                'simulation_time': getattr(env, '_simulation_time', 0.0),
            }


            agent = env.agent
            step_record.update({

                'pos_x': float(agent.position[0]),
                'pos_y': float(agent.position[1]),
                'speed': float(agent.speed),
                'heading': float(agent.heading_theta),
                'velocity_x': float(agent.velocity[0]) if hasattr(agent, 'velocity') else 0.0,
                'velocity_y': float(agent.velocity[1]) if hasattr(agent, 'velocity') else 0.0,


                'on_lane': getattr(agent, 'on_lane', None),
                'out_of_road': getattr(agent, 'out_of_road', None),
                'dist_to_left_side': getattr(agent, 'dist_to_left_side', None),
                'dist_to_right_side': getattr(agent, 'dist_to_right_side', None),


                'crash_vehicle': getattr(agent, 'crash_vehicle', None),
                'crash_object': getattr(agent, 'crash_object', None),
                'crash_sidewalk': getattr(agent, 'crash_sidewalk', None),
            })


            if hasattr(agent, 'navigation') and agent.navigation:
                nav = agent.navigation
                step_record.update({
                    'nav_route_completion': getattr(nav, 'route_completion', 0.0),
                    'nav_distance_to_dest': getattr(nav, 'distance_to_destination', None),
                    'nav_current_lane': str(nav.current_lane.index) if nav.current_lane else None,
                    'nav_route_length': len(getattr(nav, 'route', [])),
                    'nav_checkpoints_count': len(getattr(nav, 'checkpoints', [])),
                })


                if nav.current_lane:
                    try:
                        long_pos, lat_pos = nav.current_lane.local_coordinates(agent.position)
                        step_record.update({
                            'lane_longitudinal_pos': float(long_pos),
                            'lane_lateral_pos': float(lat_pos),
                            'lane_length': float(nav.current_lane.length),
                        })
                    except:
                        step_record.update({
                            'lane_longitudinal_pos': None,
                            'lane_lateral_pos': None,
                            'lane_length': None,
                        })
            else:
                step_record.update({
                    'nav_route_completion': None,
                    'nav_distance_to_dest': None,
                    'nav_current_lane': None,
                    'nav_route_length': 0,
                    'nav_checkpoints_count': 0,
                    'lane_longitudinal_pos': None,
                    'lane_lateral_pos': None,
                    'lane_length': None,
                })


            if hasattr(env, 'custom_destination'):
                dest = env.custom_destination
                distance_to_custom = np.sqrt((agent.position[0] - dest[0])**2 + (agent.position[1] - dest[1])**2)
                step_record.update({
                    'custom_dest_x': float(dest[0]),
                    'custom_dest_y': float(dest[1]),
                    'distance_to_custom_dest': float(distance_to_custom),
                })
            else:
                step_record.update({
                    'custom_dest_x': None,
                    'custom_dest_y': None,
                    'distance_to_custom_dest': None,
                })


            step_record.update({
                'action_steering': float(action[0]) if len(action) > 0 else 0.0,
                'action_throttle': float(action[1]) if len(action) > 1 else 0.0,
                'action_source': action_info.get('source', 'unknown'),
                'action_success': action_info.get('success', None),
            })


            step_record.update({
                'reward': float(reward),
                'control_mode': info.get('Control Mode', 'unknown'),
                'expert_takeover': getattr(agent, 'expert_takeover', None),
            })


            if obs is not None:
                obs_array = np.array(obs)
                step_record.update({
                    'obs_shape': list(obs_array.shape),
                    'obs_mean': float(np.mean(obs_array)),
                    'obs_std': float(np.std(obs_array)),
                    'obs_min': float(np.min(obs_array)),
                    'obs_max': float(np.max(obs_array)),
                })


                if len(obs_array) >= 10:
                    step_record.update({
                        'obs_0_speed_related': float(obs_array[0]),
                        'obs_1_steering_related': float(obs_array[1]),
                        'obs_2': float(obs_array[2]),
                        'obs_3': float(obs_array[3]),
                        'obs_4': float(obs_array[4]),
                    })


                if len(obs_array) <= 100:
                    step_record['obs_full'] = obs_array.tolist()
            else:
                step_record.update({
                    'obs_shape': None,
                    'obs_mean': None,
                    'obs_std': None,
                    'obs_min': None,
                    'obs_max': None,
                })


            self.step_data.append(step_record)


            if len(self.step_data) % 100 == 0:
                self._save_data()

        except Exception as e:
            print(f" Error while recording step {step_count}: {e}")

    def _save_data(self):
        if not self.step_data:
            return

        try:

            df = pd.DataFrame(self.step_data)
            df.to_csv(self.csv_path, index=False)


            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.step_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f" Error while saving data: {e}")

    def finalize_recording(self):
        if not self.step_data:
            print(" No data was recorded")
            return

        print("Generating analysis report...")


        self._save_data()


        self._generate_analysis_report()

        print(f"Observation recording finished!")
        print(f"   Total steps: {len(self.step_data)}")
        print(f"   CSV file: {self.csv_path}")
        print(f"   JSON file: {self.json_path}")
        print(f"   Analysis report: {self.analysis_path}")

    def _generate_analysis_report(self):
        try:
            df = pd.DataFrame(self.step_data)

            with open(self.analysis_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("MetaDrive ego-vehicle observation analysis report\n")
                f.write("="*80 + "\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session name: {self.session_name}\n")
                f.write(f"Total steps: {len(self.step_data)}\n\n")


                f.write("Basic statistics\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total simulation time: {df['simulation_time'].max():.2f} s\n")
                f.write(f"Average speed: {df['speed'].mean():.2f} m/s\n")
                f.write(f"Maximum speed: {df['speed'].max():.2f} m/s\n")
                f.write(f"Minimum speed: {df['speed'].min():.2f} m/s\n")
                f.write(f"Speed standard deviation: {df['speed'].std():.2f} m/s\n\n")


                f.write("Position trajectory summary\n")
                f.write("-" * 40 + "\n")
                f.write(f"Start position: ({df['pos_x'].iloc[0]:.1f}, {df['pos_y'].iloc[0]:.1f})\n")
                f.write(f"End position: ({df['pos_x'].iloc[-1]:.1f}, {df['pos_y'].iloc[-1]:.1f})\n")
                f.write(f"X-axis displacement: {df['pos_x'].iloc[-1] - df['pos_x'].iloc[0]:.1f} m\n")
                f.write(f"Y-axis displacement: {df['pos_y'].iloc[-1] - df['pos_y'].iloc[0]:.1f} m\n")
                f.write(f"Total displacement: {np.sqrt((df['pos_x'].iloc[-1] - df['pos_x'].iloc[0])**2 + (df['pos_y'].iloc[-1] - df['pos_y'].iloc[0])**2):.1f} m\n\n")


                f.write("Stop-behavior analysis\n")
                f.write("-" * 40 + "\n")
                low_speed_steps = df[df['speed'] < 0.5]
                stopped_steps = df[df['speed'] < 0.1]
                f.write(f"Low-speed steps (<0.5 m/s): {len(low_speed_steps)} ({len(low_speed_steps)/len(df)*100:.1f}%)\n")
                f.write(f"Stopped steps (<0.1 m/s): {len(stopped_steps)} ({len(stopped_steps)/len(df)*100:.1f}%)\n")

                if len(stopped_steps) > 0:
                    f.write(f"First stop position: ({stopped_steps['pos_x'].iloc[0]:.1f}, {stopped_steps['pos_y'].iloc[0]:.1f})\n")
                    f.write(f"Last stop position: ({stopped_steps['pos_x'].iloc[-1]:.1f}, {stopped_steps['pos_y'].iloc[-1]:.1f})\n")


                f.write("\nNavigation-state analysis\n")
                f.write("-" * 40 + "\n")
                nav_completion = df['nav_route_completion'].dropna()
                if len(nav_completion) > 0:
                    f.write(f"Route-completion range: {nav_completion.min():.3f} - {nav_completion.max():.3f}\n")
                    f.write(f"Route-completion change: {nav_completion.max() - nav_completion.min():.3f}\n")


                    completion_stuck = nav_completion.std() < 0.001
                    f.write(f"Route completion stuck: {'yes' if completion_stuck else 'no'}\n")
                else:
                    f.write("No navigation data\n")


                f.write("\nAction analysis\n")
                f.write("-" * 40 + "\n")
                f.write(f"Steering-action range: {df['action_steering'].min():.3f} - {df['action_steering'].max():.3f}\n")
                f.write(f"Throttle-action range: {df['action_throttle'].min():.3f} - {df['action_throttle'].max():.3f}\n")
                f.write(f"Average steering: {df['action_steering'].mean():.3f}\n")
                f.write(f"Average throttle: {df['action_throttle'].mean():.3f}\n")


                negative_throttle = df[df['action_throttle'] < 0]
                f.write(f"Negative-throttle steps (braking): {len(negative_throttle)} ({len(negative_throttle)/len(df)*100:.1f}%)\n")
                if len(negative_throttle) > 0:
                    f.write(f"Average braking intensity: {negative_throttle['action_throttle'].mean():.3f}\n")


                action_sources = df['action_source'].value_counts()
                f.write(f"\nAction source counts:\n")
                for source, count in action_sources.items():
                    f.write(f"  {source}: {count} ({count/len(df)*100:.1f}%)\n")


                f.write("\nReward analysis\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total reward: {df['reward'].sum():.2f}\n")
                f.write(f"Average reward: {df['reward'].mean():.4f}\n")
                f.write(f"Reward standard deviation: {df['reward'].std():.4f}\n")
                f.write(f"Maximum reward: {df['reward'].max():.4f}\n")
                f.write(f"Minimum reward: {df['reward'].min():.4f}\n")


                f.write("\n Issue detection\n")
                f.write("-" * 40 + "\n")

                issues = []


                if len(stopped_steps) > len(df) * 0.3:
                    issues.append("Vehicle remains stopped for too long (>30%)")


                if len(nav_completion) > 0 and nav_completion.std() < 0.001:
                    issues.append("Navigation route completion is stuck")


                if len(negative_throttle) > len(df) * 0.5:
                    issues.append("Too much braking behavior (>50%)")


                total_displacement = np.sqrt((df['pos_x'].iloc[-1] - df['pos_x'].iloc[0])**2 +
                                           (df['pos_y'].iloc[-1] - df['pos_y'].iloc[0])**2)
                if total_displacement < 50:
                    issues.append(
                        "Total displacement is too small; forward progress may be broken"
                    )

                if issues:
                    for i, issue in enumerate(issues, 1):
                        f.write(f"{i}. {issue}\n")
                else:
                    f.write("No obvious issues detected\n")


                f.write("\nKey-moment analysis\n")
                f.write("-" * 40 + "\n")


                speed_drops = []
                for i in range(1, len(df)):
                    speed_change = df['speed'].iloc[i] - df['speed'].iloc[i-1]
                    if speed_change < -5.0:
                        speed_drops.append((i, speed_change, df['pos_x'].iloc[i], df['pos_y'].iloc[i]))

                if speed_drops:
                    f.write(f"Detected {len(speed_drops)} significant deceleration events:\n")
                    for step, change, x, y in speed_drops[:5]:
                        f.write(f"  Step {step}: speed drop {abs(change):.1f} m/s, position ({x:.1f}, {y:.1f})\n")
                else:
                    f.write("No significant deceleration events were detected\n")

                f.write("\n" + "="*80 + "\n")
                f.write("End of analysis report\n")
                f.write("="*80 + "\n")

        except Exception as e:
            print(f"Error while generating the analysis report: {e}")

    def get_current_stats(self):
        if not self.step_data:
            return {}

        df = pd.DataFrame(self.step_data)
        return {
            'total_steps': len(self.step_data),
            'current_position': (df['pos_x'].iloc[-1], df['pos_y'].iloc[-1]),
            'current_speed': df['speed'].iloc[-1],
            'average_speed': df['speed'].mean(),
            'stopped_percentage': len(df[df['speed'] < 0.1]) / len(df) * 100,
        }
