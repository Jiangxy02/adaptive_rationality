
import numpy as np
from typing import Dict, List, Optional, Any
import json
import os
import logging
from datetime import datetime
import matplotlib.pyplot as plt

logging.getLogger('cognitive_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_bias_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_perception_module').setLevel(logging.WARNING)
logging.getLogger('cognitive_module.cognitive_delay_module').setLevel(logging.WARNING)


class SaveIntermediateResults:
    def __init__(self, prediction_history, output_dir: str, pred_win):
        self.prediction_history = prediction_history
        self.output_dir = output_dir
        self.pred_win = pred_win


    def _save_collision_records(self):
        """Save collision records to disk."""
        collision_records = []

        for prediction_data in self.prediction_history:
            collision_info = prediction_data.get('collision_info')
            if not collision_info:
                continue
            if not (
                collision_info['collision_detected']
                or collision_info['out_of_road_detected']
            ):
                continue

            record = {
                'window_idx': prediction_data.get('window_idx'),
                'prediction_start_time': prediction_data.get('prediction_start_time'),
                'prediction_start_state': prediction_data.get('prediction_start_state'),
                'best_theta': prediction_data.get('best_theta'),
                'has_collision': collision_info['collision_detected'],
                'has_out_of_road': collision_info['out_of_road_detected'],
                'termination_reason': collision_info['termination_reason']
            }

            if collision_info['collision_detected']:
                record.update({
                    'collision_step': collision_info['collision_step'],
                    'collision_time': collision_info['collision_time'],
                    'collision_type': collision_info['collision_type'],
                    'time_to_collision': collision_info['collision_time'] - prediction_data.get('prediction_start_time', 0)
                })

            if collision_info['out_of_road_detected']:
                record.update({
                    'out_of_road_step': collision_info['out_of_road_step'],
                    'out_of_road_time': collision_info['out_of_road_time'],
                    'time_to_out_of_road': collision_info['out_of_road_time'] - prediction_data.get('prediction_start_time', 0)
                })

            collision_records.append(record)

        collision_json_file = os.path.join(self.output_dir, 'collision_records_detailed.json')
        with open(collision_json_file, 'w', encoding='utf-8') as f:
            json.dump(collision_records, f, indent=2, default=str, ensure_ascii=False)

        collision_csv_file = os.path.join(self.output_dir, 'collision_records_summary.csv')
        with open(collision_csv_file, 'w', encoding='utf-8', newline='') as f:
            if collision_records:
                import csv
                all_possible_fields = set()
                for record in collision_records:
                    all_possible_fields.update(record.keys())

                fieldnames = sorted(list(all_possible_fields))
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(collision_records)

        self._create_collision_statistics_report(collision_records)

        print("Collision records saved:")
        print(f"   Detailed records: {collision_json_file}")
        print(f"   Summary records: {collision_csv_file}")

        return collision_records

    def _create_collision_statistics_report(self, collision_records):
        """Create a collision statistics report."""
        report_file = os.path.join(self.output_dir, 'collision_statistics_report.txt')

        total_predictions = len(self.prediction_history)
        collision_predictions = len([r for r in collision_records if r['has_collision']])
        out_of_road_predictions = len([r for r in collision_records if r['has_out_of_road']])
        safe_predictions = total_predictions - len(collision_records)

        collision_types = {}
        collision_times = []
        out_of_road_times = []

        for record in collision_records:
            if record['has_collision']:
                collision_type = record.get('collision_type', 'unknown')
                collision_types[collision_type] = collision_types.get(collision_type, 0) + 1
                if 'time_to_collision' in record:
                    collision_times.append(record['time_to_collision'])

            if record['has_out_of_road']:
                if 'time_to_out_of_road' in record:
                    out_of_road_times.append(record['time_to_out_of_road'])

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COLLISION STATISTICS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total predictions: {total_predictions}\n")
            f.write(f"Prediction window size: {self.pred_win} steps\n\n")

            f.write("=" * 40 + "\n")
            f.write("Overall statistics\n")
            f.write("=" * 40 + "\n")
            f.write(f"Safe predictions: {safe_predictions} ({safe_predictions/total_predictions*100:.1f}%)\n")
            f.write(f"Collision predictions: {collision_predictions} ({collision_predictions/total_predictions*100:.1f}%)\n")
            f.write(f"Out-of-road predictions: {out_of_road_predictions} ({out_of_road_predictions/total_predictions*100:.1f}%)\n\n")

            if collision_types:
                f.write("=" * 40 + "\n")
                f.write("Collision type statistics\n")
                f.write("=" * 40 + "\n")
                for collision_type, count in collision_types.items():
                    f.write(f"{collision_type}: {count} ({count/collision_predictions*100:.1f}%)\n")
                f.write("\n")

            if collision_times:
                import numpy as np
                f.write("=" * 40 + "\n")
                f.write("Collision time statistics (from prediction start)\n")
                f.write("=" * 40 + "\n")
                f.write(f"Mean collision time: {np.mean(collision_times):.2f}s\n")
                f.write(f"Shortest collision time: {np.min(collision_times):.2f}s\n")
                f.write(f"Longest collision time: {np.max(collision_times):.2f}s\n")
                f.write(f"Collision time standard deviation: {np.std(collision_times):.2f}s\n\n")

            if out_of_road_times:
                import numpy as np
                f.write("=" * 40 + "\n")
                f.write("Out-of-road time statistics (from prediction start)\n")
                f.write("=" * 40 + "\n")
                f.write(f"Mean out-of-road time: {np.mean(out_of_road_times):.2f}s\n")
                f.write(f"Shortest out-of-road time: {np.min(out_of_road_times):.2f}s\n")
                f.write(f"Longest out-of-road time: {np.max(out_of_road_times):.2f}s\n")
                f.write(f"Out-of-road time standard deviation: {np.std(out_of_road_times):.2f}s\n\n")

            f.write("=" * 80 + "\n")
            f.write("Detailed collision records\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Window':<6} {'Start Time':<12} {'Event':<10} {'Event Time':<12} {'Type':<12} {'Delta':<8}\n")
            f.write("-" * 80 + "\n")

            for record in collision_records:
                window_idx = record['window_idx']
                start_time = record['prediction_start_time']

                if record['has_collision']:
                    event = "collision"
                    event_time = record.get('collision_time', 'N/A')
                    event_type = record.get('collision_type', 'unknown')
                    time_diff = record.get('time_to_collision', 'N/A')
                    f.write(f"{window_idx:<6} {start_time:<12.2f} {event:<10} {event_time:<12.2f} {event_type:<12} {time_diff:<8.2f}\n")

                if record['has_out_of_road']:
                    event = "out_of_road"
                    event_time = record.get('out_of_road_time', 'N/A')
                    event_type = "out_of_road"
                    time_diff = record.get('time_to_out_of_road', 'N/A')
                    f.write(f"{window_idx:<6} {start_time:<12.2f} {event:<10} {event_time:<12.2f} {event_type:<12} {time_diff:<8.2f}\n")

        print(f"Collision statistics report saved to: {report_file}")

    def _create_prediction_statistics_report(self):
        """Generate prediction statistics report"""
        if not self.prediction_history:
            return

        report_file = os.path.join(self.output_dir, 'prediction_statistics_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAJECTORY PREDICTION STATISTICS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Prediction Window Size: {self.pred_win} steps\n")
            f.write(f"Total Prediction Instances: {len(self.prediction_history)}\n\n")

            # Statistics for each prediction moment
            f.write("## PREDICTION INSTANCE DETAILS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Window':<8} {'Start Time':<12} {'Pred Points':<12} {'GT Points':<12} {'Best Weight':<12}\n")
            f.write("-" * 80 + "\n")

            total_predictions = 0
            total_real_segments = 0

            for pred_data in self.prediction_history:
                window_idx = pred_data['window_idx']
                start_time = pred_data['prediction_start_time']
                pred_len = len(pred_data['predicted_trajectory']) if pred_data['predicted_trajectory'] else 0
                real_len = len(pred_data['original_trajectory_segment']) if pred_data['original_trajectory_segment'] else 0
                weight = pred_data['best_particle_weight']

                f.write(f"W{window_idx+1:<7} {start_time:<12.3f} {pred_len:<12} {real_len:<12} {weight:<12.6f}\n")

                if pred_len > 0:
                    total_predictions += 1
                if real_len > 0:
                    total_real_segments += 1

            f.write("-" * 80 + "\n")
            f.write(f"\nSuccessful Predictions: {total_predictions}/{len(self.prediction_history)}\n")
            f.write(f"Predictions with Ground Truth: {total_real_segments}/{len(self.prediction_history)}\n")

            # Optimal parameter statistics
            f.write(f"\n## OPTIMAL PARAMETER STATISTICS\n")
            f.write("-" * 80 + "\n")

            # Collect all optimal parameters
            param_values = {}
            for pred_data in self.prediction_history:
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



class DataSave:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir


    def _save_intermediate_results(self, step: int, estimation_history):
        """Save intermediate results."""
        self.estimation_history = estimation_history
        history_file = os.path.join(self.output_dir, f'estimation_history_step_{step}.json')
        with open(history_file, 'w') as f:
            json.dump(self.estimation_history, f, indent=2, default=str)


    def _generate_report(self, final_stats: Dict[str, Any],
                        num_particles: Optional[int] = None,
                        window_size: Optional[int] = None,
                        step_interval: Optional[int] = None,
                        horizon: Optional[int] = None):
        """Generate the identification report.

        Args:
            final_stats: Posterior summary containing mean, standard deviation,
                MAP estimates, and credible intervals.
            num_particles: Optional particle count recorded in the report.
            window_size: Optional identification-window size.
            step_interval: Optional stride between identification windows.
            horizon: Optional look-ahead horizon.
        """
        report_file = os.path.join(self.output_dir, 'identification_report.txt')

        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("PARAMETER IDENTIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if num_particles is not None:
                f.write(f"Number of particles: {num_particles}\n")
            if window_size is not None:
                f.write(f"Window size: {window_size}\n")
            if step_interval is not None:
                f.write(f"Step interval: {step_interval}\n")
            if horizon is not None:
                f.write(f"Look-ahead horizon: {horizon}\n")
            f.write("\n")

            f.write("## Parameter estimates (posterior mean)\n")
            f.write("-" * 40 + "\n")
            for param, value in final_stats['mean'].items():
                std = final_stats['std'][param]
                f.write(f"{param:25s}: {value:8.4f} ± {std:8.4f}\n")

            f.write("\n## MAP estimate (maximum a posteriori)\n")
            f.write("-" * 40 + "\n")
            for param, value in final_stats['map'].items():
                f.write(f"{param:25s}: {value:8.4f}\n")

            f.write("\n## 95% confidence intervals\n")
            f.write("-" * 40 + "\n")
            for param, (lower, upper) in final_stats['ci_95'].items():
                f.write(f"{param:25s}: [{lower:8.4f}, {upper:8.4f}]\n")

            f.write(f"\nFinal effective sample size (ESS): {final_stats['ess']:.1f}\n")

        print(f"Report saved to: {report_file}")

    def plot_results(self, param_names: Optional[List[str]] = None):
        """Plot identification results."""
        if not hasattr(self, 'estimation_history') or not self.estimation_history:
            print("No results available for plotting")
            return

        self._create_traditional_plot(param_names)

    def _create_traditional_plot(self, param_names: Optional[List[str]] = None):
        """Create the standard parameter-estimation figure."""
        if param_names is None:
            param_names = list(self.estimation_history[0]['mean'].keys())

        fig, axes = plt.subplots(len(param_names), 2, figsize=(12, 4*len(param_names)))
        if len(param_names) == 1:
            axes = axes.reshape(1, -1)

        windows = [h['window_idx'] for h in self.estimation_history]

        for i, param in enumerate(param_names):
            means = [h['mean'][param] for h in self.estimation_history]
            stds = [h['std'][param] for h in self.estimation_history]
            maps = [h['map'][param] for h in self.estimation_history]
            ci_lowers = [h['ci_95'][param][0] for h in self.estimation_history]
            ci_uppers = [h['ci_95'][param][1] for h in self.estimation_history]

            ax1 = axes[i, 0]
            ax1.plot(windows, means, 'b-', label='Mean', linewidth=2)
            ax1.plot(windows, maps, 'r--', label='MAP', linewidth=1)
            ax1.fill_between(windows, ci_lowers, ci_uppers, alpha=0.3, color='blue')
            ax1.set_xlabel('Window Index')
            ax1.set_ylabel(param)
            ax1.set_title(f'{param} - Estimation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[i, 1]
            ax2.plot(windows, stds, 'g-', linewidth=2)
            ax2.set_xlabel('Window Index')
            ax2.set_ylabel('Std Dev')
            ax2.set_title(f'{param} - Uncertainty')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_file = os.path.join(self.output_dir, 'parameter_estimation_plot.png')
        plt.savefig(plot_file, dpi=150)
        plt.close()

        print(f"Figure saved to: {plot_file}")
