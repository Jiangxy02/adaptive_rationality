#!/usr/bin/env python3
"""Time synchronization utilities for parameter identification.

This module aligns observed trajectories with the simulation time step when the
observation frequency does not match the rollout frequency.
"""


import numpy as np
from typing import List, Dict, Optional
from parameter_identify.utils.likelihood_calculator import TrajectoryPoint


class ParameterIdentificationTimeSynchronizer:
    """Time synchronizer specialized for parameter identification."""

    def __init__(self, simulation_time_step: float = 0.1):
        """Initialize the synchronizer.

        Args:
            simulation_time_step: Simulation time step, typically 0.1 s for a
                10 Hz decision frequency.
        """
        self.simulation_time_step = simulation_time_step
        self.trajectory_start_time = None

    def analyze_trajectory_timing(self, trajectory: List[TrajectoryPoint]) -> Dict[str, float]:
        """Analyze timing statistics for one observed trajectory."""
        if len(trajectory) < 2:
            return {"avg_interval": 0.0, "std_interval": 0.0, "frequency": 0.0}

        timestamps = [p.timestamp for p in trajectory]
        intervals = np.diff(timestamps)

        analysis = {
            "avg_interval": float(np.mean(intervals)),
            "std_interval": float(np.std(intervals)),
            "min_interval": float(np.min(intervals)),
            "max_interval": float(np.max(intervals)),
            "frequency": float(1.0 / np.mean(intervals)) if np.mean(intervals) > 0 else 0.0,
            "total_duration": float(timestamps[-1] - timestamps[0]),
            "num_points": len(trajectory)
        }

        if self.trajectory_start_time is None:
            self.trajectory_start_time = timestamps[0]

        return analysis

    def create_time_aligned_windows(self,
                                  trajectory: List[TrajectoryPoint],
                                  window_duration: float,
                                  step_duration: float = None) -> List[List[TrajectoryPoint]]:
        """Create sliding windows with aligned time spans."""
        if step_duration is None:
            step_duration = window_duration

        if not trajectory:
            return []

        timing_info = self.analyze_trajectory_timing(trajectory)
        print("Trajectory timing analysis:")
        print(f"   Mean interval: {timing_info['avg_interval']:.4f}s ({timing_info['frequency']:.1f} Hz)")
        print(f"   Interval range: {timing_info['min_interval']:.4f} - {timing_info['max_interval']:.4f}s")
        print(f"   Total duration: {timing_info['total_duration']:.2f}s")

        timestamps = [p.timestamp for p in trajectory]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_duration = end_time - start_time

        num_windows = int((total_duration - window_duration) / step_duration) + 1

        windows = []
        for i in range(num_windows):
            window_start_time = start_time + i * step_duration
            window_end_time = window_start_time + window_duration

            window_points = []
            for point in trajectory:
                if window_start_time <= point.timestamp <= window_end_time:
                    window_points.append(point)

            if len(window_points) >= 2:
                windows.append(window_points)

        print(f"Created {len(windows)} time-aligned windows")
        print(f"   Window duration: {window_duration:.2f}s")
        print(f"   Sliding step: {step_duration:.2f}s")

        return windows

    def resample_trajectory_to_fixed_frequency(self,
                                             trajectory: List[TrajectoryPoint],
                                             target_frequency: float = 10.0) -> List[TrajectoryPoint]:
        """Resample a trajectory to a fixed frequency."""
        if len(trajectory) < 2:
            return trajectory

        target_interval = 1.0 / target_frequency

        timestamps = [p.timestamp for p in trajectory]
        start_time = timestamps[0]
        end_time = timestamps[-1]
        total_duration = end_time - start_time

        num_target_points = int(total_duration / target_interval) + 1
        target_timestamps = [start_time + i * target_interval for i in range(num_target_points)]

        resampled_trajectory = []

        for target_time in target_timestamps:
            if target_time > end_time:
                break

            interpolated_point = self._interpolate_trajectory_point(trajectory, target_time)
            if interpolated_point:
                resampled_trajectory.append(interpolated_point)

        print("Trajectory resampling:")
        print(f"   Original points: {len(trajectory)}")
        print(f"   Resampled points: {len(resampled_trajectory)}")
        print(f"   Target frequency: {target_frequency} Hz")
        print(f"   Effective frequency: {len(resampled_trajectory) / total_duration:.1f} Hz")

        return resampled_trajectory

    def _interpolate_trajectory_point(self,
                                    trajectory: List[TrajectoryPoint],
                                    target_time: float) -> Optional[TrajectoryPoint]:
        """Interpolate the trajectory state at one target timestamp."""
        timestamps = [p.timestamp for p in trajectory]

        if target_time <= timestamps[0]:
            return trajectory[0]
        elif target_time >= timestamps[-1]:
            return trajectory[-1]

        for i in range(len(timestamps) - 1):
            if timestamps[i] <= target_time <= timestamps[i + 1]:
                t0, t1 = timestamps[i], timestamps[i + 1]
                p0, p1 = trajectory[i], trajectory[i + 1]

                if t1 != t0:
                    alpha = (target_time - t0) / (t1 - t0)
                else:
                    alpha = 0.0

                interpolated_point = TrajectoryPoint(
                    px=p0.px + alpha * (p1.px - p0.px),
                    py=p0.py + alpha * (p1.py - p0.py),
                    vx=p0.vx + alpha * (p1.vx - p0.vx),
                    vy=p0.vy + alpha * (p1.vy - p0.vy),
                    timestamp=target_time,
                    yaw=p0.yaw,
                )

                return interpolated_point

        closest_idx = min(range(len(timestamps)),
                         key=lambda i: abs(timestamps[i] - target_time))
        return trajectory[closest_idx]

    def create_simulation_aligned_trajectory(self,
                                           trajectory: List[TrajectoryPoint],
                                           num_steps: int) -> List[TrajectoryPoint]:
        """Create a trajectory aligned with simulation steps."""
        if not trajectory:
            return []

        start_time = trajectory[0].timestamp
        simulation_timestamps = [start_time + i * self.simulation_time_step
                               for i in range(num_steps)]

        aligned_trajectory = []
        for sim_time in simulation_timestamps:
            aligned_point = self._interpolate_trajectory_point(trajectory, sim_time)
            if aligned_point:
                aligned_trajectory.append(aligned_point)

        print("Created simulation-aligned trajectory:")
        print(f"   Simulation steps: {num_steps}")
        print(f"   Simulation time step: {self.simulation_time_step}s")
        print(f"   Aligned trajectory points: {len(aligned_trajectory)}")

        return aligned_trajectory
