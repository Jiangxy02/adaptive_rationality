"""
Trajectory data loading and processing.

Capabilities:
- Read and parse CSV trajectory data.
- Filter by time range, such as the first 100 seconds.
- Apply coordinate transforms such as translation or normalization.
- Format and validate trajectory records.
- Inspect and visualize initial vehicle positions.

Supported transforms:
1. translate_to_origin: translate the ego vehicle (-1) to the road start.
2. use_original_position: keep original coordinates unchanged.
3. normalize_position: normalization transform kept for legacy use.

Example:
```python
from trajectory_loader import TrajectoryLoader

loader = TrajectoryLoader()
traj_data = loader.load_trajectory(
    csv_path="path/to/file.csv",
    max_duration=100,
    translate_to_origin=True
)
```
"""


import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple


TRAJECTORY_CSV_COLUMNS = (
    'timestamp',
    'vehicle_id',
    'speed_x',
    'speed_y',
    'position_x',
    'position_y',
)


class TrajectoryLoader:
    """
    Trajectory data loader.

    Loads vehicle trajectories from CSV and provides multiple coordinate
    transform options.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the trajectory loader.

        Args:
            verbose: Whether to print detailed processing information.
        """
        self.verbose = verbose

    def read_raw_csv(self, data_path):
        """Read numeric CSV trajectories that match the public example exactly."""
        path = Path(data_path)
        if path.suffix.lower() != '.csv':
            raise ValueError(f"Trajectory data only supports CSV format: {data_path}")

        df = pd.read_csv(path)
        actual_columns = list(df.columns)
        expected_columns = list(TRAJECTORY_CSV_COLUMNS)
        if actual_columns != expected_columns:
            raise ValueError(
                "CSV columns must exactly match the public example: "
                f"expected {expected_columns}, got {actual_columns}"
            )
        if df.empty:
            raise ValueError("CSV trajectory data must not be empty")

        for column in TRAJECTORY_CSV_COLUMNS:
            numeric_values = pd.to_numeric(df[column], errors='coerce')
            if numeric_values.isna().any() or not np.isfinite(numeric_values.to_numpy(dtype=float)).all():
                raise ValueError(f"CSV column '{column}' must contain only finite numeric values")
            df[column] = numeric_values

        vehicle_ids = df['vehicle_id'].to_numpy(dtype=float)
        if not np.equal(vehicle_ids, np.floor(vehicle_ids)).all():
            raise ValueError("CSV column 'vehicle_id' must contain integer values")
        df['vehicle_id'] = df['vehicle_id'].astype(np.int64)
        if not (df['vehicle_id'] == -1).any():
            raise ValueError("CSV trajectory data must include the ego vehicle_id=-1")

        return df.sort_values(['vehicle_id', 'timestamp'], kind='mergesort')

    def apply_translation(self, df):
        """Public thin wrapper around the existing translation transform."""
        return self._apply_translation_transform(df)

    def to_trajectory_dict_original(self, df):
        """Public thin wrapper that converts data using original timestamps."""
        return self._convert_to_trajectory_dict_original_timestamps(df)

    def to_trajectory_dict_resampled(self, df, target_fps=20.0):
        """Public thin wrapper that converts data at the target frame rate."""
        return self._convert_to_trajectory_dict(df, target_fps)

    def get_initial_timestamp(self, csv_path) -> float:
        """
        Read the initial timestamp from the CSV file.

        This uses the first timestamp of the ego vehicle (vehicle_id=-1).
        """
        df = self.read_raw_csv(csv_path)
        initial_timestamp = df.loc[df['vehicle_id'] == -1, 'timestamp'].iloc[0]
        print(f"Read initial timestamp from CSV: {initial_timestamp}")
        return float(initial_timestamp)

    def validate_scene_csv(self, csv_file, min_duration=3.0, min_points=30) -> bool:
        """
        Validate that the file is sufficient for parameter identification.
        """
        try:
            df = self.read_raw_csv(csv_file)

            # In the public format the ego vehicle is always vehicle_id=-1.
            main_vehicle_data = df[df['vehicle_id'] == -1]

            if len(main_vehicle_data) < min_points:
                return False

            # Check the available time range.
            timestamps = main_vehicle_data['timestamp'].values
            time_duration = timestamps.max() - timestamps.min()

            if time_duration < min_duration:
                return False

            return True

        except Exception as e:
            print(f"Error while validating file {csv_file}: {e}")
            return False

    def load_trajectory(self,
                       csv_path: str,
                       normalize_position: bool = False,
                       max_duration: Optional[float] = 100,
                       use_original_position: bool = False,
                       translate_to_origin: bool = True,
                       target_fps: float = 20.0,
                       use_original_timestamps: bool = False) -> Dict[int, List[Dict]]:
        """
        Load CSV trajectory data and optionally crop time and translate positions.
        """
        # Step 1: read and sort the CSV data.
        df = self._load_csv_data(csv_path)

        # Step 2: preprocess by removing abnormal position jumps.
        df = self._preprocess_trajectory_data(df)

        # Step 3: apply duration filtering.
        if max_duration is not None:
            df = self._filter_by_duration(df, max_duration)

        # Step 4: apply the requested position transform.
        if translate_to_origin:
            df = self._apply_translation_transform(df)
        elif use_original_position:
            self._display_original_positions(df)
        elif normalize_position:
            df = self._apply_normalization_transform(df)

        # Step 5: convert to the trajectory-dictionary format.
        if use_original_timestamps:
            trajectory_dict = self._convert_to_trajectory_dict_original_timestamps(df)
        else:
            trajectory_dict = self._convert_to_trajectory_dict(df, target_fps)

        # Step 6: print summary statistics.
        self._print_trajectory_summary(trajectory_dict)

        # Step 7: inspect sampling quality.
        self._analyze_sampling_quality(trajectory_dict)

        return trajectory_dict

    def _load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """Read and sort CSV data."""
        if self.verbose:
            print(f"Loading trajectory data from: {csv_path}")

        df = self.read_raw_csv(csv_path)

        if self.verbose:
            print(f"Loaded {len(df)} data points for {df['vehicle_id'].nunique()} vehicles")

        return df

    def _preprocess_trajectory_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess trajectory data by removing abnormal samples.
        """
        if self.verbose:
            print(f"\n=== Data Preprocessing ===")
            print(f"Original data points: {len(df)}")

        df_cleaned = df.copy()
        original_count = len(df_cleaned)

        # 1. Remove (0, 0) points, which are obvious outliers.
        zero_mask = (df_cleaned['position_x'] == 0.0) & (df_cleaned['position_y'] == 0.0)
        zero_count = zero_mask.sum()
        df_cleaned = df_cleaned[~zero_mask].copy()

        if self.verbose and zero_count > 0:
            print(f"  Removed (0,0) outliers: {zero_count} rows")

        # 2. Detect and remove position-jump outliers.
        position_outliers_removed = 0
        for vid in df_cleaned['vehicle_id'].unique():
            vehicle_mask = df_cleaned['vehicle_id'] == vid
            vehicle_data = df_cleaned[vehicle_mask].sort_values('timestamp').copy()

            if len(vehicle_data) < 2:
                continue

            # Compute position deltas between adjacent points.
            vehicle_data['pos_diff'] = np.sqrt(
                (vehicle_data['position_x'].diff())**2 +
                (vehicle_data['position_y'].diff())**2
            )

            # Treat gaps above 200 m between adjacent points as outliers.
            distance_threshold = 200.0
            outlier_mask = vehicle_data['pos_diff'] > distance_threshold
            outlier_indices = vehicle_data[outlier_mask].index

            if len(outlier_indices) > 0:
                position_outliers_removed += len(outlier_indices)
                df_cleaned = df_cleaned.drop(outlier_indices)

        if self.verbose and position_outliers_removed > 0:
            print(f"  Removed position-jump outliers: {position_outliers_removed} rows")

        # 3. Remove speed outliers.
        # Compute speed magnitude.
        df_cleaned['speed_magnitude'] = np.sqrt(df_cleaned['speed_x']**2 + df_cleaned['speed_y']**2)

        # Treat speeds above 150 km/h (about 42 m/s) as outliers.
        speed_threshold = 150.0 / 3.6  # Convert to m/s.
        speed_outlier_mask = df_cleaned['speed_magnitude'] > speed_threshold
        speed_outliers_removed = speed_outlier_mask.sum()
        df_cleaned = df_cleaned[~speed_outlier_mask].copy()

        if self.verbose and speed_outliers_removed > 0:
            print(f"  Removed speed outliers: {speed_outliers_removed} rows")

        # 4. Remove abnormal time-gap samples.
        time_outliers_removed = 0
        for vid in df_cleaned['vehicle_id'].unique():
            vehicle_mask = df_cleaned['vehicle_id'] == vid
            vehicle_data = df_cleaned[vehicle_mask].sort_values('timestamp').copy()

            if len(vehicle_data) < 2:
                continue

            # Compute time deltas.
            vehicle_data['time_diff'] = vehicle_data['timestamp'].diff()

            # Treat gaps above 1 second as data interruptions.
            time_threshold = 1.0
            time_outlier_mask = vehicle_data['time_diff'] > time_threshold
            time_outlier_indices = vehicle_data[time_outlier_mask].index

            if len(time_outlier_indices) > 0:
                time_outliers_removed += len(time_outlier_indices)
                df_cleaned = df_cleaned.drop(time_outlier_indices)

        if self.verbose and time_outliers_removed > 0:
            print(f"  Removed time-gap outliers: {time_outliers_removed} rows")

        # Clean up temporary columns.
        df_cleaned = df_cleaned.drop(columns=['speed_magnitude'], errors='ignore')

        if self.verbose:
            total_removed = original_count - len(df_cleaned)
            print(f"  Total outliers removed: {total_removed} rows ({total_removed/original_count*100:.1f}%)")
            print(f"  Data points after preprocessing: {len(df_cleaned)}")
            print(f"  Vehicles after preprocessing: {df_cleaned['vehicle_id'].nunique()}")

        return df_cleaned

    def _filter_by_duration(self, df: pd.DataFrame, max_duration: float) -> pd.DataFrame:
        """Filter data by duration."""
        min_timestamp = df['timestamp'].min()
        max_timestamp = min_timestamp + max_duration
        filtered_df = df[df['timestamp'] <= max_timestamp]

        if self.verbose:
            print(f"Filtering data to first {max_duration} seconds")
            print(f"  Timestamp range: [{min_timestamp:.1f}, {filtered_df['timestamp'].max():.1f}]")
            print(f"  Total frames: {len(filtered_df)}")

        return filtered_df

    def _apply_translation_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the translation transform that moves vehicle -1 to the road start."""
        # Get the translation reference point and offsets.
        translate_x, translate_y, ref_info = self._calculate_translation_offset(df)

        if self.verbose:
            print(f"\nUsing vehicle -1 initial position as reference: {ref_info}")
            print(f"Translation offset: ({translate_x:.1f}, {translate_y:.1f})")

        # Show the original position range.
        self._display_position_range(df, "Original position range")

        print(translate_x, translate_y)

        # Apply the translation transform.
        df_translated = df.copy()
        df_translated['position_x'] = df['position_x'] + translate_x
        df_translated['position_y'] = df['position_y'] + translate_y
        # Speed does not change because it is relative.

        # Show the translated position range.
        self._display_position_range(df_translated, "Translated position range (vehicle -1 at x=200)")

        # Show initial positions and relative relationships.
        self._display_vehicle_positions(df_translated)

        return df_translated

    def _calculate_translation_offset(self, df: pd.DataFrame) -> Tuple[float, float, str]:
        """Compute translation offsets."""
        # Use vehicle -1's initial position as the reference point.
        vehicle_minus1 = df[df['vehicle_id'] == -1]
        if not vehicle_minus1.empty:
            # Use vehicle -1's initial position.
            ref_x = vehicle_minus1.iloc[0]['position_x']
            ref_y = vehicle_minus1.iloc[0]['position_y']
            ref_info = f"({ref_x:.1f}, {ref_y:.1f})"

            # Shift vehicle -1 to x=200 and the road centerline at y=7.0.
            translate_x = 200.0 - ref_x
            translate_y = 7.0 - ref_y  # Road centerline in the default MetaDrive setup.
        else:
            # If vehicle -1 is absent, fall back to the minimum position.
            min_x = df['position_x'].min()
            min_y = df['position_y'].min()
            ref_info = f"minimum position ({min_x:.1f}, {min_y:.1f})"
            translate_x = 200.0 - min_x
            translate_y = 7.0 - min_y

        return translate_x, translate_y, ref_info

    def _display_original_positions(self, df: pd.DataFrame):
        """Display original positions when no transform is applied."""
        if not self.verbose:
            return

        print(f"\nUsing original positions without transformation")
        self._display_position_range(df, "Position range")

        # Show each vehicle's initial position.
        initial_positions = df.groupby('vehicle_id').first()[['position_x', 'position_y']]
        print(f"\nInitial vehicle positions:")
        for vid in initial_positions.index:
            x, y = initial_positions.loc[vid]
            print(f"  Vehicle {vid}: ({x:.1f}, {y:.1f})")

    def _apply_normalization_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the legacy normalization transform."""
        # Use vehicle -1's initial position as the reference point.
        vehicle_minus1 = df[df['vehicle_id'] == -1]
        if not vehicle_minus1.empty:
            # Use vehicle -1's initial position as the scene start.
            ref_x = vehicle_minus1.iloc[0]['position_x']
            ref_y = vehicle_minus1.iloc[0]['position_y']
            if self.verbose:
                print(f"Using vehicle -1 initial position as reference: ({ref_x:.1f}, {ref_y:.1f})")
        else:
            # If vehicle -1 is absent, use the minimum position.
            ref_x = df['position_x'].min()
            ref_y = df['position_y'].min()
            if self.verbose:
                print(f"Vehicle -1 not found, using minimum position as reference")

        # Translate all positions relative to the reference point.
        df_normalized = df.copy()
        df_normalized['position_x'] = df['position_x'] - ref_x + 5.0
        df_normalized['position_y'] = df['position_y'] - ref_y + 10.0

        if self.verbose:
            self._display_position_range(df_normalized, "Normalized position range")

        return df_normalized

    def _display_position_range(self, df: pd.DataFrame, title: str):
        """Display position-range information."""
        if not self.verbose:
            return

        print(f"\n{title}:")
        print(f"  X: [{df['position_x'].min():.1f}, {df['position_x'].max():.1f}], diff: {df['position_x'].max() - df['position_x'].min():.1f}")
        print(f"  Y: [{df['position_y'].min():.1f}, {df['position_y'].max():.1f}], diff: {df['position_y'].max() - df['position_y'].min():.1f}")

    def _display_vehicle_positions(self, df: pd.DataFrame):
        """Display initial vehicle positions and relative offsets."""
        if not self.verbose:
            return

        initial_positions = df.groupby('vehicle_id').first()[['position_x', 'position_y']]
        print(f"\nInitial vehicle positions after translation:")

        # Get vehicle -1's translated position.
        if -1 in initial_positions.index:
            v1_x, v1_y = initial_positions.loc[-1]
            print(f"  Vehicle -1 (main): ({v1_x:.1f}, {v1_y:.1f}) [Reference]")

            # Show other vehicles relative to vehicle -1.
            for vid in sorted(initial_positions.index):
                if vid != -1:
                    x, y = initial_positions.loc[vid]
                    rel_x = x - v1_x
                    rel_y = y - v1_y
                    distance = np.sqrt(rel_x**2 + rel_y**2)
                    print(f"  Vehicle {vid}: ({x:.1f}, {y:.1f}) [Relative: x={rel_x:+.1f}, y={rel_y:+.1f}, dist={distance:.1f}m]")
        else:
            for vid in initial_positions.index:
                x, y = initial_positions.loc[vid]
                print(f"  Vehicle {vid}: ({x:.1f}, {y:.1f})")

    def _convert_to_trajectory_dict(self, df: pd.DataFrame, target_fps: float) -> Dict[int, List[Dict]]:
        """Convert a DataFrame to trajectory dictionaries using time sampling."""
        grouped = df.groupby("vehicle_id")
        trajectory_dict = {}

        # Analyze timestamp spacing.
        timestamps = df['timestamp'].unique()
        timestamps = np.sort(timestamps)

        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            min_interval = np.min(intervals)
            max_interval = np.max(intervals)

            if self.verbose:
                print(f"\nCSV Timestamp Analysis:")
                print(f"  Mean interval: {avg_interval:.6f} s")
                print(f"  Min interval: {min_interval:.6f} s")
                print(f"  Max interval: {max_interval:.6f} s")
                print(f"  Total duration: {timestamps[-1] - timestamps[0]:.3f} s")
                print(f"  Total timestamp samples: {len(timestamps)}")

        # Use time sampling rather than interpolation.
        target_dt = 1.0 / target_fps  # Target time step.

        if self.verbose:
            print(f"\nTime Sampling Configuration:")
            print(f"  Target step size: {target_dt:.6f} s ({target_fps:.1f} Hz)")
            print(f"  Sampling strategy: choose the closest real point for each vehicle")

        for vid, group in grouped:
            group = group.reset_index(drop=True)

            # Generate sample times separately for each vehicle.
            vehicle_start_time = group['timestamp'].iloc[0]
            vehicle_end_time = group['timestamp'].iloc[-1]
            vehicle_sample_times = np.arange(vehicle_start_time, vehicle_end_time + target_dt, target_dt)

            # Sample each vehicle trajectory in time.
            sampled_traj = self._sample_vehicle_trajectory(group, vehicle_sample_times)
            trajectory_dict[int(vid)] = sampled_traj

            if self.verbose:
                print(f"  Vehicle {vid}: {len(sampled_traj)} points, time range: {vehicle_start_time:.3f} - {vehicle_end_time:.3f}s")

        return trajectory_dict

    def _convert_to_trajectory_dict_original_timestamps(self, df: pd.DataFrame) -> Dict[int, List[Dict]]:
        """
        Convert trajectory data using original CSV timestamps without resampling.
        """
        trajectory_dict = {}
        grouped = df.groupby("vehicle_id")

        if self.verbose:
            print(f"\nTime Sampling Configuration:")
            print(f"  Use original CSV timestamps (no resampling)")
            print(f"  Keep all original data points")

        for vid, group in grouped:
            group = group.reset_index(drop=True)

            # Use original data directly with no time resampling.
            original_traj = []
            for _, row in group.iterrows():
                speed = np.sqrt(row["speed_x"]**2 + row["speed_y"]**2)

                original_traj.append({
                    "x": row["position_x"],
                    "y": row["position_y"],
                    "speed": speed,
                    "heading": 0.0,  # Temporary placeholder; computed later.
                    "timestamp": row["timestamp"],  # Use the original timestamp.
                    "original_timestamp": row["timestamp"],  # Preserve the raw timestamp.
                    "time_error": 0.0,  # No sampling error.
                    # Preserve raw speed components for dynamics mode.
                    "speed_x": row["speed_x"],
                    "speed_y": row["speed_y"]
                })

            # Compute stable headings from the original data.
            self._calculate_stable_headings(original_traj)

            trajectory_dict[int(vid)] = original_traj

            if self.verbose:
                print(f"  Vehicle {vid}: {len(original_traj)} original points, time range: {group['timestamp'].iloc[0]:.3f} - {group['timestamp'].iloc[-1]:.3f}s")

        return trajectory_dict

    def _sample_vehicle_trajectory(self, vehicle_df: pd.DataFrame, sample_times: np.ndarray) -> List[Dict]:
        """
        Time-sample one vehicle trajectory using the nearest real data point.
        """
        original_timestamps = vehicle_df['timestamp'].values

        sampled_traj = []
        for i, target_time in enumerate(sample_times):
            # Find the index with the closest timestamp.
            closest_idx = np.argmin(np.abs(original_timestamps - target_time))
            row = vehicle_df.iloc[closest_idx]

            speed = np.sqrt(row["speed_x"]**2 + row["speed_y"]**2)

            # Compute the true time error.
            time_error = abs(row["timestamp"] - target_time)

            sampled_traj.append({
                "x": row["position_x"],
                "y": row["position_y"],
                "speed": speed,
                "heading": 0.0,  # Temporary placeholder; computed later.
                "timestamp": target_time,  # Use the target time for synchronization.
                "original_timestamp": row["timestamp"],  # Preserve the raw timestamp.
                "time_error": time_error,  # Record the time error.
                # Preserve raw speed components for dynamics mode.
                "speed_x": row["speed_x"],
                "speed_y": row["speed_y"]
            })

        # Compute stable headings from the sampled data.
        self._calculate_stable_headings(sampled_traj)

        return sampled_traj

    def _calculate_stable_headings(self, trajectory: List[Dict]):
        """
        Compute stable headings from neighbor-point position differences.
        """
        if len(trajectory) < 2:
            # Keep the default heading when fewer than two points are available.
            for point in trajectory:
                point["heading"] = 0.0
            return

        # Compute one heading per point.
        for i in range(len(trajectory)):
            if i == 0:
                # First point: use the direction to the next point.
                dx = trajectory[i + 1]["x"] - trajectory[i]["x"]
                dy = trajectory[i + 1]["y"] - trajectory[i]["y"]
            elif i == len(trajectory) - 1:
                # Last point: use the direction from the previous point.
                dx = trajectory[i]["x"] - trajectory[i - 1]["x"]
                dy = trajectory[i]["y"] - trajectory[i - 1]["y"]
            else:
                # Middle points: use the average of forward and backward directions.
                dx1 = trajectory[i]["x"] - trajectory[i - 1]["x"]
                dy1 = trajectory[i]["y"] - trajectory[i - 1]["y"]
                dx2 = trajectory[i + 1]["x"] - trajectory[i]["x"]
                dy2 = trajectory[i + 1]["y"] - trajectory[i]["y"]
                dx = (dx1 + dx2) / 2.0
                dy = (dy1 + dy2) / 2.0

            # Compute heading angle.
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # Avoid numerical issues while stopped.
                heading = np.arctan2(dy, dx)
            else:
                heading = 0.0  # Default to the positive x direction while stopped.

            trajectory[i]["heading"] = heading

        if self.verbose and len(trajectory) > 0:
            # Print heading statistics for diagnostics.
            headings = [point["heading"] for point in trajectory]
            heading_changes = [abs(headings[i] - headings[i-1]) for i in range(1, len(headings))]
            if heading_changes:
                max_change = max(heading_changes) * 180 / np.pi  # Convert to degrees.
                avg_change = np.mean(heading_changes) * 180 / np.pi
                print(f"    Heading stability: max_change={max_change:.1f}°, avg_change={avg_change:.1f}°")

    def _print_trajectory_summary(self, trajectory_dict: Dict[int, List[Dict]]):
        """Print a trajectory summary."""
        if not self.verbose:
            return

        print(f"\nLoaded trajectories for {len(trajectory_dict)} vehicles")
        print(f"Trajectory lengths: {[len(traj) for traj in trajectory_dict.values()]}")

    def _analyze_sampling_quality(self, trajectory_dict: Dict[int, List[Dict]]):
        """Analyze sampling quality and display time-error statistics."""
        if not self.verbose:
            return

        all_time_errors = []
        for vid, traj in trajectory_dict.items():
            for point in traj:
                all_time_errors.append(point["time_error"])

        if all_time_errors:
            avg_time_error = np.mean(all_time_errors)
            max_time_error = np.max(all_time_errors)
            min_time_error = np.min(all_time_errors)

            print(f"\nSampling Quality Analysis:")
            print(f"  Mean time error: {avg_time_error:.6f} s")
            print(f"  Max time error: {max_time_error:.6f} s")
            print(f"  Min time error: {min_time_error:.6f} s")
            print(f"  Total sampled points: {sum(len(traj) for traj in trajectory_dict.values())}")
            print(f"  Total time-error samples: {len(all_time_errors)}")
        else:
            print(f"\nSampling Quality Analysis: no time-error data")

    def get_vehicle_info(self, trajectory_dict: Dict[int, List[Dict]]) -> Dict:
        """
        Get summary statistics for trajectory data.
        """
        if not trajectory_dict:
            return {"vehicle_count": 0, "trajectory_lengths": []}

        return {
            "vehicle_count": len(trajectory_dict),
            "vehicle_ids": list(trajectory_dict.keys()),
            "trajectory_lengths": [len(traj) for traj in trajectory_dict.values()],
            "max_trajectory_length": max(len(traj) for traj in trajectory_dict.values()),
            "min_trajectory_length": min(len(traj) for traj in trajectory_dict.values()),
            "avg_trajectory_length": np.mean([len(traj) for traj in trajectory_dict.values()]),
            "has_main_vehicle": -1 in trajectory_dict.keys()
        }

    def validate_trajectory_data(self, trajectory_dict: Dict[int, List[Dict]]) -> Tuple[bool, List[str]]:
        """
        Validate trajectory-data completeness and integrity.
        """
        errors = []

        # Check for empty input.
        if not trajectory_dict:
            errors.append("Trajectory data is empty")
            return False, errors

        # Check for the ego vehicle (vehicle -1).
        if -1 not in trajectory_dict:
            errors.append("Missing ego-vehicle data (vehicle -1)")

        # Check each trajectory for required fields.
        for vid, traj in trajectory_dict.items():
            if not traj:
                errors.append(f"Vehicle {vid} has an empty trajectory")
                continue

            # Check required fields on each trajectory point.
            required_fields = ["x", "y", "speed", "heading"]
            for i, point in enumerate(traj):
                for field in required_fields:
                    if field not in point:
                        errors.append(f"Vehicle {vid} trajectory point {i} is missing field '{field}'")
                    elif not isinstance(point[field], (int, float)):
                        errors.append(f"Vehicle {vid} trajectory point {i} has invalid type for field '{field}'")

        return len(errors) == 0, errors


def load_trajectory(csv_path: str,
                   normalize_position: bool = False,
                   max_duration: float = None,
                   use_original_position: bool = False,
                   translate_to_origin: bool = True,
                   target_fps: float = 50.0,
                   use_original_timestamps: bool = False) -> Dict[int, List[Dict]]:
    """
    Convenience wrapper for loading trajectory data.
    """
    loader = TrajectoryLoader(verbose=True)
    return loader.load_trajectory(
        csv_path=csv_path,
        normalize_position=normalize_position,
        max_duration=max_duration,
        use_original_position=use_original_position,
        translate_to_origin=translate_to_origin,
        target_fps=target_fps,
        use_original_timestamps=use_original_timestamps
    )
