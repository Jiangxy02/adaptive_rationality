from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from parameter_identify.utils.likelihood_calculator import TrajectoryPoint
from parameter_identify.utils.trajectory_loader import TrajectoryLoader


class DataLoad:
    def __init__(self):
        pass

    def load_trajectory_data(
        self,
        data_path: str,
        trajectory_duration: Optional[float] = None,
        translate_to_origin: bool = True,
    ) -> Tuple[
        List[TrajectoryPoint],
        Dict[int, List[Dict]],
        np.ndarray,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        loader = TrajectoryLoader(verbose=False)

        full_data = loader.read_raw_csv(data_path)
        processed_data = full_data.copy()

        if translate_to_origin:
            processed_data = loader.apply_translation(processed_data)

        trajectory_dict = loader.to_trajectory_dict_original(processed_data)
        main_vehicle_data = processed_data[processed_data["vehicle_id"] == -1].copy()

        if len(main_vehicle_data) == 0:
            raise ValueError("Main vehicle data (vehicle_id=-1) was not found")

        timestamps = main_vehicle_data["timestamp"].values

        if trajectory_duration is not None:
            start_timestamp = timestamps[0]
            end_timestamp = start_timestamp + trajectory_duration
            valid_indices = timestamps <= end_timestamp
            num_steps = int(valid_indices.sum())
            num_steps = min(num_steps, len(main_vehicle_data))

            if num_steps > 0:
                main_vehicle_data = main_vehicle_data.iloc[:num_steps].copy()
                timestamps = main_vehicle_data["timestamp"].values
                actual_duration = timestamps[-1] - timestamps[0]
                print(
                    f"Selected {num_steps} data points for requested duration "
                    f"{trajectory_duration}s"
                )
                print(f"   Actual duration: {actual_duration:.2f}s")
                print(f"   Time range: {timestamps[0]:.1f} - {timestamps[-1]:.1f}")
            else:
                raise ValueError(
                    "No valid data points could be selected; the requested duration "
                    f"{trajectory_duration}s exceeds the data range"
                )
        else:
            num_steps = len(main_vehicle_data)
            total_duration = timestamps[-1] - timestamps[0]
            print(f"Using the full dataset: {num_steps} data points")
            print(f"   Total duration: {total_duration:.2f}s")

        trajectory = []
        for i in range(len(main_vehicle_data)):
            row = main_vehicle_data.iloc[i]
            point = TrajectoryPoint(
                px=row["position_x"],
                py=row["position_y"],
                vx=row["speed_x"],
                vy=row["speed_y"],
                timestamp=row["timestamp"],
                yaw=None,
            )
            trajectory.append(point)

        return (
            trajectory,
            trajectory_dict,
            timestamps,
            processed_data,
            main_vehicle_data,
        )
