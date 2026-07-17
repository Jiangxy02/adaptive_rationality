# Trajectory Data

**English** | [中文](DATA.zh-CN.md)

## Source and purpose

The bundled example is a multi-vehicle trajectory recorded during a human-participant driving takeover experiment conducted in a CARLA scenario. It captures the ego vehicle's motion during the participant's takeover together with surrounding-vehicle motion. The released CSV contains the timestamps, vehicle identifiers, planar velocity components, and planar positions exported from the simulator.

This trajectory is input data for cognitive-parameter identification and future-trajectory prediction. It is not used to train the PPO policy, which is trained separately in MetaDrive.

The example file is:

```text
examples/trajectory_scenario/example_trajectory_scene.csv
```

Each CSV file represents one scene. Users may replace the example with their own scene files as long as they follow the contract below.

## CSV schema

The header and column order must match exactly:

```csv
timestamp,vehicle_id,speed_x,speed_y,position_x,position_y
```

| Column | Meaning | Unit |
| --- | --- | --- |
| `timestamp` | Absolute or relative sample time | seconds |
| `vehicle_id` | Integer vehicle identifier; `-1` is the ego vehicle | — |
| `speed_x` | Velocity along the x axis | m/s |
| `speed_y` | Velocity along the y axis | m/s |
| `position_x` | Position along the x axis | m |
| `position_y` | Position along the y axis | m |

All vehicles in one scene must use the same coordinate system. Absolute and relative timestamps are both accepted because the workflow uses elapsed time within the scene.

## Batch processing

For `parameter_identify/multi_main.py`, place one scene in each CSV file and store the files directly in the directory passed through `--data_dir`. The batch entrypoint scans only top-level `*.csv` files.

Each batch scene must contain at least 30 ego-vehicle points spanning at least 3 seconds. Background vehicles are not required by the CSV validator, but traffic-aware replay and collision analysis require trajectories for vehicles other than `vehicle_id=-1`.

The batch runner considers all valid top-level candidates in sorted order until the requested number of successful scenes is reached. Per-scene outputs are isolated under `scenarios/<sequence>_<csv-stem>/`; an incomplete scene is moved to the matching `.failed` directory without deleting another scene's results.

## Data availability

This repository includes only the example trajectory described above. The full research dataset is not included.
