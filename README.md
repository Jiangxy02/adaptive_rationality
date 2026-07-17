# Adaptive Bounded-Rationality Modeling of Early-Stage Takeover in Shared-Control Driving


[![Paper](https://img.shields.io/badge/CHI%202026-Paper-blue.svg)](https://doi.org/10.1145/3772318.3790701)


This repository accompanies *Adaptive Bounded-Rationality Modeling of Early-Stage Takeover in Shared-Control Driving* and provides the current public implementation of its core modeling workflow. The paper studies the first seconds after a driving-control handover, when rapidly changing cognitive states make steering and pedal responses difficult to predict. It embeds perceptual uncertainty, looming aversion, and action delay in a bounded-rationality reinforcement-learning policy, then identifies the corresponding latent cognitive parameters online from observed actions to predict near-future control behavior and support earlier safety intervention.

The repository supports two main stages:

1. **Training:** Train a cognitive-modulated PPO driving policy in MetaDrive.
2. **Identification and prediction:** Use a trained checkpoint and an observed multi-vehicle trajectory CSV to identify four cognitive parameters and predict the ego vehicle's future trajectory.

![Research framework](fig/Fig.%201.%20Research%20framework.png)

## Release Status

- [x] Release the training, cognitive-parameter-identification, and trajectory-prediction code.
- [x] Add environment installation and verification instructions.

## Installation

Clone the repository, enter its root directory, create a Python 3.8 environment, and install the project:

```bash
git clone https://github.com/Jiangxy02/adaptive_rationality.git
cd adaptive_rationality
conda create -n adaptive_rationality python=3.8
conda activate adaptive_rationality
pip install -e .
```

### MetaDrive dependency

This project depends on a [pinned public fork of MetaDrive 0.4.3](https://github.com/Jiangxy02/metadrive/tree/fad0a7e18db3ed899a32e5e42aaa88cbe4c4df52). Relative to the upstream version, the fork changes only two runtime behaviors required by this project: IDM traffic target speeds are sampled in the 60–80 km/h range, and Panda3D `NodePath` cleanup safely handles already-invalidated objects.

The fork does not bundle MetaDrive's asset archive. When a MetaDrive environment is initialized for the first time, it downloads the [official MetaDrive 0.4.3 runtime assets](https://github.com/metadriverse/metadrive/releases/tag/MetaDrive-0.4.3) from the upstream release into the installed Python package, so the first initialization requires network access to GitHub Releases. Note that the MetaDrive asset download only retrieves MetaDrive runtime resources.

### Installation verification

Run the following command to verify the installation:

```bash
python scripts/smoke_test.py
```

The final line should be `SMOKE PASS`.

## Training

Training learns a cognitive-conditioned PPO driving policy in MetaDrive traffic scenarios. At each simulation step, the policy combines a 275-dimensional MetaDrive observation with four cognitive parameters and their four masks, then outputs continuous steering and acceleration actions. The cognitive parameters are sampled during training so that the resulting checkpoint can support the parameter-identification and future-trajectory-prediction workflow described below.

Run the following command from the repository root:

```bash
python ppo_train/scripts/train.py \
  --device auto \
  --save_dir outputs/training
```

Use `--device cpu` when CUDA is unavailable. To view and override all available options, run:

```bash
python ppo_train/scripts/train.py --help
```

### Training configuration

Training options are defined under `ppo_train/config/`, including PPO hyperparameters, cognitive-parameter sampling, reward settings, curriculum behavior, random seeds, and checkpoint frequency. Each run saves the resolved configuration as `config.json` and `scenario_config.json` in its output directory.

### Training outputs

Training logs and checkpoints are written while training is running. `report.md` is generated only after training finishes successfully. The run directory is organized as:

```text
outputs/training/runs/ppo_expert_reproduction_<timestamp>/
├── config.json
├── scenario_config.json
├── training_logs.csv
├── tensorboard/
├── report.md
└── checkpoints/
    ├── checkpoint_<iteration>.pt
    ├── latest_model.pt
    └── best_model.pt
```

Use `checkpoints/latest_model.pt` for parameter identification and prediction. `best_model.pt` stores the checkpoint with the highest mean evaluation reward.

## Data

The included example is a multi-vehicle trajectory recorded during a human-participant driving takeover experiment conducted in a CARLA scenario. It captures the ego vehicle's motion during the participant's takeover together with surrounding-vehicle motion. The CSV contains simulator-exported timestamps, vehicle IDs, planar velocities, and planar positions. It is used for cognitive-parameter identification and trajectory prediction, not for PPO training.

The example file is located at:

```text
examples/trajectory_scenario/example_trajectory_scene.csv
```

See [DATA.md](DATA.md) for column definitions, units, loader behavior, batch-processing requirements, and data availability.

## Cognitive Parameter Identification and Prediction

Use a trained checkpoint and one trajectory CSV to identify `perception_sigma0` (*σ₀*), `perception_sigma_max` (*σₘₐₓ*), `bias_inverse_tta_coef` (*c*), and `delay_steps` (*d*), and to predict the ego vehicle's future trajectory:

```bash
python parameter_identify/main.py \
  --ppo_model_path outputs/training/runs/ppo_expert_reproduction_<timestamp>/checkpoints/latest_model.pt \
  --data_path examples/trajectory_scenario/example_trajectory_scene.csv \
  --output_dir outputs/identification
```

## Updates

- **2026-07-17:** Released the current public implementation with installation, training, data, cognitive-parameter-identification, and trajectory-prediction instructions.
- **2026-01-28:** Initialized the README for the upcoming code release.

## Citation

```bibtex
@inproceedings{sun2026adaptive,
  author    = {Sun, Jian and Jiang, Xiyan and Zhao, Xiaocong and Wang, Jie and Hang, Peng and Li, Zirui},
  title     = {Adaptive Bounded-Rationality Modeling of Early-Stage Takeover in Shared-Control Driving},
  booktitle = {Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26)},
  year      = {2026},
  location  = {Barcelona, Spain},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  numpages  = {23},
  url       = {https://doi.org/10.1145/3772318.3790701},
  doi       = {10.1145/3772318.3790701}
}
```

## Acknowledgments

This software release builds on widely used open-source tools:

- [**MetaDrive**](https://github.com/metadriverse/metadrive) provides the driving-simulation environment used for PPO training and checkpoint evaluation. This repository installs the pinned public fork described above.
- [**Stable-Baselines3**](https://github.com/DLR-RM/stable-baselines3) provides the vectorized-environment wrappers used by the training pipeline.
- [**PyTorch**](https://pytorch.org/) provides the neural-network, optimization, and checkpointing backend.

The human-participant takeover study described in the paper used **CARLA**, **Unreal Engine**, and **RoadRunner** as part of the experimental driving-simulator stack. These tools are not runtime dependencies of this Python release and are not distributed in this repository.

## License and Attribution

Original code in this repository is released under the MIT License; see [`LICENSE`](LICENSE).

The pinned [`Jiangxy02/metadrive`](https://github.com/Jiangxy02/metadrive/tree/fad0a7e18db3ed899a32e5e42aaa88cbe4c4df52) fork remains third-party Apache-2.0 software. See [`NOTICE`](NOTICE) for attribution.
