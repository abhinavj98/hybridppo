# Hybrid PPO

A reinforcement learning framework that combines online PPO training with offline behavioral cloning using Minari datasets.

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

## Overview

This project implements hybrid offline-online reinforcement learning algorithms:
- **PPOExpert**: Online PPO mixed with offline transitions using V-trace-style advantages and out-of-distribution down-weighting
- **PPOBC**: Standard online PPO with behavioral cloning loss from offline expert data

Built on Stable-Baselines3 with Minari dataset integration for seamless offline-to-online transfer.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

```bash
uv sync
```

This creates a virtual environment and installs all dependencies from the lock file.

## Quick Start

### Automated Pipeline (Recommended)
Train BC then hybrid PPO automatically:
```bash
python training_files/train_hybrid_pipeline.py --dataset mujoco --env walker2d --names expert-v0 --seed 42
```

### Manual Training
1. Train behavioral cloning:
```bash
python training_files/train_bc_expert.py --dataset mujoco --env walker2d --names expert-v0 --hparam Walker2d-v4-bc-large --bc_epochs 50
```

2. Train hybrid PPO:
```bash
python training_files/test_ppo_expert.py --dataset mujoco --env walker2d --names expert-v0 --hparam Walker2d-v4-hybrid --bc_policy bc_checkpoints/mujoco/walker2d/best_checkpoint.zip
```

### Evaluation
```bash
python eval_model.py --dataset mujoco --env walker2d --name expert-v0 --algo PPOExpert --model_path path/to/model.zip
```

## Project Structure

- `hybridppo/`: Core algorithms and utilities
  - `ppo_expert.py`: PPOExpert implementation
  - `ppo_bc.py`: PPOBC implementation
  - `policies.py`: Custom actor-critic policies
  - `minari_helpers.py`: Dataset loading and streaming utilities
- `training_files/`: Training scripts for different algorithms
- `bc_checkpoints/`: Saved behavioral cloning models
- `logs/`: Training logs and evaluation results

## Key Features

- V-trace advantage estimation for offline-online mixing
- Out-of-distribution down-weighting using log-prob ratios
- Automatic dataset resolution and environment setup
- WandB and TensorBoard logging
- Hyperparameter configurations in `hparam.yml`

## Datasets

Uses Minari datasets for offline expert demonstrations. Datasets are automatically downloaded and cached locally.