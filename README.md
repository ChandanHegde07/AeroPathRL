# AeroPath RL

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-blue?logo=gym)](https://gymnasium.farama.org)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.2.1-green?logo=stable-baselines3)](https://stable-baselines3.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

AeroPath RL is an autonomous drone navigation project using reinforcement learning. The agent is trained to fly from spawn to target in 3D space while avoiding collisions using PPO with a local simulator backend.

## Why Reinforcement Learning?

Traditional drone navigation relies on rule-based systems or path planning algorithms (A*, RRT) that require pre-defined maps and explicit obstacle modeling. Reinforcement learning offers:

- **Adaptive Behavior**: The agent learns to handle dynamic, unknown environments without explicit programming
- **End-to-End Learning**: Maps raw sensor observations directly to control actions
- **Collision Avoidance**: Learns safe navigation through trial and error with reward shaping
- **Generalization**: Can adapt to different drone configurations and environments

## What This Repo Includes

- PPO training pipeline for drone navigation
- Evaluation pipeline for single and batch episodes
- Saved model checkpoints and best/final model artifacts
- `dashboard.html` for visualizing trained/evaluation data

## Project Objective

The drone agent learns to:

- Start from spawn position
- Interpret state + distance sensor signals
- Take continuous control actions
- Reach the target safely and efficiently

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Show Configuration

```bash
python3 main.py info
```

### Train

```bash
python3 main.py train --timesteps 200000
```

### Evaluate

```bash
python3 main.py evaluate \
  --model models/saved_models/final_drone_ppo.zip \
  --mode batch \
  --n 100 \
  --save \
  --out_dir eval_results/
```

## Dashboard

Dashboard file: `dashboard.html`

Dashboard reads data from:

- `eval_results/eval_stats.json`
- `eval_results/trajectories.csv`
- `logs/*.csv` (optional training log curves)

Run locally:

```bash
python3 -m http.server
```

Then open: `http://localhost:8000/dashboard.html`

## Main Commands

- Train: `python3 main.py train`
- Resume training: `python3 main.py train --resume <model_path>`
- Evaluate: `python3 main.py evaluate --model <model_path> --mode batch --n 20`
- Single episode with live 2D simulation: `python3 main.py evaluate --model <model_path> --mode single --render2d`
- Demo run: `python3 main.py demo --model <model_path>`