# Autonomous Drone Navigation using Reinforcement Learning

> A production-grade deep RL pipeline for training a drone agent to navigate from a start position to a target in a 3D simulated environment — avoiding obstacles and optimising path efficiency — using **Proximal Policy Optimization (PPO)** and **Microsoft AirSim**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [AirSim Setup](#airsim-setup)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [Observation & Action Space](#observation--action-space)
- [Reward Function](#reward-function)
- [Neural Network Architecture](#neural-network-architecture)
- [TensorBoard Monitoring](#tensorboard-monitoring)
- [Notebook Analysis](#notebook-analysis)
- [Extending the Project](#extending-the-project)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project implements an end-to-end reinforcement learning system where a simulated drone:

1. **Perceives** its environment through position, velocity, orientation and 8 radial distance sensors
2. **Acts** by outputting continuous 3-axis velocity commands `(vx, vy, vz)`
3. **Learns** via PPO to reach a target location while avoiding collisions and staying within bounds

The pipeline works **with or without AirSim** — a lightweight mock client enables full pipeline testing (training, evaluation, notebooks) on any machine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DroneNavigationEnv                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ AirSim/Mock  │→ │StateProcessor│→ │ obs: float32 │  │
│  │   Client     │  │   (norm.)    │  │  vector[19]  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────────────────────────────────────────┐   │
│  │              RewardFunction                      │   │
│  │  goal + progress + collision + boundary + step   │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
              │  obs            ▲  action (vx,vy,vz)
              ▼                 │
┌─────────────────────────────────────────────────────────┐
│                  PPO Agent (SB3)                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Encoder [256,256] → Actor [128] / Critic [128]  │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
drone_rl_navigation/
│
├── config/
│   ├── env_config.py          # Environment hyperparameters (target pos, rewards…)
│   └── training_config.py     # PPO hyperparameters, paths, schedule
│
├── environment/
│   ├── airsim_env.py          # Custom Gymnasium environment (+ mock client)
│   ├── state_processing.py    # Raw AirSim data → normalised obs vector
│   └── reward_function.py     # Modular, logged reward shaping
│
├── agent/
│   ├── model.py               # Standalone PyTorch actor-critic + SB3 factory
│   ├── train.py               # DroneTrainer: PPO loop, callbacks, checkpointing
│   └── evaluate.py            # DroneEvaluator: single / batch / record modes
│
├── utils/
│   ├── logger.py              # CSV + TensorBoard training logger
│   └── visualization.py       # Reward curves, 3-D trajectories, sensor heat-maps
│
├── simulations/
│   ├── airsim_settings.json   # AirSim vehicle + 8 distance-sensor configuration
│   └── environments/          # Custom UE4 map files (optional)
│
├── models/
│   └── saved_models/          # Best & final checkpoints written here
│
├── notebooks/
│   └── analysis.ipynb         # Full training analysis (works with synthetic data)
│
├── main.py                    # Unified CLI entry point
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| CUDA (optional) | ≥ 11.8 |
| Unreal Engine | 4.27 (for AirSim) |

### 1. Clone & create environment

```bash
git clone https://github.com/your-org/drone_rl_navigation.git
cd drone_rl_navigation

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Without a GPU:** PyTorch will fall back to CPU automatically. Training will be slower but fully functional.

---

## AirSim Setup

> **Skip this section if you just want to test the pipeline** — the mock environment works without AirSim.

1. Download a prebuilt AirSim binary from the [AirSim releases page](https://github.com/microsoft/AirSim/releases) (e.g. `Blocks` environment).
2. Copy `simulations/airsim_settings.json` to:
   - **Windows:** `C:\Users\<user>\Documents\AirSim\settings.json`
   - **Linux/macOS:** `~/Documents/AirSim/settings.json`
3. Launch the AirSim binary.
4. The environment will auto-connect to `127.0.0.1:41451`.

---

## Quick Start

```bash
# Print current config
python main.py info

# Train for 200k steps (uses mock env if AirSim is not running)
python main.py train --timesteps 200000

# Evaluate the best saved model (20 episodes)
python main.py evaluate --model models/saved_models/best_model --n 20 --save

# Run a single rendered demo episode
python main.py demo --model models/saved_models/best_model

# Resume interrupted training
python main.py train --resume models/saved_models/drone_ppo_ckpt_100000_steps
```

---

## Configuration

All hyperparameters are centralised in two dataclasses:

### `config/env_config.py` — Environment

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_position` | `(30, 0, -10)` | Goal position in NED metres |
| `max_velocity` | `5.0` m/s | Maximum velocity per axis |
| `goal_tolerance` | `2.0` m | Acceptance radius around target |
| `max_steps_per_episode` | `500` | Episode length cap |
| `num_distance_sensors` | `8` | Radial proximity sensors |
| `reward_goal_reached` | `+200` | Terminal success reward |
| `reward_collision` | `−100` | Terminal collision penalty |
| `reward_progress_scale` | `5.0` | Scales Δdistance reward |

### `config/training_config.py` — PPO

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | `1 000 000` | Total environment steps |
| `learning_rate` | `3e-4` | Adam learning rate |
| `n_steps` | `2048` | Steps per PPO update |
| `batch_size` | `64` | Mini-batch size |
| `n_epochs` | `10` | Gradient update passes |
| `gamma` | `0.99` | Discount factor |
| `ent_coef` | `0.01` | Entropy regularisation |
| `net_arch` | `[256, 256]` | Shared encoder layer sizes |

---

## Training

```bash
python main.py train
```

Training produces:
- `models/saved_models/best_model.zip` — best checkpoint by eval reward
- `models/saved_models/final_drone_ppo.zip` — model at training end
- `models/saved_models/checkpoints/` — periodic checkpoints
- `logs/run_<timestamp>.csv` — per-episode metrics
- `logs/tensorboard/` — TensorBoard event files

### Curriculum Learning (optional)

Enable in `training_config.py`:

```python
use_curriculum = True
curriculum_stages = [
    {"timesteps": 0,        "target_distance": 10.0, "num_obstacles": 2},
    {"timesteps": 200_000,  "target_distance": 20.0, "num_obstacles": 5},
    {"timesteps": 500_000,  "target_distance": 30.0, "num_obstacles": 10},
]
```

---

## Evaluation

```bash
# Batch evaluation (20 episodes) + save trajectories
python main.py evaluate \
    --model models/saved_models/best_model \
    --mode batch --n 20 --save --out_dir eval_results/

# Single episode with step-by-step console output
python main.py evaluate \
    --model models/saved_models/best_model \
    --mode single --render
```

Output files in `eval_results/`:
- `eval_stats.json` — aggregate metrics (success rate, mean reward, …)
- `trajectories.csv` — per-step (x, y, z, reward) for every episode

---

## Observation & Action Space

### Observation vector — 19 dimensions

| Slice | Content | Normalisation |
|-------|---------|---------------|
| `[0:3]` | Relative position to target `(dx, dy, dz)` | ÷ bounding-box diagonal |
| `[3:6]` | Linear velocity `(vx, vy, vz)` | ÷ max_velocity |
| `[6:9]` | Orientation `(roll, pitch, yaw)` | ÷ π |
| `[9:17]` | 8 distance-sensor readings | ÷ sensor_max_range |
| `[17]` | Scalar distance to target | ÷ bounding-box diagonal |

### Action space — 3 dimensions

Continuous `Box(−1, 1, shape=(3,))` → scaled to `±max_velocity` m/s before sending to AirSim.

---

## Reward Function

```
R(t) = R_goal  (terminal +200 on success)
     + R_collision  (terminal −100 on crash)
     + R_boundary   (−50 if out of navigable box)
     + R_progress   (reward_progress_scale × Δdistance_to_goal)
     + R_step       (−0.1 per step)
     + R_altitude   (linear penalty outside [2, 25] m)
```

All coefficients are configurable via `EnvConfig`.

---

## Neural Network Architecture

```
Input: obs[19]
    │
    ▼
┌─────────────────────────┐
│  Shared Encoder          │
│  Linear(19→256) + LN + Tanh  │
│  Linear(256→256) + LN + Tanh │
└─────────┬───────────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
 Actor          Critic
 Linear(256→128) + Tanh    Linear(256→128) + Tanh
 Linear(128→3) → tanh      Linear(128→1)
 + learnable log_std
```

Weights initialised with **orthogonal initialisation** (gain=0.01 for actor output, 1.0 elsewhere).

---

## TensorBoard Monitoring

```bash
tensorboard --logdir logs/tensorboard/
```

Tracked scalars:
- `train/` — PPO loss, value loss, entropy, clip fraction (SB3 defaults)
- `drone/mean_episode_reward_50` — rolling 50-episode mean
- `drone/goal_rate` — cumulative goal-success fraction
- `drone/collision_rate` — cumulative collision fraction
- `eval/mean_reward` — EvalCallback reward

---

## Notebook Analysis

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

The notebook runs entirely on **synthetic data** when no training logs are present, making it safe to explore before training.

Sections:
1. Load training CSV logs
2. Reward learning curve
3. Goal-success vs collision-rate trends
4. Reward component stacked area chart
5. Interactive 3-D flight trajectory
6. Distance-sensor heat-map
7. Batch evaluation statistics
8. Environment + policy sanity check

---

## Extending the Project

| Goal | Where to change |
|------|----------------|
| Different algorithm (SAC, TD3) | `training_config.py` → `algorithm`, `agent/train.py` |
| Add camera observations | `env_config.py` → `use_camera=True`, update `state_processing.py` |
| Custom obstacle maps | `simulations/environments/` + update `airsim_settings.json` |
| Multi-agent training | Wrap env with SB3 `SubprocVecEnv`, set `n_envs > 1` |
| ONNX export for deployment | `model.DroneActorCritic.export_onnx(path, obs_dim)` |
| Custom reward shaping | Edit `environment/reward_function.py` |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: airsim` | Install AirSim or run without it — mock client activates automatically |
| `ConnectionRefusedError` on port 41451 | Start the AirSim binary before running training |
| CUDA out-of-memory | Reduce `batch_size` or set `device="cpu"` in `training_config.py` |
| Drone doesn't learn | Increase `total_timesteps`, check reward scale, try `ent_coef=0.05` |
| TensorBoard not showing | Run `pip install tensorboard` and check `logs/tensorboard/` path |

---

