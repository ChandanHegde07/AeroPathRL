# AeroPath RL: Autonomous Drone Navigation

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![RL Algorithm](https://img.shields.io/badge/RL-PPO-orange.svg)](#training)
[![Simulator](https://img.shields.io/badge/Simulator-AirSim-00bcd4.svg)](https://github.com/microsoft/AirSim)
[![Code Style: Clean](https://img.shields.io/badge/Code%20Style-Clean-2ea44f.svg)](#)

A deep reinforcement learning pipeline for training a drone to reach a 3D target while avoiding collisions, using **PPO + Gymnasium + AirSim**.

## Quick Facts

| Item | Value |
|---|---|
| Primary Task | Drone target navigation in 3D |
| Algorithm | Proximal Policy Optimization (PPO) |
| Action Space | Continuous velocity `(vx, vy, vz)` |
| Observation Space | 19-D normalized vector |
| Simulator | Microsoft AirSim (with mock fallback) |
| Entry Point | `main.py` |

## Project Structure

| Path | Purpose |
|---|---|
| `config/env_config.py` | Environment, reward, and boundary settings |
| `config/training_config.py` | PPO hyperparameters and training paths |
| `environment/airsim_env.py` | Gymnasium environment + AirSim/mock client |
| `environment/state_processing.py` | Raw state to normalized observation vector |
| `environment/reward_function.py` | Reward shaping and reward breakdown |
| `agent/train.py` | Training orchestration, callbacks, checkpointing |
| `agent/evaluate.py` | Single and batch evaluation modes |
| `agent/model.py` | Actor-critic model and SB3 policy kwargs |
| `utils/logger.py` | CSV + TensorBoard logging utilities |
| `utils/visualization.py` | Training and trajectory plotting helpers |
| `simulations/airsim_settings.json` | AirSim vehicle and sensor config |
| `notebooks/analysis.ipynb` | Analysis notebook |

## Architecture

```text
AirSim/Mock Client
      -> DroneNavigationEnv
      -> StateProcessor (obs[19])
      -> PPO Agent (SB3)
      -> Velocity Action (vx, vy, vz)
      -> RewardFunction (goal/progress/collision/boundary/step/altitude)
```

## Installation

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| CUDA (optional) | 11.8+ |
| Unreal Engine (AirSim usage) | 4.27 |

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## AirSim Setup (Optional)

Use this only if you want full AirSim simulation.

1. Download an AirSim binary from the [official releases](https://github.com/microsoft/AirSim/releases).
2. Copy `simulations/airsim_settings.json` to your AirSim settings path.
3. Start the AirSim environment.
4. The project connects to `127.0.0.1:41451`.

If AirSim is unavailable, the project automatically falls back to a mock client.

## Commands

| Goal | Command |
|---|---|
| Show current config | `python3 main.py info` |
| Start training | `python3 main.py train` |
| Train with custom timesteps | `python3 main.py train --timesteps 200000` |
| Resume training | `python3 main.py train --resume models/saved_models/drone_ppo_ckpt_100000_steps` |
| Evaluate (batch) | `python3 main.py evaluate --model models/saved_models/best_model --mode batch --n 20 --save --out_dir eval_results/` |
| Evaluate (single) | `python3 main.py evaluate --model models/saved_models/best_model --mode single --render` |
| Demo run | `python3 main.py demo --model models/saved_models/best_model` |

## Configuration

### Environment (`config/env_config.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `target_position` | `(30.0, 0.0, -10.0)` | Goal position in NED meters |
| `spawn_position` | `(0.0, 0.0, -5.0)` | Initial drone spawn point |
| `goal_tolerance` | `2.0` | Goal acceptance radius |
| `max_velocity` | `5.0` | Max velocity per axis (m/s) |
| `num_distance_sensors` | `8` | Number of radial distance sensors |
| `max_steps_per_episode` | `500` | Episode length cap |
| `reward_goal_reached` | `200.0` | Success terminal reward |
| `reward_collision` | `-100.0` | Collision terminal penalty |
| `reward_progress_scale` | `5.0` | Progress reward multiplier |

### Training (`config/training_config.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `total_timesteps` | `1_000_000` | Total training steps |
| `learning_rate` | `3e-4` | PPO optimizer LR |
| `n_steps` | `2048` | Rollout length per update |
| `batch_size` | `64` | Minibatch size |
| `n_epochs` | `10` | PPO optimization epochs/update |
| `gamma` | `0.99` | Discount factor |
| `ent_coef` | `0.01` | Entropy regularization |
| `net_arch` | `[256, 256]` | MLP hidden layer sizes |
| `n_envs` | `1` | Parallel environment count |

## Observation and Action Space

### Observation (19-D)

| Slice | Content | Normalization |
|---|---|---|
| `[0:3]` | Relative target position `(dx, dy, dz)` | Divide by boundary diagonal |
| `[3:6]` | Linear velocity `(vx, vy, vz)` | Divide by `max_velocity` |
| `[6:9]` | Orientation `(roll, pitch, yaw)` | Divide by `pi` |
| `[9:17]` | Distance sensor readings (8 values) | Divide by `sensor_max_range` |
| `[17]` | Scalar distance-to-goal | Divide by boundary diagonal |

### Action (3-D)

`Box(-1, 1, shape=(3,))`, scaled to `+-max_velocity` before sending velocity commands.

## Reward Function

```text
R = goal + collision + boundary + progress + step + altitude
```

| Component | Behavior |
|---|---|
| `goal` | Large positive reward when target reached |
| `collision` | Large negative reward on impact |
| `boundary` | Penalty outside allowed bounds |
| `progress` | Reward from reduction in distance-to-goal |
| `step` | Small per-step penalty for efficiency |
| `altitude` | Penalty outside nominal altitude range |

## Training Outputs

| Output | Location |
|---|---|
| Best model | `models/saved_models/best_model.zip` |
| Final model | `models/saved_models/final_drone_ppo.zip` |
| Checkpoints | `models/saved_models/checkpoints/` |
| CSV logs | `logs/run_<timestamp>.csv` |
| TensorBoard logs | `logs/tensorboard/` |

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard/
```

## Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

The notebook supports synthetic fallback data when training logs are not present.

## Extending the Project

| Goal | Change Area |
|---|---|
| Try another algorithm (SAC/TD3) | `training_config.py`, `agent/train.py` |
| Add camera features | `env_config.py`, `state_processing.py` |
| Custom map or obstacle setup | `simulations/airsim_settings.json` |
| Increase parallel rollout | `training_config.py` (`n_envs > 1`) |
| Export policy to ONNX | `agent/model.py` |
| Modify reward shaping | `environment/reward_function.py` |

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: airsim` | Install AirSim or use mock fallback |
| AirSim connection refused | Start AirSim binary first |
| CUDA out of memory | Reduce `batch_size` or use CPU |
| Weak learning progress | Increase timesteps and tune rewards |
| TensorBoard empty | Verify `logs/tensorboard/` path and dependency |


