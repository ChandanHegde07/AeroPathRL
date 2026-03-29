# AeroPath RL

AeroPath RL is an autonomous drone navigation project using reinforcement learning.
The agent is trained to fly from spawn to target in 3D space while avoiding collisions.

The project uses PPO in an AirSim-based setup (with mock fallback support for development).

## What This Repo Includes

- PPO training pipeline for drone navigation
- Evaluation pipeline for single and batch episodes
- Saved model checkpoints and best/final model artifacts
- `dashboard.html` for visualizing trained/evaluation data

## Project Objective

The drone agent learns to:

- start from spawn position
- interpret state + distance sensor signals
- take continuous control actions
- reach the target safely and efficiently

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Show current configuration:

```bash
python3 main.py info
```

3. Train:

```bash
python3 main.py train --timesteps 200000
```

4. Evaluate and save dashboard-ready data:

```bash
python3 main.py evaluate \
  --model models/saved_models/final_drone_ppo.zip \
  --mode batch \
  --n 100 \
  --save \
  --out_dir eval_results/
```

## Dashboard

Dashboard file:

- `dashboard.html`

Dashboard reads data from:

- `eval_results/eval_stats.json`
- `eval_results/trajectories.csv`
- `logs/*.csv` (optional training log curves)

Run locally:

```bash
python3 -m http.server
```

Then open:

- `http://localhost:8000/dashboard.html`

## Vercel Deployment (Static)

This repository also contains Python code, so Vercel may auto-detect it as a Python app.
To host only the dashboard as static content, keep/use `vercel.json` with static build routing.

Important:

- Commit dashboard data files so Vercel can serve them:
  - `eval_results/eval_stats.json`
  - `eval_results/trajectories.csv`
  - any needed `logs/*.csv`
- If these files are missing in deployment, graphs will be empty.

## Main Commands

- Train: `python3 main.py train`
- Resume training: `python3 main.py train --resume <model_path>`
- Evaluate: `python3 main.py evaluate --model <model_path> --mode batch --n 20`
- Demo run: `python3 main.py demo --model <model_path>`
