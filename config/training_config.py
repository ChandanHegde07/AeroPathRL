"""
config/training_config.py
Hyperparameters and paths for the PPO training run.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # ── Algorithm ────────────────────────────────────────────────────────────
    algorithm: str = "PPO"          # "PPO" | "SAC" | "TD3"

    # ── PPO hyperparameters ───────────────────────────────────────────────────
    learning_rate: float = 3e-4
    n_steps: int = 2048             # steps collected per update
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99             # discount factor
    gae_lambda: float = 0.95        # GAE-lambda
    clip_range: float = 0.2
    clip_range_vf: float = None     # None → no value-function clipping
    ent_coef: float = 0.01          # entropy bonus
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False           # State-Dependent Exploration
    sde_sample_freq: int = -1

    # ── Training schedule ────────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000         # eval every N timesteps
    n_eval_episodes: int = 5
    save_freq: int = 50_000         # checkpoint every N timesteps

    # ── Network architecture ─────────────────────────────────────────────────
    policy_type: str = "MlpPolicy"  # "MlpPolicy" | "CnnPolicy" | "MultiInputPolicy"
    net_arch: list = field(default_factory=lambda: [256, 256])
    activation_fn: str = "tanh"     # "tanh" | "relu"

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_envs: int = 1                 # parallel environments (set >1 for SubprocVecEnv)

    # ── Paths ─────────────────────────────────────────────────────────────────
    log_dir: Path = Path("logs/")
    model_dir: Path = Path("models/saved_models/")
    tensorboard_log: Path = Path("logs/tensorboard/")
    best_model_name: str = "best_drone_ppo"
    final_model_name: str = "final_drone_ppo"

    # ── Device ────────────────────────────────────────────────────────────────
    device: str = "auto"            # "auto" | "cpu" | "cuda"

    # ── Curriculum learning ───────────────────────────────────────────────────
    use_curriculum: bool = False
    curriculum_stages: list = field(default_factory=lambda: [
        {"timesteps": 0,        "target_distance": 10.0, "num_obstacles": 2},
        {"timesteps": 200_000,  "target_distance": 20.0, "num_obstacles": 5},
        {"timesteps": 500_000,  "target_distance": 30.0, "num_obstacles": 10},
    ])

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed: int = 42
    verbose: int = 1                # SB3 verbosity (0=silent,1=info,2=debug)
    deterministic_eval: bool = True


# Singleton used throughout the project
TRAIN_CONFIG = TrainingConfig()
