from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ── Stable-Baselines3 ────────────────────────────────────────────────────────
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecNormalize,
)

from config.env_config import EnvConfig, ENV_CONFIG
from config.training_config import TrainingConfig, TRAIN_CONFIG
from agent.model import build_sb3_policy_kwargs
from environment.airsim_env import DroneNavigationEnv
from utils.logger import TrainingLogger

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Custom callback
# ─────────────────────────────────────────────────────────────────────────────

class DroneTrainingCallback(BaseCallback):
    """
    Logs episode reward, goal-rate, collision-rate to TensorBoard and console.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards: list[float] = []
        self._goal_count  = 0
        self._crash_count = 0
        self._ep_count    = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:  # Monitor wrapper signals end-of-episode
                ep_r = info["episode"]["r"]
                self._episode_rewards.append(ep_r)
                self._ep_count += 1

            if info.get("goal_reached"):
                self._goal_count += 1
            if info.get("collision"):
                self._crash_count += 1

        if self.n_calls % self.log_freq == 0 and self._episode_rewards:
            mean_r = np.mean(self._episode_rewards[-50:])
            total  = max(self._ep_count, 1)
            self.logger.record("drone/mean_episode_reward_50", mean_r)
            self.logger.record("drone/goal_rate",    self._goal_count / total)
            self.logger.record("drone/collision_rate", self._crash_count / total)
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_env(cfg: EnvConfig, rank: int = 0, seed: int = 0):
    """Return a callable that creates a monitored DroneNavigationEnv."""
    def _init():
        env = DroneNavigationEnv(cfg=cfg)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(
    cfg: EnvConfig,
    n_envs: int,
    seed: int,
    normalise: bool = False,
) -> DummyVecEnv | SubprocVecEnv | VecNormalize:
    fns   = [_make_env(cfg, rank=i, seed=seed) for i in range(n_envs)]
    vec   = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    if normalise:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True)
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# Trainer class
# ─────────────────────────────────────────────────────────────────────────────

class DroneTrainer:
    """
    High-level training orchestrator.

    Parameters
    ----------
    env_cfg   : Environment configuration.
    train_cfg : Training / algorithm hyperparameters.
    """

    def __init__(
        self,
        env_cfg: EnvConfig = ENV_CONFIG,
        train_cfg: TrainingConfig = TRAIN_CONFIG,
    ):
        self.env_cfg   = env_cfg
        self.train_cfg = train_cfg
        self._logger   = TrainingLogger(log_dir=str(train_cfg.log_dir))

        # Create directories
        train_cfg.log_dir.mkdir(parents=True, exist_ok=True)
        train_cfg.model_dir.mkdir(parents=True, exist_ok=True)
        train_cfg.tensorboard_log.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None) -> PPO:
        self._print_header()

        # ── Environments ─────────────────────────────────────────────────────
        console.print("[cyan]Building training environment…[/cyan]")
        train_env = build_vec_env(
            self.env_cfg,
            n_envs=self.train_cfg.n_envs,
            seed=self.train_cfg.seed,
        )
        eval_env = DummyVecEnv([_make_env(self.env_cfg, seed=self.train_cfg.seed + 999)])

        # ── Model ─────────────────────────────────────────────────────────────
        policy_kwargs = build_sb3_policy_kwargs(
            net_arch=self.train_cfg.net_arch,
            activation_fn_name=self.train_cfg.activation_fn,
        )
        if resume_from:
            console.print(f"[yellow]Resuming from {resume_from}[/yellow]")
            model = PPO.load(resume_from, env=train_env, device=self.train_cfg.device)
        else:
            model = self._build_model(train_env, policy_kwargs)

        console.print(
            f"[green]Policy parameters: "
            f"{sum(p.numel() for p in model.policy.parameters()):,}[/green]"
        )

        # ── Callbacks ─────────────────────────────────────────────────────────
        callbacks = self._build_callbacks(eval_env)

        # ── Training ──────────────────────────────────────────────────────────
        console.print(
            Panel(
                f"[bold]Starting PPO training[/bold]\n"
                f"Total timesteps : {self.train_cfg.total_timesteps:,}\n"
                f"Parallel envs   : {self.train_cfg.n_envs}\n"
                f"Device          : {self.train_cfg.device}",
                title="🚁  Drone RL Training",
                border_style="green",
            )
        )
        t0 = time.time()
        model.learn(
            total_timesteps=self.train_cfg.total_timesteps,
            callback=callbacks,
            tb_log_name="PPO_drone",
            reset_num_timesteps=resume_from is None,
        )
        elapsed = time.time() - t0

        # ── Save final model ──────────────────────────────────────────────────
        final_path = self.train_cfg.model_dir / self.train_cfg.final_model_name
        model.save(str(final_path))
        console.print(
            f"[bold green]Training complete in {elapsed/60:.1f} min.[/bold green]\n"
            f"Final model → [cyan]{final_path}[/cyan]"
        )

        train_env.close()
        eval_env.close()
        return model

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_model(self, env, policy_kwargs: dict) -> PPO:
        tc = self.train_cfg
        return PPO(
            policy=tc.policy_type,
            env=env,
            learning_rate=tc.learning_rate,
            n_steps=tc.n_steps,
            batch_size=tc.batch_size,
            n_epochs=tc.n_epochs,
            gamma=tc.gamma,
            gae_lambda=tc.gae_lambda,
            clip_range=tc.clip_range,
            clip_range_vf=tc.clip_range_vf,
            ent_coef=tc.ent_coef,
            vf_coef=tc.vf_coef,
            max_grad_norm=tc.max_grad_norm,
            use_sde=tc.use_sde,
            sde_sample_freq=tc.sde_sample_freq,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tc.tensorboard_log),
            device=tc.device,
            verbose=tc.verbose,
            seed=tc.seed,
        )

    def _build_callbacks(self, eval_env) -> CallbackList:
        tc = self.train_cfg
        checkpoint_cb = CheckpointCallback(
            save_freq=max(tc.save_freq // tc.n_envs, 1),
            save_path=str(tc.model_dir / "checkpoints"),
            name_prefix="drone_ppo_ckpt",
            verbose=1,
        )
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(tc.model_dir),
            log_path=str(tc.log_dir / "eval"),
            eval_freq=max(tc.eval_freq // tc.n_envs, 1),
            n_eval_episodes=tc.n_eval_episodes,
            deterministic=tc.deterministic_eval,
            render=False,
            verbose=1,
        )
        custom_cb = DroneTrainingCallback(log_freq=1000)
        return CallbackList([checkpoint_cb, eval_cb, custom_cb])

    def _print_header(self):
        table = Table(title="Configuration", border_style="blue")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        rows = [
            ("Algorithm",      self.train_cfg.algorithm),
            ("Learning rate",  str(self.train_cfg.learning_rate)),
            ("Total timesteps",f"{self.train_cfg.total_timesteps:,}"),
            ("Batch size",     str(self.train_cfg.batch_size)),
            ("N epochs",       str(self.train_cfg.n_epochs)),
            ("Net arch",       str(self.train_cfg.net_arch)),
            ("Target pos",     str(self.env_cfg.target_position)),
            ("Max velocity",   f"{self.env_cfg.max_velocity} m/s"),
        ]
        for r in rows:
            table.add_row(*r)
        console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train drone RL agent")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to saved model to resume from")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    if args.timesteps:
        TRAIN_CONFIG.total_timesteps = args.timesteps

    trainer = DroneTrainer()
    trainer.train(resume_from=args.resume)
