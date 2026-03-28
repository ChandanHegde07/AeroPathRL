"""
utils/visualization.py
Plotting utilities for training analysis and flight-path visualisation.

Functions
─────────
  plot_training_rewards     – episode reward curve with moving average
  plot_reward_components    – stacked area chart of reward breakdown
  plot_episode_trajectory   – 3-D flight path with start / goal markers
  plot_evaluation_summary   – bar chart of batch evaluation metrics
  plot_sensor_heatmap       – heat-map of distance-sensor activations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#c8d0e0",
    "xtick.color":      "#8891a5",
    "ytick.color":      "#8891a5",
    "text.color":       "#c8d0e0",
    "grid.color":       "#2a2d3d",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "lines.linewidth":  1.8,
    "font.family":      "monospace",
})

_PALETTE = ["#4fc3f7", "#81c784", "#ff8a65", "#ba68c8", "#ffd54f", "#f06292"]



def plot_training_rewards(
    rewards: List[float],
    window: int = 50,
    save_path: Optional[str] = None,
    title: str = "Training Reward Curve",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, color=_PALETTE[0], alpha=0.35, lw=1, label="Episode reward")

    if len(rewards) >= window:
        ma = pd.Series(rewards).rolling(window).mean().values
        ax.plot(episodes, ma, color=_PALETTE[1], lw=2.2, label=f"{window}-ep MA")

    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    _maybe_save(fig, save_path)
    return fig


def plot_reward_components(
    component_history: List[Dict[str, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked area chart showing per-step reward components.

    Parameters
    ----------
    component_history : List of dicts from RewardInfo.as_dict()
    """
    df = pd.DataFrame(component_history)
    cols = [c for c in df.columns if c != "reward/total"]
    df = df[cols].clip(lower=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(
        np.arange(len(df)),
        [df[c].values for c in cols],
        labels=[c.split("/")[-1] for c in cols],
        colors=_PALETTE[:len(cols)],
        alpha=0.75,
    )
    ax.set_title("Reward Component Breakdown (positive part)", fontsize=13, pad=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward contribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig



def plot_episode_trajectory(
    positions: List[Tuple[float, float, float]],
    target: Tuple[float, float, float],
    spawn: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    title: str = "Flight Trajectory",
    save_path: Optional[str] = None,
) -> plt.Figure:
    xs, ys, zs = zip(*positions) if positions else ([], [], [])

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")

    n = len(xs)
    colours = plt.cm.cool(np.linspace(0, 1, max(n, 1)))
    for i in range(max(n - 1, 0)):
        ax.plot(
            [xs[i], xs[i + 1]],
            [ys[i], ys[i + 1]],
            [zs[i], zs[i + 1]],
            color=colours[i], lw=1.5,
        )

    ax.scatter(*spawn,  s=120, c="#ffd54f", marker="^", zorder=5, label="Spawn")
    ax.scatter(*target, s=180, c="#81c784", marker="*", zorder=5, label="Target")
    if positions:
        ax.scatter(*positions[-1], s=80, c="#f06292", marker="x", zorder=5,
                   label="End position")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig



def plot_evaluation_summary(
    stats: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    keys = [k for k in stats if isinstance(stats[k], float)]
    vals = [stats[k] for k in keys]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.barh(keys, vals, color=_PALETTE[:len(keys)], alpha=0.85, height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, color="#c8d0e0", fontsize=9)
    ax.set_title("Evaluation Metrics", fontsize=13, pad=10)
    ax.set_xlabel("Value")
    ax.grid(True, axis="x")
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig



def plot_sensor_heatmap(
    sensor_history: np.ndarray,
    sensor_max: float = 20.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(
        sensor_history.T,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        vmin=0,
        vmax=sensor_max,
    )
    plt.colorbar(im, ax=ax, label="Distance (m)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Sensor index")
    ax.set_title("Distance Sensor Readings Over Episode", fontsize=13, pad=10)
    fig.tight_layout()
    _maybe_save(fig, save_path)
    return fig



def _maybe_save(fig: plt.Figure, path: Optional[str]):
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
