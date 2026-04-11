#!/usr/bin/env python3
"""Tutorial: Reading and analysing recorded data traces.

This script demonstrates how to:
  1. Load robots_payload.jsonl and planner_paths.jsonl from recorded traces.
  2. Extract per-robot trajectories (position, velocity, reward, etc.).
  3. Compare trajectories across robots, episodes and runs.
  4. Visualise results with matplotlib.

Expected folder layout
----------------------
traces/
  run_001/ … run_010/
    env_000/
      episode_0000/ … episode_0002/
        robots_payload.jsonl
        planner_paths.jsonl
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ────────────────────────────────────────────────────
TRACES_ROOT = Path(__file__).resolve().parent / "traces"
RUNS = [f"run_{i:03d}" for i in range(1, 11)]       # run_001 … run_010
ENV = "env_000"
EPISODES = [f"episode_{i:04d}" for i in range(3)]    # episode_0000 … episode_0002


# =====================================================================
#  1.  Loading helpers
# =====================================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON-Lines file.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of parsed JSON objects. Returns an empty list when the file does
        not exist.
    """
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_episode(episode_dir: Path):
    """Load both JSONL files for one episode.

    Args:
        episode_dir: Episode directory containing recorder outputs.

    Returns
    -------
    robots_rows : list[dict]
        One entry per simulation step.  Each entry has:
          - "step"      : int
          - "wall_time" : float   (seconds since episode start)
          - "robots"    : list of per-robot dicts, each containing:
                robot_position, robot_yaw, robot_velocity,
                target_delta, reward, success, collision,
                target_position (opt), agent_action (opt), …
    planner_rows : list[dict]
        One entry per (step, robot_idx) pair.  Fields:
          - "step"        : int
          - "robot_idx"   : int
          - "global_path" : list of [x, y]  (planned global path)
          - "dwa_traj"    : list of [x, y]  (DWA lookahead trajectory)
          - "p_traj"      : list of [x, y]  (predicted trajectory)
    """
    robots_rows = load_jsonl(episode_dir / "robots_payload.jsonl")
    planner_rows = load_jsonl(episode_dir / "planner_paths.jsonl")
    return robots_rows, planner_rows


# =====================================================================
#  2.  Extracting per-robot time-series from robots_payload
# =====================================================================

def extract_robot_timeseries(
    robots_rows: List[Dict[str, Any]],
    robot_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """Pull scalar time-series for one robot across all steps.

    Returns a dict with numpy arrays keyed by field name:
      step, wall_time, x, y, yaw, vx, vy, speed,
      reward, cumulative_reward, success, collision,
      target_x, target_y, distance_to_target,
      action_v, action_omega  (when available)

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.
        robot_idx: Robot index to extract.

    Returns:
        Dictionary of per-step numpy arrays for the selected robot.
    """
    steps, wall_times = [], []
    xs, ys, yaws = [], [], []
    vxs, vys, speeds = [], [], []
    rewards, successes, collisions = [], [], []
    target_xs, target_ys, dists = [], [], []
    action_vs, action_omegas = [], []

    for row in robots_rows:
        robots = row.get("robots", [])
        if robot_idx >= len(robots):
            continue
        r = robots[robot_idx]

        steps.append(int(row.get("step", len(steps))))
        wall_times.append(float(row.get("wall_time", 0.0)))

        pos = r.get("robot_position", [0.0, 0.0])
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        yaws.append(float(r.get("robot_yaw", 0.0)))

        vel = r.get("robot_velocity", [0.0, 0.0])
        vx, vy = float(vel[0]), float(vel[1])
        vxs.append(vx)
        vys.append(vy)
        speeds.append(math.hypot(vx, vy))

        rewards.append(float(r.get("reward", 0.0)))
        successes.append(int(r.get("success", False)))
        collisions.append(int(r.get("collision", False)))

        tp = r.get("target_position")
        if tp is not None:
            tx, ty = float(tp[0]), float(tp[1])
        else:
            # Fall back to target_delta if target_position unavailable
            td = r.get("target_delta", [0.0, 0.0, 0.0])
            tx = float(pos[0]) + float(td[0])
            ty = float(pos[1]) + float(td[1])
        target_xs.append(tx)
        target_ys.append(ty)
        dists.append(math.hypot(tx - float(pos[0]), ty - float(pos[1])))

        act = r.get("agent_action")
        if act is not None and len(act) >= 2:
            action_vs.append(float(act[0]))
            action_omegas.append(float(act[1]))
        else:
            action_vs.append(float("nan"))
            action_omegas.append(float("nan"))

    result = {
        "step": np.array(steps),
        "wall_time": np.array(wall_times),
        "x": np.array(xs),
        "y": np.array(ys),
        "yaw": np.array(yaws),
        "vx": np.array(vxs),
        "vy": np.array(vys),
        "speed": np.array(speeds),
        "reward": np.array(rewards),
        "cumulative_reward": np.cumsum(rewards),
        "success": np.array(successes),
        "collision": np.array(collisions),
        "target_x": np.array(target_xs),
        "target_y": np.array(target_ys),
        "distance_to_target": np.array(dists),
        "action_v": np.array(action_vs),
        "action_omega": np.array(action_omegas),
    }
    return result


# =====================================================================
#  3.  Extracting planner paths for a given step
# =====================================================================

def get_planner_paths(
    planner_rows: List[Dict[str, Any]],
    step: int,
    robot_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """Return the planned paths for a specific (step, robot_idx).

    Returns dict with keys 'global_path', 'dwa_traj', 'p_traj',
    each an (N, 2) numpy array (or empty (0, 2) if missing).

    Args:
        planner_rows: Parsed rows from planner_paths.jsonl.
        step: Step index to query.
        robot_idx: Robot index to query.

    Returns:
        Mapping containing global path, DWA trajectory, and predicted
        trajectory arrays.
    """
    empty = np.empty((0, 2))
    for row in planner_rows:
        if int(row.get("step", -1)) == step and \
           int(row.get("robot_idx", 0)) == robot_idx:
            return {
                "global_path": np.array(row["global_path"]) if row.get("global_path") else empty.copy(),
                "dwa_traj": np.array(row["dwa_traj"]) if row.get("dwa_traj") else empty.copy(),
                "p_traj": np.array(row["p_traj"]) if row.get("p_traj") else empty.copy(),
            }
    return {"global_path": empty, "dwa_traj": empty, "p_traj": empty}


# =====================================================================
#  4.  How many robots are in this episode?
# =====================================================================

def count_robots(robots_rows: List[Dict[str, Any]]) -> int:
    """Return the number of robots present in the first step.

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.

    Returns:
        Robot count for the first row, or 0 if rows are empty.
    """
    if not robots_rows:
        return 0
    return len(robots_rows[0].get("robots", []))


# =====================================================================
#  5.  Plotting helpers
# =====================================================================

def plot_single_episode_trajectory(
    robots_rows: List[Dict[str, Any]],
    planner_rows: List[Dict[str, Any]],
    title: str = "Episode trajectory",
):
    """Plot XY trajectories, rewards, and distance-to-target curves.

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.
        planner_rows: Parsed rows from planner_paths.jsonl.
        title: Figure title.

    Returns:
        A Matplotlib Figure object.
    """
    n_robots = count_robots(robots_rows)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- (a) XY path ---
    ax = axes[0]
    for ridx in range(n_robots):
        ts = extract_robot_timeseries(robots_rows, ridx)
        ax.plot(ts["x"], ts["y"], label=f"Robot {ridx}")
        ax.scatter(ts["x"][0], ts["y"][0], marker="o", s=60, zorder=5)   # start
        ax.scatter(ts["x"][-1], ts["y"][-1], marker="x", s=60, zorder=5) # end
        # Target position (last known)
        ax.scatter(ts["target_x"][-1], ts["target_y"][-1],
                   marker="*", s=120, zorder=5, label=f"Target {ridx}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("XY Trajectory")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    # --- (b) Reward over time ---
    ax = axes[1]
    for ridx in range(n_robots):
        ts = extract_robot_timeseries(robots_rows, ridx)
        ax.plot(ts["step"], ts["cumulative_reward"], label=f"Robot {ridx}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- (c) Distance to target ---
    ax = axes[2]
    for ridx in range(n_robots):
        ts = extract_robot_timeseries(robots_rows, ridx)
        ax.plot(ts["step"], ts["distance_to_target"], label=f"Robot {ridx}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance to Target")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    return fig


def plot_planner_snapshot(
    robots_rows: List[Dict[str, Any]],
    planner_rows: List[Dict[str, Any]],
    step: int,
    robot_idx: int = 0,
    title: str = "",
):
    """Overlay planner paths on the robot trajectory at a given step.

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.
        planner_rows: Parsed rows from planner_paths.jsonl.
        step: Step to visualize.
        robot_idx: Robot index to visualize.
        title: Optional figure title override.

    Returns:
        A Matplotlib Figure object.
    """
    ts = extract_robot_timeseries(robots_rows, robot_idx)
    paths = get_planner_paths(planner_rows, step, robot_idx)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Full trajectory (faded)
    ax.plot(ts["x"], ts["y"], "k-", alpha=0.2, label="Full trajectory")

    # Trajectory up to this step
    mask = ts["step"] <= step
    ax.plot(ts["x"][mask], ts["y"][mask], "b-", linewidth=2, label="Trajectory so far")

    # Current position
    idx = np.searchsorted(ts["step"], step)
    if idx < len(ts["x"]):
        ax.scatter(ts["x"][idx], ts["y"][idx], c="blue", s=100,
                   zorder=10, label="Current pose")

    # Planner paths
    gp = paths["global_path"]
    if gp.shape[0] > 0:
        ax.plot(gp[:, 0], gp[:, 1], "g--", linewidth=1.5, label="Global path")
    dwa = paths["dwa_traj"]
    if dwa.shape[0] > 0:
        ax.plot(dwa[:, 0], dwa[:, 1], "r-", linewidth=1.5, label="DWA trajectory")
    pt = paths["p_traj"]
    if pt.shape[0] > 0:
        ax.plot(pt[:, 0], pt[:, 1], "m:", linewidth=1.5, label="Predicted trajectory")

    # Target
    if idx < len(ts["target_x"]):
        ax.scatter(ts["target_x"][idx], ts["target_y"][idx],
                   c="gold", marker="*", s=200, zorder=10, label="Target")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title or f"Planner snapshot — step {step}, robot {robot_idx}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_cross_run_comparison(
    run_data: Dict[str, Dict[str, np.ndarray]],
    field: str = "cumulative_reward",
    ylabel: str = "Cumulative Reward",
    title: str = "Cross-run comparison",
):
    """Compare one scalar time-series field across multiple runs.

    Args:
        run_data: Mapping from run label to extracted timeseries dictionary.
        field: Timeseries key to plot.
        ylabel: Y-axis label.
        title: Figure title.

    Returns:
        A Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, ts in run_data.items():
        ax.plot(ts["step"], ts[field], label=label, alpha=0.7)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# =====================================================================
#  6.  Summary statistics
# =====================================================================

def episode_summary(robots_rows: List[Dict[str, Any]], robot_idx: int = 0) -> Dict[str, Any]:
    """Compute summary statistics for one robot in one episode.

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.
        robot_idx: Robot index to summarize.

    Returns:
        Dictionary with episode-level statistics, or an empty dictionary when
        no robot samples are available.
    """
    ts = extract_robot_timeseries(robots_rows, robot_idx)
    n = len(ts["step"])
    if n == 0:
        return {}
    return {
        "num_steps": n,
        "duration_s": float(ts["wall_time"][-1] - ts["wall_time"][0]),
        "total_reward": float(ts["cumulative_reward"][-1]),
        "mean_speed": float(np.nanmean(ts["speed"])),
        "max_speed": float(np.nanmax(ts["speed"])),
        "final_distance_to_target": float(ts["distance_to_target"][-1]),
        "min_distance_to_target": float(np.min(ts["distance_to_target"])),
        "any_collision": bool(np.any(ts["collision"])),
        "any_success": bool(np.any(ts["success"])),
        "path_length": float(np.sum(np.hypot(
            np.diff(ts["x"]), np.diff(ts["y"])
        ))),
    }


# =====================================================================
#  MAIN — run the tutorial
# =====================================================================

def main():
    """Run the end-to-end tutorial workflow."""
    print("=" * 60)
    print("  Warehouse Data Trace Sample Analysis Tutorial")
    print("=" * 60)

    # ------------------------------------------------------------------
    # A. Load a single episode and inspect it
    # ------------------------------------------------------------------
    sample_run = RUNS[0]
    sample_ep = EPISODES[0]
    ep_dir = TRACES_ROOT / sample_run / ENV / sample_ep

    print(f"\n▸ Loading {ep_dir.relative_to(TRACES_ROOT)} …")
    robots_rows, planner_rows = load_episode(ep_dir)
    n_robots = count_robots(robots_rows)
    print(f"  Steps: {len(robots_rows)},  Robots: {n_robots},  "
          f"Planner entries: {len(planner_rows)}")

    # Show first step's fields
    if robots_rows:
        first = robots_rows[0]
        robot0 = first["robots"][0] if first.get("robots") else {}
        print(f"  Fields in robots[0]: {sorted(robot0.keys())}")

    # ------------------------------------------------------------------
    # B. Extract time-series for robot 0 and print summary
    # ------------------------------------------------------------------
    print(f"\n▸ Episode summary for robot 0:")
    summary = episode_summary(robots_rows, robot_idx=0)
    for k, v in summary.items():
        print(f"    {k:30s} = {v}")

    # ------------------------------------------------------------------
    # C. Compare all robots within the same episode
    # ------------------------------------------------------------------
    if n_robots > 1:
        print(f"\n▸ Comparing {n_robots} robots in {sample_run}/{sample_ep}:")
        for ridx in range(n_robots):
            s = episode_summary(robots_rows, ridx)
            print(f"    Robot {ridx}: reward={s['total_reward']:+.2f}  "
                  f"path_len={s['path_length']:.2f}m  "
                  f"collision={s['any_collision']}  "
                  f"success={s['any_success']}")

    # ------------------------------------------------------------------
    # D. Compare one robot across all episodes of a run
    # ------------------------------------------------------------------
    print(f"\n▸ Comparing episodes within {sample_run}:")
    for ep_name in EPISODES:
        ed = TRACES_ROOT / sample_run / ENV / ep_name
        rr, _ = load_episode(ed)
        if rr:
            s = episode_summary(rr, robot_idx=0)
            print(f"    {ep_name}: steps={s['num_steps']:4d}  "
                  f"reward={s['total_reward']:+.2f}  "
                  f"dist_to_target={s['final_distance_to_target']:.3f}m")
        else:
            print(f"    {ep_name}: (no data)")

    # ------------------------------------------------------------------
    # E. Cross-run comparison (robot 0, episode 0)
    # ------------------------------------------------------------------
    print(f"\n▸ Cross-run comparison (robot 0, {EPISODES[0]}):")
    run_data: Dict[str, Dict[str, np.ndarray]] = {}
    for run_name in RUNS:
        ed = TRACES_ROOT / run_name / ENV / EPISODES[0]
        rr, _ = load_episode(ed)
        if rr:
            ts = extract_robot_timeseries(rr, robot_idx=0)
            run_data[run_name] = ts
            s = episode_summary(rr, robot_idx=0)
            print(f"    {run_name}: reward={s['total_reward']:+.2f}  "
                  f"path_len={s['path_length']:.2f}m  "
                  f"success={s['any_success']}")

    # ------------------------------------------------------------------
    # F. Aggregate statistics across all runs
    # ------------------------------------------------------------------
    if run_data:
        all_rewards = [float(ts["cumulative_reward"][-1]) for ts in run_data.values()]
        all_paths = []
        for ts in run_data.values():
            all_paths.append(float(np.sum(np.hypot(
                np.diff(ts["x"]), np.diff(ts["y"])))))
        print(f"\n▸ Aggregate across {len(run_data)} runs:")
        print(f"    Reward   — mean={np.mean(all_rewards):.2f}  "
              f"std={np.std(all_rewards):.2f}  "
              f"min={np.min(all_rewards):.2f}  max={np.max(all_rewards):.2f}")
        print(f"    Path len — mean={np.mean(all_paths):.2f}m  "
              f"std={np.std(all_paths):.2f}m")

    # ------------------------------------------------------------------
    # G. Generate plots (saved to files)
    # ------------------------------------------------------------------
    print("\n▸ Generating plots …")
    out_dir = Path(__file__).resolve().parent / "trace_analysis_output"
    out_dir.mkdir(exist_ok=True)

    # (G1) Single-episode trajectory + reward + distance
    fig = plot_single_episode_trajectory(
        robots_rows, planner_rows,
        title=f"{sample_run} / {sample_ep}",
    )
    fig.savefig(out_dir / "episode_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved episode_trajectory.png")

    # (G2) Planner snapshot at the midpoint step
    if planner_rows:
        mid_step = int(robots_rows[len(robots_rows) // 2].get("step", 0))
        fig = plot_planner_snapshot(
            robots_rows, planner_rows, step=mid_step, robot_idx=0,
            title=f"Planner @ step {mid_step} — {sample_run}/{sample_ep}",
        )
        fig.savefig(out_dir / "planner_snapshot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved planner_snapshot.png (step={mid_step})")

    # (G3) Cross-run reward comparison
    if run_data:
        fig = plot_cross_run_comparison(
            run_data, field="cumulative_reward",
            ylabel="Cumulative Reward",
            title=f"Reward across runs — robot 0, {EPISODES[0]}",
        )
        fig.savefig(out_dir / "cross_run_reward.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved cross_run_reward.png")

        # (G4) Cross-run distance-to-target
        fig = plot_cross_run_comparison(
            run_data, field="distance_to_target",
            ylabel="Distance to Target (m)",
            title=f"Distance to target across runs — robot 0, {EPISODES[0]}",
        )
        fig.savefig(out_dir / "cross_run_distance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved cross_run_distance.png")

        # (G5) All trajectories overlaid
        fig, ax = plt.subplots(figsize=(8, 8))
        for label, ts in run_data.items():
            ax.plot(ts["x"], ts["y"], alpha=0.5, label=label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Trajectory overlay — robot 0, {EPISODES[0]}")
        ax.legend(fontsize=7, ncol=2)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_overlay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved trajectory_overlay.png")

    print(f"\n✓ All outputs saved to {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
