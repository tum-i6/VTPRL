#!/usr/bin/env python3
"""Comprehensive analysis toolkit for recorded mobile-robot navigation traces.

This script provides a research-grade analysis pipeline for single-agent and
multi-agent mobile robot navigation experiments recorded by the VTPRL
simulator's ``DataTraceRecorder``.  It reads **all** recorded data channels
(JSONL scalars, NPZ sensor arrays, planner paths, agent states, images) and
computes a wide catalogue of metrics drawn from the robotics navigation
literature.

Capabilities
------------
* **Auto-discovery** — automatically finds every run / env / episode in a
  traces folder regardless of count.
* **Single-agent metrics** — success rate, collision rate, SPL (Success
  weighted by Path Length), path efficiency, smoothness, jerk, time-to-goal,
  clearance from obstacles (from laser), energy proxy, reward curves.
* **Multi-agent metrics** — inter-robot distance, collision avoidance,
  formation coherence, cooperative performance, fairness (Jain index).
* **Planner analysis** — path tracking error, global-vs-executed path
  deviation, DWA trajectory quality, planning horizon statistics.
* **Sensor / state analysis** — laser scan coverage, occupancy grid
  evolution, agent state distributions, costmap statistics.
* **Statistical analysis** — per-run / per-episode / cross-run aggregation,
  confidence intervals, hypothesis tests, effect sizes.
* **Visualisation** — publication-quality figures (trajectory maps, box
  plots, heatmaps, radar charts, learning curves, sensor overlays).
* **Report generation** — Markdown summary with tables and embedded figures.

Usage
-----
.. code-block:: bash

    # Analyse all traces with default settings
    python warehouse_trace_analysis_toolkit.py

    # Custom traces root and output directory
    python warehouse_trace_analysis_toolkit.py --traces ./my_traces --output ./results

    # Filter specific runs, limit analysis
    python warehouse_trace_analysis_toolkit.py --runs run_001 run_005 --no-sensor

References
----------
* Anderson et al. "On Evaluation of Embodied Navigation Agents" (2018)
* Mavrogiannis et al. "Core Challenges of Social Robot Navigation" (2023)
* Francis et al. "Principles and Guidelines for Evaluating Social Robot
  Navigation Algorithms" (2023)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Optional imports — degrade gracefully when unavailable.
try:
    import scipy.stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Use non-interactive backend when saving figures headless.
matplotlib.use("Agg")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.titlesize": 13,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

logger = logging.getLogger(__name__)

# =====================================================================
#  §1  CONFIGURATION
# =====================================================================

@dataclass
class AnalysisConfig:
    """All tuneable knobs for the analysis pipeline.

    Override at construction time or via CLI arguments.
    """

    #: Root folder containing run_XXX directories.
    traces_root: Path = Path("traces")

    #: Output directory for figures and the Markdown report.
    output_dir: Path = Path("trace_analysis_output")

    #: Optional list of run names to include (``None`` = all discovered).
    run_filter: Optional[List[str]] = None

    #: Optional list of env names to include (``None`` = all discovered).
    env_filter: Optional[List[str]] = None

    #: Simulation timestep (seconds) — used to convert steps to real time
    #: when ``wall_time`` is unavailable.  Overridden from metadata when
    #: possible.
    dt: float = 0.04

    # ── feature flags ────────────────────────────────────────────────

    #: Compute sensor-level metrics (laser clearance, occupancy analysis).
    enable_sensor_analysis: bool = True

    #: Compute agent-state / observation analysis.
    enable_state_analysis: bool = True

    #: Decode and display camera images (requires Pillow).
    enable_image_analysis: bool = False

    #: Generate all plots.
    enable_plots: bool = True

    #: Generate Markdown report.
    enable_report: bool = True

    #: Confidence level for bootstrap intervals.
    confidence_level: float = 0.95

    #: Number of bootstrap resamples for confidence intervals.
    n_bootstrap: int = 1000

    #: Minimum clearance (m) considered "near-collision".
    near_collision_threshold: float = 0.15

    #: Success is defined by reaching within this distance of target (m).
    success_distance_threshold: float = 0.3

    #: Figure format (png, pdf, svg).
    fig_format: str = "png"


# =====================================================================
#  §2  DATA LOADING
# =====================================================================

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
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
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_npz(path: Path) -> Optional[np.lib.npyio.NpzFile]:
    """Load an NPZ archive when available and valid.

    Args:
        path: Path to an .npz file.

    Returns:
        An opened NPZ handle, or None when the file is missing/corrupt.
    """
    if not path.exists():
        return None
    try:
        return np.load(str(path), allow_pickle=False)
    except Exception as exc:          # BadZipFile, truncated files, etc.
        logger.info("Skipping corrupt NPZ %s: %s", path.name, exc)
        return None


def _load_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load run-level metadata.json.

    Args:
        run_dir: Run directory that may contain metadata.json.

    Returns:
        Parsed metadata dictionary, or an empty dictionary if unavailable.
    """
    p = run_dir / "metadata.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _finite_float64(values: np.ndarray) -> np.ndarray:
    """Convert an array-like input to finite float64 values.

    Args:
        values: Numeric array-like values.

    Returns:
        Flattened float64 array containing only finite values.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _safe_mean(values: np.ndarray) -> float:
    """Compute a numerically stable mean.

    Args:
        values: Numeric array-like values.

    Returns:
        Mean of finite values as float, or NaN when no finite values exist.
    """
    arr = _finite_float64(values)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr, dtype=np.float64))


def _safe_std(values: np.ndarray) -> float:
    """Compute a numerically stable standard deviation.

    Args:
        values: Numeric array-like values.

    Returns:
        Standard deviation of finite values as float, or NaN when no finite
        values exist.
    """
    arr = _finite_float64(values)
    if arr.size == 0:
        return float("nan")
    return float(np.std(arr, dtype=np.float64))


# ── auto-discovery ───────────────────────────────────────────────────

def discover_structure(
    traces_root: Path,
    run_filter: Optional[List[str]] = None,
    env_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, Path]]:
    """Walk the traces tree and return (run, env, episode, path) tuples.

    The discovery is fully generic: any number of runs, envs and episodes.

    Args:
        traces_root: Root traces directory containing run folders.
        run_filter: Optional list of run names to include.
        env_filter: Optional list of environment names to include.

    Returns:
        List of tuples as (run_name, env_name, episode_name, episode_path).
    """
    if not traces_root.is_dir():
        logger.warning("Traces root does not exist: %s", traces_root)
        return []

    results: List[Tuple[str, str, str, Path]] = []
    for run_dir in sorted(traces_root.iterdir()):
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        if run_filter and run_name not in run_filter:
            continue
        for env_dir in sorted(run_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            env_name = env_dir.name
            if env_filter and env_name not in env_filter:
                continue
            for ep_dir in sorted(env_dir.iterdir()):
                if not ep_dir.is_dir():
                    continue
                ep_name = ep_dir.name
                results.append((run_name, env_name, ep_name, ep_dir))
    return results


# ── episode data bundle ──────────────────────────────────────────────

@dataclass
class EpisodeData:
    """All data loaded for a single episode."""

    run: str
    env: str
    episode: str
    path: Path

    robots_rows: List[Dict[str, Any]] = field(default_factory=list)
    planner_rows: List[Dict[str, Any]] = field(default_factory=list)
    item_rows: List[Dict[str, Any]] = field(default_factory=list)

    # NPZ archives (lazy-loaded, may be None)
    laser_npz: Optional[Any] = None
    laser_pts_npz: Optional[Any] = None
    occ_npz: Optional[Any] = None
    cost_npz: Optional[Any] = None
    nav_npz: Optional[Any] = None
    state_npz: Optional[Any] = None
    img_npz: Optional[Any] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Episode length in recorded steps.

        Returns:
            Number of rows in robots_rows.
        """
        return len(self.robots_rows)

    @property
    def n_robots(self) -> int:
        """Number of robots represented in this episode.

        Returns:
            Robot count from the first payload row, or 0 if empty.
        """
        if not self.robots_rows:
            return 0
        return len(self.robots_rows[0].get("robots", []))

    @property
    def label(self) -> str:
        """Human-readable episode identifier.

        Returns:
            String formatted as run/env/episode.
        """
        return f"{self.run}/{self.env}/{self.episode}"


def load_episode(
    ep_dir: Path,
    run: str = "",
    env: str = "",
    episode: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    load_npz: bool = True,
) -> EpisodeData:
    """Load all available data for one episode directory.

    Args:
        ep_dir: Episode directory path.
        run: Run identifier.
        env: Environment identifier.
        episode: Episode identifier.
        metadata: Optional run-level metadata dictionary.
        load_npz: Whether to load NPZ channels in addition to JSONL files.

    Returns:
        Populated EpisodeData object.
    """
    ed = EpisodeData(
        run=run, env=env, episode=episode, path=ep_dir,
        robots_rows=_load_jsonl(ep_dir / "robots_payload.jsonl"),
        planner_rows=_load_jsonl(ep_dir / "planner_paths.jsonl"),
        item_rows=_load_jsonl(ep_dir / "item_poses.jsonl"),
        metadata=metadata or {},
    )
    if load_npz:
        ed.laser_npz = _load_npz(ep_dir / "laser_scans.npz")
        ed.laser_pts_npz = _load_npz(ep_dir / "laser_points.npz")
        ed.occ_npz = _load_npz(ep_dir / "occupancy_grids.npz")
        ed.cost_npz = _load_npz(ep_dir / "costmaps.npz")
        ed.nav_npz = _load_npz(ep_dir / "navmeshes.npz")
        ed.state_npz = _load_npz(ep_dir / "agent_states.npz")
        ed.img_npz = _load_npz(ep_dir / "images.npz")
    return ed


# =====================================================================
#  §3  PER-ROBOT TIMESERIES EXTRACTION
# =====================================================================

@dataclass
class RobotTimeseries:
    """Numpy arrays for every scalar tracked for one robot across an episode."""

    step: np.ndarray
    wall_time: np.ndarray

    # Pose
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray

    # Velocity
    vx: np.ndarray
    vy: np.ndarray
    speed: np.ndarray

    # Reward / outcome
    reward: np.ndarray
    cumulative_reward: np.ndarray
    success: np.ndarray
    collision: np.ndarray

    # Target
    target_x: np.ndarray
    target_y: np.ndarray
    distance_to_target: np.ndarray

    # Action (may contain NaNs when unavailable)
    action_v: np.ndarray
    action_omega: np.ndarray

    @property
    def n(self) -> int:
        """Number of samples in this robot timeseries.

        Returns:
            Length of the step array.
        """
        return len(self.step)


def extract_robot_timeseries(
    robots_rows: List[Dict[str, Any]],
    robot_idx: int = 0,
) -> RobotTimeseries:
    """Extract scalar time-series for one robot across all steps.

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.
        robot_idx: Robot index to extract.

    Returns:
        RobotTimeseries containing aligned per-step arrays.
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

    return RobotTimeseries(
        step=np.array(steps), wall_time=np.array(wall_times),
        x=np.array(xs), y=np.array(ys), yaw=np.array(yaws),
        vx=np.array(vxs), vy=np.array(vys), speed=np.array(speeds),
        reward=np.array(rewards),
        cumulative_reward=np.cumsum(rewards),
        success=np.array(successes), collision=np.array(collisions),
        target_x=np.array(target_xs), target_y=np.array(target_ys),
        distance_to_target=np.array(dists),
        action_v=np.array(action_vs), action_omega=np.array(action_omegas),
    )


# ── Legacy helpers kept for backward compatibility ───────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Public alias for internal JSONL loading helper.

    Args:
        path: Path to a JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    return _load_jsonl(path)


def count_robots(robots_rows: List[Dict[str, Any]]) -> int:
    """Count robots in an episode payload.

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.

    Returns:
        Number of robots in the first row, or 0 if no rows exist.
    """
    if not robots_rows:
        return 0
    return len(robots_rows[0].get("robots", []))


def get_planner_paths(
    planner_rows: List[Dict[str, Any]],
    step: int,
    robot_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """Return planner trajectories for one robot at one step.

    Args:
        planner_rows: Parsed planner rows.
        step: Step index to query.
        robot_idx: Robot index to query.

    Returns:
        Mapping with keys global_path, dwa_traj, and p_traj as (N, 2) arrays.
        Missing entries are returned as empty (0, 2) arrays.
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


def episode_summary(robots_rows: List[Dict[str, Any]], robot_idx: int = 0) -> Dict[str, Any]:
    """Compute episode summary statistics for one robot.

    Args:
        robots_rows: Parsed rows from robots_payload.jsonl.
        robot_idx: Robot index to summarize.

    Returns:
        Dictionary of summary statistics, or an empty dictionary when no
        samples are available.
    """
    ts = extract_robot_timeseries(robots_rows, robot_idx)
    if ts.n == 0:
        return {}
    return {
        "num_steps": ts.n,
        "duration_s": float(ts.wall_time[-1] - ts.wall_time[0]),
        "total_reward": float(ts.cumulative_reward[-1]),
        "mean_speed": float(np.nanmean(ts.speed)),
        "max_speed": float(np.nanmax(ts.speed)),
        "final_distance_to_target": float(ts.distance_to_target[-1]),
        "min_distance_to_target": float(np.min(ts.distance_to_target)),
        "any_collision": bool(np.any(ts.collision)),
        "any_success": bool(np.any(ts.success)),
        "path_length": float(np.sum(np.hypot(np.diff(ts.x), np.diff(ts.y)))),
    }


# =====================================================================
#  §4  SINGLE-AGENT NAVIGATION METRICS
# =====================================================================
#
# References:
#   - Anderson et al., "On Evaluation of Embodied Navigation Agents", 2018
#   - Francis et al., "Principles and Guidelines …", 2023
#   - Mavrogiannis et al., "Core Challenges …", 2023
# =====================================================================

@dataclass
class NavigationMetrics:
    """Metrics for a single robot in a single episode."""

    # ── outcome ──────────────────────────────────────────────────────
    success: bool                   # reached goal within threshold
    collision_occurred: bool        # any collision flag raised
    collision_count: int            # number of steps with collision=True
    collision_rate: float           # fraction of steps in collision

    # ── timing ───────────────────────────────────────────────────────
    num_steps: int
    episode_duration_s: float       # wall-clock seconds
    time_to_goal_s: float           # seconds to first success (inf if none)
    steps_to_goal: int              # steps to first success (-1 if none)

    # ── path quality ─────────────────────────────────────────────────
    path_length: float              # total Euclidean distance traversed (m)
    euclidean_distance: float       # straight-line start → goal (m)
    path_efficiency: float          # euclidean / path_length  (≤ 1)
    spl: float                      # Success weighted by Path Length

    # ── kinematics ───────────────────────────────────────────────────
    mean_speed: float
    max_speed: float
    mean_acceleration: float        # |Δv| / Δt
    max_acceleration: float
    mean_jerk: float                # |Δa| / Δt  (smoothness indicator)

    # ── smoothness ───────────────────────────────────────────────────
    path_curvature_mean: float      # mean unsigned curvature (1/m)
    path_curvature_max: float
    heading_change_total: float     # total absolute yaw change (rad)
    heading_change_rate: float      # rad / s

    # ── action quality ───────────────────────────────────────────────
    action_v_mean: float
    action_omega_mean: float
    action_v_std: float
    action_omega_std: float

    # ── reward ───────────────────────────────────────────────────────
    total_reward: float
    mean_reward: float
    final_distance_to_target: float
    min_distance_to_target: float

    # ── energy proxy ─────────────────────────────────────────────────
    energy_proxy: float             # ∫ v² dt  (proportional to energy)

    # ── safety (from laser) ──────────────────────────────────────────
    min_clearance: float            # closest obstacle reading ever (m)
    mean_clearance: float           # mean of per-step minimum range
    near_collision_fraction: float  # fraction of steps below threshold


def compute_navigation_metrics(
    ts: RobotTimeseries,
    dt: float = 0.04,
    success_dist: float = 0.3,
    near_collision_m: float = 0.15,
    laser_npz: Optional[Any] = None,
    robot_idx: int = 0,
) -> NavigationMetrics:
    """Compute single-robot navigation metrics.

    Args:
        ts: Robot time-series data.
        dt: Fallback simulation timestep in seconds.
        success_dist: Reserved success distance threshold parameter.
        near_collision_m: Clearance threshold for near-collision ratio.
        laser_npz: Optional laser NPZ archive for clearance metrics.
        robot_idx: Robot index used in NPZ key construction.

    Returns:
        NavigationMetrics populated for the provided timeseries.
    """
    n = ts.n
    if n < 2:
        return NavigationMetrics(
            success=False, collision_occurred=False, collision_count=0,
            collision_rate=0.0, num_steps=n, episode_duration_s=0.0,
            time_to_goal_s=float("inf"), steps_to_goal=-1,
            path_length=0.0, euclidean_distance=0.0, path_efficiency=0.0,
            spl=0.0, mean_speed=0.0, max_speed=0.0,
            mean_acceleration=0.0, max_acceleration=0.0, mean_jerk=0.0,
            path_curvature_mean=0.0, path_curvature_max=0.0,
            heading_change_total=0.0, heading_change_rate=0.0,
            action_v_mean=0.0, action_omega_mean=0.0,
            action_v_std=0.0, action_omega_std=0.0,
            total_reward=0.0, mean_reward=0.0,
            final_distance_to_target=0.0, min_distance_to_target=0.0,
            energy_proxy=0.0, min_clearance=float("inf"),
            mean_clearance=float("inf"), near_collision_fraction=0.0,
        )

    # ── timing ───────────────────────────────────────────────────────
    wt = ts.wall_time
    episode_dur = float(wt[-1] - wt[0]) if wt[-1] > wt[0] else n * dt

    first_success_idx = int(np.argmax(ts.success)) if np.any(ts.success) else -1
    is_success = bool(np.any(ts.success))
    if first_success_idx >= 0 and is_success:
        ttg = float(wt[first_success_idx] - wt[0]) if wt[first_success_idx] > wt[0] else first_success_idx * dt
        stg = int(ts.step[first_success_idx])
    else:
        ttg = float("inf")
        stg = -1

    # ── collisions ───────────────────────────────────────────────────
    coll_count = int(np.sum(ts.collision))
    coll_rate = coll_count / max(n, 1)

    # ── path ─────────────────────────────────────────────────────────
    dx = np.diff(ts.x)
    dy = np.diff(ts.y)
    seg_lengths = np.hypot(dx, dy)
    path_len = float(np.sum(seg_lengths))
    euclid = math.hypot(ts.x[-1] - ts.x[0], ts.y[-1] - ts.y[0])
    path_eff = euclid / max(path_len, 1e-9)

    # SPL — Success weighted by Path Length (Anderson et al. 2018)
    optimal_len = math.hypot(
        ts.target_x[0] - ts.x[0], ts.target_y[0] - ts.y[0]
    )
    spl = float(is_success) * optimal_len / max(path_len, optimal_len, 1e-9)

    # ── kinematics ───────────────────────────────────────────────────
    dt_arr = np.diff(wt)
    dt_arr[dt_arr <= 0] = dt

    speed_arr = ts.speed
    mean_spd = float(np.mean(speed_arr))
    max_spd = float(np.max(speed_arr))

    accel = np.abs(np.diff(speed_arr)) / dt_arr
    mean_acc = float(np.mean(accel)) if len(accel) > 0 else 0.0
    max_acc = float(np.max(accel)) if len(accel) > 0 else 0.0

    if len(accel) > 1:
        jerk = np.abs(np.diff(accel)) / dt_arr[1:]
        mean_jrk = float(np.mean(jerk))
    else:
        mean_jrk = 0.0

    # ── smoothness / curvature ───────────────────────────────────────
    dyaw = np.diff(ts.yaw)
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    heading_total = float(np.sum(np.abs(dyaw)))
    heading_rate = heading_total / max(episode_dur, 1e-9)

    curvature = np.abs(dyaw) / np.maximum(seg_lengths, 1e-6)
    curv_mean = float(np.mean(curvature))
    curv_max = float(np.max(curvature))

    # ── actions ──────────────────────────────────────────────────────
    v_valid = ts.action_v[~np.isnan(ts.action_v)]
    w_valid = ts.action_omega[~np.isnan(ts.action_omega)]
    av_mean = float(np.mean(v_valid)) if len(v_valid) else 0.0
    aw_mean = float(np.mean(w_valid)) if len(w_valid) else 0.0
    av_std = float(np.std(v_valid)) if len(v_valid) else 0.0
    aw_std = float(np.std(w_valid)) if len(w_valid) else 0.0

    # ── reward ───────────────────────────────────────────────────────
    total_r = float(ts.cumulative_reward[-1])
    mean_r = float(np.mean(ts.reward))
    final_dist = float(ts.distance_to_target[-1])
    min_dist = float(np.min(ts.distance_to_target))

    # ── energy proxy: ∫ v² dt ────────────────────────────────────────
    energy = float(np.sum(speed_arr[:-1] ** 2 * dt_arr))

    # ── laser-based clearance ────────────────────────────────────────
    min_clear = float("inf")
    clearances: List[float] = []
    near_coll_count = 0
    if laser_npz is not None:
        for step_val in ts.step:
            key = f"r{robot_idx}_ranges_{int(step_val):06d}"
            if key in laser_npz:
                ranges = laser_npz[key].astype(np.float32)
                valid = ranges[(ranges > 0.01) & np.isfinite(ranges)]
                if len(valid) > 0:
                    step_min = float(np.min(valid))
                    clearances.append(step_min)
                    if step_min < min_clear:
                        min_clear = step_min
                    if step_min < near_collision_m:
                        near_coll_count += 1

    mean_clear = float(np.mean(clearances)) if clearances else float("inf")
    near_frac = near_coll_count / max(n, 1)

    return NavigationMetrics(
        success=is_success,
        collision_occurred=coll_count > 0,
        collision_count=coll_count,
        collision_rate=coll_rate,
        num_steps=n,
        episode_duration_s=episode_dur,
        time_to_goal_s=ttg,
        steps_to_goal=stg,
        path_length=path_len,
        euclidean_distance=euclid,
        path_efficiency=path_eff,
        spl=spl,
        mean_speed=mean_spd,
        max_speed=max_spd,
        mean_acceleration=mean_acc,
        max_acceleration=max_acc,
        mean_jerk=mean_jrk,
        path_curvature_mean=curv_mean,
        path_curvature_max=curv_max,
        heading_change_total=heading_total,
        heading_change_rate=heading_rate,
        action_v_mean=av_mean,
        action_omega_mean=aw_mean,
        action_v_std=av_std,
        action_omega_std=aw_std,
        total_reward=total_r,
        mean_reward=mean_r,
        final_distance_to_target=final_dist,
        min_distance_to_target=min_dist,
        energy_proxy=energy,
        min_clearance=min_clear,
        mean_clearance=mean_clear,
        near_collision_fraction=near_frac,
    )


# =====================================================================
#  §5  MULTI-AGENT METRICS
# =====================================================================

@dataclass
class MultiAgentMetrics:
    """Metrics characterising the multi-robot system as a whole."""

    n_robots: int
    mean_inter_robot_distance: float
    min_inter_robot_distance: float
    inter_robot_collision_steps: int
    formation_coherence: float          # std of pairwise distances
    jain_fairness_reward: float         # Jain's fairness index on total rewards
    jain_fairness_path: float           # Jain's fairness index on path lengths
    mean_cooperative_reward: float
    std_cooperative_reward: float


def compute_multi_agent_metrics(
    episode: EpisodeData,
    dt: float = 0.04,
    robot_threshold: float = 0.5,
) -> Optional[MultiAgentMetrics]:
    """Compute multi-agent coordination metrics.

    Args:
        episode: Episode data bundle.
        dt: Timestep in seconds (reserved for extensions).
        robot_threshold: Distance threshold for inter-robot near-collision.

    Returns:
        MultiAgentMetrics when at least two robots are present; otherwise None.
    """
    n_robots = episode.n_robots
    if n_robots < 2:
        return None

    all_ts = [
        extract_robot_timeseries(episode.robots_rows, ridx)
        for ridx in range(n_robots)
    ]

    n_steps = min(ts.n for ts in all_ts)
    if n_steps == 0:
        return None

    pairwise_all: List[float] = []
    collision_step_count = 0

    for t in range(n_steps):
        positions = np.array([[all_ts[r].x[t], all_ts[r].y[t]] for r in range(n_robots)])
        step_min = float("inf")
        for i in range(n_robots):
            for j in range(i + 1, n_robots):
                d = float(np.linalg.norm(positions[i] - positions[j]))
                pairwise_all.append(d)
                if d < step_min:
                    step_min = d
        if step_min < robot_threshold:
            collision_step_count += 1

    pairwise_arr = np.array(pairwise_all)
    mean_inter = float(np.mean(pairwise_arr))
    min_inter = float(np.min(pairwise_arr))
    formation_coh = float(np.std(pairwise_arr))

    total_rewards = np.array([
        float(ts.cumulative_reward[-1]) if ts.n > 0 else 0.0
        for ts in all_ts
    ])
    path_lengths = np.array([
        float(np.sum(np.hypot(np.diff(ts.x), np.diff(ts.y))))
        if ts.n > 1 else 0.0
        for ts in all_ts
    ])

    def _jain(x: np.ndarray) -> float:
        """Compute Jain fairness for local arrays in this episode.

        Args:
            x: Per-robot scalar values.

        Returns:
            Jain fairness score in [1/n, 1].
        """
        x = np.abs(x) + 1e-12
        return float(np.sum(x) ** 2 / (len(x) * np.sum(x ** 2)))

    return MultiAgentMetrics(
        n_robots=n_robots,
        mean_inter_robot_distance=mean_inter,
        min_inter_robot_distance=min_inter,
        inter_robot_collision_steps=collision_step_count,
        formation_coherence=formation_coh,
        jain_fairness_reward=_jain(total_rewards),
        jain_fairness_path=_jain(path_lengths),
        mean_cooperative_reward=float(np.mean(total_rewards)),
        std_cooperative_reward=float(np.std(total_rewards)),
    )


# =====================================================================
#  §6  PLANNER ANALYSIS
# =====================================================================

@dataclass
class PlannerMetrics:
    """Metrics evaluating planner path quality for one robot."""

    mean_cross_track_error: float
    max_cross_track_error: float
    plan_vs_executed_ratio: float
    mean_dwa_horizon_length: float
    planning_coverage: float


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    """Compute minimum Euclidean distance from a point to a polyline.

    Args:
        point: Query point with shape (2,).
        polyline: Polyline vertices with shape (N, 2).

    Returns:
        Minimum distance from point to any line segment of the polyline.
    """
    if len(polyline) < 2:
        if len(polyline) == 1:
            return float(np.linalg.norm(point - polyline[0]))
        return float("inf")
    a = polyline[:-1]
    b = polyline[1:]
    ab = b - a
    ap = point - a
    t = np.sum(ap * ab, axis=1) / (np.sum(ab * ab, axis=1) + 1e-12)
    t = np.clip(t, 0, 1)
    proj = a + t[:, None] * ab
    dists = np.linalg.norm(point - proj, axis=1)
    return float(np.min(dists))


def compute_planner_metrics(
    episode: EpisodeData,
    robot_idx: int = 0,
) -> Optional[PlannerMetrics]:
    """Compute planner quality metrics for one robot.

    Args:
        episode: Episode data bundle.
        robot_idx: Robot index to evaluate.

    Returns:
        PlannerMetrics when sufficient planner/path data exists; otherwise None.
    """
    if not episode.planner_rows:
        return None

    ts = extract_robot_timeseries(episode.robots_rows, robot_idx)
    if ts.n < 2:
        return None

    plans_by_step: Dict[int, Dict[str, Any]] = {}
    for row in episode.planner_rows:
        if int(row.get("robot_idx", 0)) == robot_idx:
            plans_by_step[int(row.get("step", -1))] = row

    cross_track_errors: List[float] = []
    dwa_lengths: List[float] = []
    plan_lengths: List[float] = []
    steps_with_plan = 0

    for i, step_val in enumerate(ts.step):
        plan = plans_by_step.get(int(step_val))
        if plan is None:
            continue
        gp_raw = plan.get("global_path", [])
        if not gp_raw or len(gp_raw) < 2:
            continue

        gp = np.array(gp_raw, dtype=float)
        steps_with_plan += 1

        pos = np.array([ts.x[i], ts.y[i]])
        cte = _point_to_polyline_distance(pos, gp)
        cross_track_errors.append(cte)

        plan_len = float(np.sum(np.linalg.norm(np.diff(gp, axis=0), axis=1)))
        plan_lengths.append(plan_len)

        dwa_raw = plan.get("dwa_traj", [])
        if dwa_raw and len(dwa_raw) >= 2:
            dwa = np.array(dwa_raw, dtype=float)
            if dwa.ndim == 2 and dwa.shape[1] >= 2:
                dwa_len = float(np.sum(np.linalg.norm(
                    np.diff(dwa[:, :2], axis=0), axis=1)))
                dwa_lengths.append(dwa_len)

    if not cross_track_errors:
        return None

    exec_len = float(np.sum(np.hypot(np.diff(ts.x), np.diff(ts.y))))
    mean_plan_len = float(np.mean(plan_lengths)) if plan_lengths else exec_len

    return PlannerMetrics(
        mean_cross_track_error=float(np.mean(cross_track_errors)),
        max_cross_track_error=float(np.max(cross_track_errors)),
        plan_vs_executed_ratio=exec_len / max(mean_plan_len, 1e-9),
        mean_dwa_horizon_length=float(np.mean(dwa_lengths)) if dwa_lengths else 0.0,
        planning_coverage=steps_with_plan / max(ts.n, 1),
    )


# =====================================================================
#  §7  SENSOR & STATE ANALYSIS (NPZ-based)
# =====================================================================

@dataclass
class SensorMetrics:
    """Metrics derived from sensor data (laser, occupancy, agent state)."""

    laser_mean_range: float
    laser_std_range: float
    laser_coverage_fraction: float

    occupancy_mean_free: float
    occupancy_resolution: float

    costmap_mean_cost: float
    costmap_max_cost: float

    state_dim: int
    state_mean: Optional[np.ndarray]
    state_std: Optional[np.ndarray]


def compute_sensor_metrics(
    episode: EpisodeData,
    robot_idx: int = 0,
) -> Optional[SensorMetrics]:
    """Compute sensor/state metrics from NPZ channels.

    Args:
        episode: Episode data bundle.
        robot_idx: Robot index to evaluate.

    Returns:
        SensorMetrics when timeseries data exists; otherwise None.
    """
    ts = extract_robot_timeseries(episode.robots_rows, robot_idx)
    if ts.n == 0:
        return None

    # ── Laser ────────────────────────────────────────────────────────
    laser_means, laser_stds, coverages = [], [], []
    if episode.laser_npz is not None:
        for step_val in ts.step:
            key = f"r{robot_idx}_ranges_{int(step_val):06d}"
            mkey = f"r{robot_idx}_meta_{int(step_val):06d}"
            if key in episode.laser_npz:
                ranges = np.asarray(episode.laser_npz[key], dtype=np.float64)
                max_range = None
                if mkey in episode.laser_npz:
                    meta = np.asarray(episode.laser_npz[mkey], dtype=np.float64).ravel()
                    if meta.size >= 3 and np.isfinite(meta[2]) and meta[2] > 0:
                        max_range = float(meta[2])

                valid = ranges[(ranges > 0.01) & np.isfinite(ranges)]
                if max_range is not None:
                    valid = valid[valid <= max_range]

                if valid.size > 0:
                    laser_means.append(_safe_mean(valid))
                    laser_stds.append(_safe_std(valid))
                    coverages.append(float(valid.size) / max(len(ranges), 1))

    l_mean = _safe_mean(np.asarray(laser_means, dtype=np.float64)) if laser_means else float("nan")
    l_std = _safe_mean(np.asarray(laser_stds, dtype=np.float64)) if laser_stds else float("nan")
    l_cov = _safe_mean(np.asarray(coverages, dtype=np.float64)) if coverages else float("nan")

    # ── Occupancy ────────────────────────────────────────────────────
    occ_frees, occ_res = [], 0.0
    if episode.occ_npz is not None:
        for step_val in ts.step:
            gkey = f"r{robot_idx}_grid_{int(step_val):06d}"
            mkey = f"r{robot_idx}_meta_{int(step_val):06d}"
            if gkey in episode.occ_npz:
                grid = episode.occ_npz[gkey]
                free_frac = _safe_mean((grid == 0).astype(np.float64))
                occ_frees.append(free_frac)
                if mkey in episode.occ_npz:
                    meta = episode.occ_npz[mkey]
                    occ_res = float(meta[0])

    occ_free = _safe_mean(np.asarray(occ_frees, dtype=np.float64)) if occ_frees else float("nan")

    # ── Costmap ──────────────────────────────────────────────────────
    cost_means, cost_maxes = [], []
    if episode.cost_npz is not None:
        for step_val in ts.step:
            gkey = f"r{robot_idx}_grid_{int(step_val):06d}"
            if gkey in episode.cost_npz:
                grid = np.asarray(episode.cost_npz[gkey], dtype=np.float64)
                finite_grid = _finite_float64(grid)
                if finite_grid.size > 0:
                    cost_means.append(_safe_mean(finite_grid))
                    cost_maxes.append(float(np.max(finite_grid)))

    cm_mean = _safe_mean(np.asarray(cost_means, dtype=np.float64)) if cost_means else float("nan")
    cm_max = float(np.max(cost_maxes)) if cost_maxes else float("nan")

    # ── Agent state ──────────────────────────────────────────────────
    state_vecs: List[np.ndarray] = []
    if episode.state_npz is not None:
        for step_val in ts.step:
            key = f"r{robot_idx}_step_{int(step_val):06d}"
            if key in episode.state_npz:
                state_vecs.append(episode.state_npz[key].astype(np.float32))

    if state_vecs:
        mat = np.asarray(np.stack(state_vecs), dtype=np.float64)
        s_dim = mat.shape[1]
        s_mean = np.mean(mat, axis=0, dtype=np.float64)
        s_std = np.std(mat, axis=0, dtype=np.float64)
    else:
        s_dim = 0
        s_mean = None
        s_std = None

    return SensorMetrics(
        laser_mean_range=l_mean, laser_std_range=l_std,
        laser_coverage_fraction=l_cov,
        occupancy_mean_free=occ_free, occupancy_resolution=occ_res,
        costmap_mean_cost=cm_mean, costmap_max_cost=cm_max,
        state_dim=s_dim, state_mean=s_mean, state_std=s_std,
    )


# =====================================================================
#  §8  STATISTICAL UTILITIES
# =====================================================================

def _bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    statistic=np.mean,
) -> Tuple[float, float, float]:
    """Estimate bootstrap confidence interval for a statistic.

    Args:
        values: Sample values.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level in range (0, 1).
        statistic: Callable statistic applied to each resample.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).
    """
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(values) == 1:
        v = float(statistic(values))
        return (v, v, v)
    rng = np.random.default_rng(42)
    boot = np.array([
        statistic(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = float(np.percentile(boot, 100 * alpha))
    hi = float(np.percentile(boot, 100 * (1 - alpha)))
    return (float(statistic(values)), lo, hi)


def jain_fairness(x: np.ndarray) -> float:
    """Compute Jain's fairness index.

    Args:
        x: Non-negative utility values (negative values are abs-transformed).

    Returns:
        Jain fairness score in [1/n, 1], where 1 means perfectly fair.
    """
    x = np.abs(np.asarray(x, dtype=float)) + 1e-12
    return float(np.sum(x) ** 2 / (len(x) * np.sum(x ** 2)))


# =====================================================================
#  §9  AGGREGATE RESULT CONTAINERS
# =====================================================================

@dataclass
class EpisodeResult:
    """All computed results for a single episode."""
    data: EpisodeData
    robot_ts: List[RobotTimeseries]
    nav_metrics: List[NavigationMetrics]
    planner_metrics: List[Optional[PlannerMetrics]]
    sensor_metrics: List[Optional[SensorMetrics]]
    multi_agent: Optional[MultiAgentMetrics]


@dataclass
class AggregateResults:
    """Collection of all episode results with convenience accessors."""
    episodes: List[EpisodeResult]
    config: AnalysisConfig

    @property
    def runs(self) -> List[str]:
        """Sorted list of run identifiers in the result set.

        Returns:
            Sorted unique run names.
        """
        return sorted({er.data.run for er in self.episodes})

    @property
    def envs(self) -> List[str]:
        """Sorted list of environment identifiers in the result set.

        Returns:
            Sorted unique environment names.
        """
        return sorted({er.data.env for er in self.episodes})

    def filter(
        self,
        run: Optional[str] = None,
        env: Optional[str] = None,
        episode: Optional[str] = None,
    ) -> List[EpisodeResult]:
        """Filter episode results by run/env/episode identifiers.

        Args:
            run: Optional run name filter.
            env: Optional environment name filter.
            episode: Optional episode name filter.

        Returns:
            List of EpisodeResult entries matching the provided filters.
        """
        out = self.episodes
        if run:
            out = [er for er in out if er.data.run == run]
        if env:
            out = [er for er in out if er.data.env == env]
        if episode:
            out = [er for er in out if er.data.episode == episode]
        return out


# =====================================================================
#  §10  ANALYSIS PIPELINE
# =====================================================================

def analyse_episode(
    episode: EpisodeData,
    cfg: AnalysisConfig,
) -> EpisodeResult:
    """Run full metric computation for a single episode.

    Args:
        episode: Episode data bundle.
        cfg: Analysis configuration.

    Returns:
        EpisodeResult containing per-robot and multi-agent metrics.
    """
    n_robots = episode.n_robots

    robot_ts = [
        extract_robot_timeseries(episode.robots_rows, ridx)
        for ridx in range(n_robots)
    ]
    nav_metrics = [
        compute_navigation_metrics(
            robot_ts[ridx],
            dt=cfg.dt,
            success_dist=cfg.success_distance_threshold,
            near_collision_m=cfg.near_collision_threshold,
            laser_npz=episode.laser_npz if cfg.enable_sensor_analysis else None,
            robot_idx=ridx,
        )
        for ridx in range(n_robots)
    ]
    planner_metrics = [
        compute_planner_metrics(episode, robot_idx=ridx)
        for ridx in range(n_robots)
    ]
    sensor_metrics: List[Optional[SensorMetrics]] = []
    if cfg.enable_sensor_analysis:
        sensor_metrics = [
            compute_sensor_metrics(episode, robot_idx=ridx)
            for ridx in range(n_robots)
        ]
    else:
        sensor_metrics = [None] * n_robots

    multi = compute_multi_agent_metrics(episode, dt=cfg.dt)

    return EpisodeResult(
        data=episode, robot_ts=robot_ts, nav_metrics=nav_metrics,
        planner_metrics=planner_metrics, sensor_metrics=sensor_metrics,
        multi_agent=multi,
    )


def run_analysis(cfg: AnalysisConfig) -> AggregateResults:
    """Discover, load, and analyze all matching episodes.

    Args:
        cfg: Analysis configuration.

    Returns:
        AggregateResults containing analyzed episode results and config.
    """
    logger.info("Discovering traces in %s …", cfg.traces_root)
    items = discover_structure(cfg.traces_root, cfg.run_filter, cfg.env_filter)
    if not items:
        logger.warning("No episodes found.")
        return AggregateResults(episodes=[], config=cfg)
    logger.info("Found %d episode(s) across %d run(s).",
                len(items), len({i[0] for i in items}))

    meta_cache: Dict[str, Dict[str, Any]] = {}
    results: List[EpisodeResult] = []

    for run, env, ep, ep_dir in items:
        if run not in meta_cache:
            meta_cache[run] = _load_metadata(cfg.traces_root / run)
        logger.info("  Loading %s/%s/%s …", run, env, ep)
        episode = load_episode(
            ep_dir, run=run, env=env, episode=ep,
            metadata=meta_cache[run],
            load_npz=cfg.enable_sensor_analysis or cfg.enable_state_analysis,
        )
        result = analyse_episode(episode, cfg)
        results.append(result)

    return AggregateResults(episodes=results, config=cfg)


# =====================================================================
#  §11  VISUALISATION
# =====================================================================

def _save_fig(fig: plt.Figure, path: Path, cfg: AnalysisConfig):
    """Save and close a figure using configured format.

    Args:
        fig: Matplotlib figure to save.
        path: Output path without extension.
        cfg: Analysis configuration providing figure format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path = path.with_suffix(f".{cfg.fig_format}")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("    Saved %s", path.name)


# ── 11a  Trajectory map ──────────────────────────────────────────────

def plot_trajectory_map(
    results: List[EpisodeResult],
    title: str = "Robot Trajectories",
    cfg: AnalysisConfig = AnalysisConfig(),
    out: Optional[Path] = None,
):
    """Plot XY trajectories for all robots in selected episodes.

    Args:
        results: Episode results to visualize.
        title: Plot title.
        cfg: Analysis configuration for saving options.
        out: Optional output path stem for saving.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    cmap = plt.cm.tab10

    for eidx, er in enumerate(results):
        for ridx in range(er.data.n_robots):
            ts = er.robot_ts[ridx]
            if ts.n == 0:
                continue
            color = cmap(ridx % 10)
            label = f"{er.data.label} R{ridx}" if len(results) <= 5 else None
            ax.plot(ts.x, ts.y, color=color, alpha=0.6, linewidth=1, label=label)
            ax.scatter(ts.x[0], ts.y[0], marker="o", color=color, s=40, zorder=5)
            ax.scatter(ts.x[-1], ts.y[-1], marker="x", color=color, s=40, zorder=5)
            ax.scatter(ts.target_x[0], ts.target_y[0], marker="*",
                       color=color, s=100, zorder=5, edgecolors="k", linewidths=0.5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    if len(results) <= 5:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    if out:
        _save_fig(fig, out, cfg)
    return fig


# ── 11b  Box-plot comparison across runs ─────────────────────────────

def plot_metric_boxplot(
    agg: AggregateResults,
    metric_name: str = "total_reward",
    ylabel: str = "Total Reward",
    title: str = "Metric Distribution Across Runs",
    out: Optional[Path] = None,
):
    """Plot boxplots of a navigation metric grouped by run.

    Args:
        agg: Aggregate analysis results.
        metric_name: NavigationMetrics field name to plot.
        ylabel: Y-axis label.
        title: Plot title.
        out: Optional output path stem for saving.

    Returns:
        Matplotlib Figure object.
    """
    runs = agg.runs
    data_per_run: Dict[str, List[float]] = {r: [] for r in runs}

    for er in agg.episodes:
        for nm in er.nav_metrics:
            val = getattr(nm, metric_name, None)
            if val is not None and np.isfinite(val):
                data_per_run[er.data.run].append(val)

    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 0.8), 5))
    labels = list(data_per_run.keys())
    data = [data_per_run[r] for r in labels]
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    except TypeError:
        # Older Matplotlib releases use "labels" instead of "tick_labels".
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], plt.cm.Set2.colors):
        patch.set_facecolor(color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    if out:
        _save_fig(fig, out, agg.config)
    return fig


# ── 11c  Reward / distance learning curves ──────────────────────────

def plot_learning_curves(
    agg: AggregateResults,
    out: Optional[Path] = None,
):
    """Plot run-level mean reward and final-distance bars.

    Args:
        agg: Aggregate analysis results.
        out: Optional output path stem for saving.

    Returns:
        Matplotlib Figure object.
    """
    runs = agg.runs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    reward_per_run, dist_per_run = [], []
    for run in runs:
        eps = agg.filter(run=run)
        rewards = [nm.total_reward for er in eps for nm in er.nav_metrics]
        dists = [nm.final_distance_to_target for er in eps for nm in er.nav_metrics]
        reward_per_run.append(np.mean(rewards) if rewards else 0)
        dist_per_run.append(np.mean(dists) if dists else 0)

    x = np.arange(len(runs))
    ax1.bar(x, reward_per_run, color="steelblue", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(runs, rotation=45, fontsize=7)
    ax1.set_ylabel("Mean Total Reward")
    ax1.set_title("Reward Across Runs")

    ax2.bar(x, dist_per_run, color="coral", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(runs, rotation=45, fontsize=7)
    ax2.set_ylabel("Mean Final Distance to Target (m)")
    ax2.set_title("Final Distance Across Runs")

    fig.tight_layout()
    if out:
        _save_fig(fig, out, agg.config)
    return fig


# ── 11d  Radar chart (multi-metric) ─────────────────────────────────

def plot_radar_comparison(
    agg: AggregateResults,
    out: Optional[Path] = None,
):
    """Plot radar comparison of normalized metrics across runs.

    Args:
        agg: Aggregate analysis results.
        out: Optional output path stem for saving.

    Returns:
        Matplotlib Figure object, or None when fewer than two runs exist.
    """
    runs = agg.runs
    if len(runs) < 2:
        return None

    radar_fields = [
        ("spl", "SPL", True),
        ("path_efficiency", "Path Eff.", True),
        ("mean_speed", "Speed", True),
        ("total_reward", "Reward", True),
        ("collision_rate", "1-Coll.Rate", False),
        ("energy_proxy", "1/Energy", False),
    ]

    run_means: Dict[str, Dict[str, float]] = {}
    for run in runs:
        eps = agg.filter(run=run)
        vals: Dict[str, List[float]] = {f: [] for f, _, _ in radar_fields}
        for er in eps:
            for nm in er.nav_metrics:
                for f, _, _ in radar_fields:
                    v = getattr(nm, f, None)
                    if v is not None and np.isfinite(v):
                        vals[f].append(v)
        run_means[run] = {f: np.mean(vals[f]) if vals[f] else 0 for f, _, _ in radar_fields}

    all_vals = {f: [run_means[r][f] for r in runs] for f, _, _ in radar_fields}
    normed: Dict[str, Dict[str, float]] = {r: {} for r in runs}
    for f, _, higher_better in radar_fields:
        arr = np.array(all_vals[f])
        rng = arr.max() - arr.min()
        if rng < 1e-12:
            for r in runs:
                normed[r][f] = 0.5
        else:
            for r in runs:
                v = (run_means[r][f] - arr.min()) / rng
                normed[r][f] = v if higher_better else (1 - v)

    labels = [lab for _, lab, _ in radar_fields]
    n_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    cmap_r = plt.cm.Set1
    for i, run in enumerate(runs):
        values = [normed[run][f] for f, _, _ in radar_fields]
        values += values[:1]
        ax.plot(angles, values, "o-", label=run, color=cmap_r(i % 9), linewidth=1.5)
        ax.fill(angles, values, alpha=0.1, color=cmap_r(i % 9))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Multi-Metric Radar Comparison", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=7)
    fig.tight_layout()
    if out:
        _save_fig(fig, out, agg.config)
    return fig


# ── 11e  Multi-agent heatmap ─────────────────────────────────────────

def plot_inter_robot_distance_heatmap(
    er: EpisodeResult,
    out: Optional[Path] = None,
    cfg: AnalysisConfig = AnalysisConfig(),
):
    """Plot a heatmap of pairwise inter-robot distances over time.

    Args:
        er: Episode result to visualize.
        out: Optional output path stem for saving.
        cfg: Analysis configuration for save format.

    Returns:
        Matplotlib Figure object, or None for single-robot episodes.
    """
    n_robots = er.data.n_robots
    if n_robots < 2:
        return None

    n_steps = min(ts.n for ts in er.robot_ts)
    n_pairs = n_robots * (n_robots - 1) // 2
    dist_matrix = np.zeros((n_pairs, n_steps))

    pair_labels = []
    pair_idx = 0
    for i in range(n_robots):
        for j in range(i + 1, n_robots):
            pair_labels.append(f"R{i}-R{j}")
            for t in range(n_steps):
                d = math.hypot(
                    er.robot_ts[i].x[t] - er.robot_ts[j].x[t],
                    er.robot_ts[i].y[t] - er.robot_ts[j].y[t],
                )
                dist_matrix[pair_idx, t] = d
            pair_idx += 1

    fig, ax = plt.subplots(figsize=(12, max(3, n_pairs * 0.6)))
    im = ax.imshow(dist_matrix, aspect="auto", cmap="RdYlGn",
                   origin="lower", interpolation="nearest")
    ax.set_yticks(range(n_pairs))
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel("Step")
    ax.set_ylabel("Robot Pair")
    ax.set_title(f"Inter-Robot Distance — {er.data.label}")
    plt.colorbar(im, ax=ax, label="Distance (m)")
    fig.tight_layout()
    if out:
        _save_fig(fig, out, cfg)
    return fig


# ── 11f  Velocity / action profiles ─────────────────────────────────

def plot_kinematics_profile(
    er: EpisodeResult,
    robot_idx: int = 0,
    out: Optional[Path] = None,
    cfg: AnalysisConfig = AnalysisConfig(),
):
    """Plot speed/acceleration/action/heading-rate profiles.

    Args:
        er: Episode result to visualize.
        robot_idx: Robot index to visualize.
        out: Optional output path stem for saving.
        cfg: Analysis configuration providing timestep and format.

    Returns:
        Matplotlib Figure object, or None when too few samples exist.
    """
    ts = er.robot_ts[robot_idx]
    if ts.n < 3:
        return None

    dt_arr = np.diff(ts.wall_time)
    dt_arr[dt_arr <= 0] = cfg.dt
    accel = np.abs(np.diff(ts.speed)) / dt_arr
    dyaw = np.diff(ts.yaw)
    dyaw = (dyaw + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = np.abs(dyaw) / dt_arr

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    axes[0, 0].plot(ts.step, ts.speed, "b-", linewidth=0.8)
    axes[0, 0].set_ylabel("Speed (m/s)")
    axes[0, 0].set_title("Speed")

    axes[0, 1].plot(ts.step[1:], accel, "r-", linewidth=0.8)
    axes[0, 1].set_ylabel("Acceleration (m/s²)")
    axes[0, 1].set_title("Acceleration Magnitude")

    axes[1, 0].plot(ts.step, ts.action_v, "g-", linewidth=0.8, label="v_cmd")
    axes[1, 0].plot(ts.step, ts.action_omega, "m-", linewidth=0.8, label="w_cmd")
    axes[1, 0].set_ylabel("Action")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_title("Action Commands")
    axes[1, 0].legend()

    axes[1, 1].plot(ts.step[1:], np.degrees(yaw_rate), "k-", linewidth=0.8)
    axes[1, 1].set_ylabel("Heading Rate (deg/s)")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_title("Heading Change Rate")

    fig.suptitle(f"Kinematics — {er.data.label} Robot {robot_idx}", y=1.01)
    fig.tight_layout()
    if out:
        _save_fig(fig, out, cfg)
    return fig


# ── 11g  Planner snapshot ────────────────────────────────────────────

def plot_planner_snapshot(
    er: EpisodeResult,
    step: int,
    robot_idx: int = 0,
    out: Optional[Path] = None,
    cfg: AnalysisConfig = AnalysisConfig(),
):
    """Overlay planner trajectories on executed path at a given step.

    Args:
        er: Episode result to visualize.
        step: Step index for snapshot.
        robot_idx: Robot index to visualize.
        out: Optional output path stem for saving.
        cfg: Analysis configuration for save format.

    Returns:
        Matplotlib Figure object, or None when no robot samples exist.
    """
    ts = er.robot_ts[robot_idx]
    if ts.n == 0:
        return None

    plan = None
    for row in er.data.planner_rows:
        if int(row.get("step", -1)) == step and \
           int(row.get("robot_idx", 0)) == robot_idx:
            plan = row
            break

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ts.x, ts.y, "k-", alpha=0.15, label="Full path")
    mask = ts.step <= step
    ax.plot(ts.x[mask], ts.y[mask], "b-", linewidth=2, label="Path so far")

    idx = int(np.searchsorted(ts.step, step))
    if idx < ts.n:
        ax.scatter(ts.x[idx], ts.y[idx], c="blue", s=100, zorder=10, label="Pose")
        ax.scatter(ts.target_x[idx], ts.target_y[idx], c="gold", marker="*",
                   s=200, zorder=10, edgecolors="k", linewidths=0.5, label="Target")

    if plan:
        for key, style, lbl in [
            ("global_path", "g--", "Global plan"),
            ("dwa_traj", "r-", "DWA traj"),
            ("p_traj", "m:", "Pred. traj"),
        ]:
            pts = plan.get(key)
            if pts and len(pts) >= 2:
                arr = np.array(pts)
                ax.plot(arr[:, 0], arr[:, 1], style, linewidth=1.5, label=lbl)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Planner Snapshot — step {step}, R{robot_idx}")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if out:
        _save_fig(fig, out, cfg)
    return fig


# ── 11h  Success / collision rate bar chart ──────────────────────────

def plot_success_collision_rates(
    agg: AggregateResults,
    out: Optional[Path] = None,
):
    """Plot success and collision rates for each run.

    Args:
        agg: Aggregate analysis results.
        out: Optional output path stem for saving.

    Returns:
        Matplotlib Figure object.
    """
    runs = agg.runs
    success_rates, collision_rates = [], []
    for run in runs:
        eps = agg.filter(run=run)
        nm_list = [nm for er in eps for nm in er.nav_metrics]
        suc = np.mean([nm.success for nm in nm_list]) if nm_list else 0
        col = np.mean([nm.collision_occurred for nm in nm_list]) if nm_list else 0
        success_rates.append(suc)
        collision_rates.append(col)

    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 0.8), 5))
    ax.bar(x - width / 2, success_rates, width, label="Success Rate", color="seagreen")
    ax.bar(x + width / 2, collision_rates, width, label="Collision Rate", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(runs, rotation=45, fontsize=7)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Success & Collision Rates")
    ax.legend()
    fig.tight_layout()
    if out:
        _save_fig(fig, out, agg.config)
    return fig


# ── 11i  SPL distribution ───────────────────────────────────────────

def plot_spl_distribution(
    agg: AggregateResults,
    out: Optional[Path] = None,
):
    """Plot histogram of SPL values across robot-episodes.

    Args:
        agg: Aggregate analysis results.
        out: Optional output path stem for saving.

    Returns:
        Matplotlib Figure object, or None when no SPL values are available.
    """
    spl_vals = [nm.spl for er in agg.episodes for nm in er.nav_metrics
                if np.isfinite(nm.spl)]
    if not spl_vals:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(spl_vals, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(spl_vals), color="red", linestyle="--",
               label=f"Mean SPL = {np.mean(spl_vals):.3f}")
    ax.set_xlabel("SPL")
    ax.set_ylabel("Count")
    ax.set_title("SPL Distribution")
    ax.legend()
    fig.tight_layout()
    if out:
        _save_fig(fig, out, agg.config)
    return fig


# ── 11j  Sensor analysis plot ────────────────────────────────────────

def plot_laser_clearance_over_time(
    er: EpisodeResult,
    robot_idx: int = 0,
    out: Optional[Path] = None,
    cfg: AnalysisConfig = AnalysisConfig(),
):
    """Plot minimum laser clearance over time for one robot.

    Args:
        er: Episode result to visualize.
        robot_idx: Robot index to visualize.
        out: Optional output path stem for saving.
        cfg: Analysis configuration with near-collision threshold.

    Returns:
        Matplotlib Figure object, or None when laser data is unavailable.
    """
    ts = er.robot_ts[robot_idx]
    if ts.n == 0 or er.data.laser_npz is None:
        return None

    steps_list, clearances = [], []
    for i, step_val in enumerate(ts.step):
        key = f"r{robot_idx}_ranges_{int(step_val):06d}"
        if key in er.data.laser_npz:
            ranges = er.data.laser_npz[key].astype(np.float32)
            valid = ranges[(ranges > 0.01) & np.isfinite(ranges)]
            if len(valid) > 0:
                steps_list.append(int(step_val))
                clearances.append(float(np.min(valid)))

    if not clearances:
        return None

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps_list, clearances, "b-", linewidth=0.8)
    ax.axhline(cfg.near_collision_threshold, color="red", linestyle="--",
               alpha=0.7, label=f"Threshold = {cfg.near_collision_threshold}m")
    ax.fill_between(steps_list, 0, cfg.near_collision_threshold, alpha=0.1, color="red")
    ax.set_xlabel("Step")
    ax.set_ylabel("Min Clearance (m)")
    ax.set_title(f"Obstacle Clearance — {er.data.label} R{robot_idx}")
    ax.legend()
    fig.tight_layout()
    if out:
        _save_fig(fig, out, cfg)
    return fig


def _serialize_scalar(value: Any) -> Any:
    """Convert numpy-heavy values into CSV/JSON-friendly scalars.

    Args:
        value: Any metric value, potentially numpy scalar/array, list, tuple,
            or native Python scalar.

    Returns:
        JSON-safe scalar. Arrays/lists are JSON-stringified, numpy scalars are
        converted to Python scalars, and non-finite floats become None.
    """
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]]):
    """Write row dictionaries to a CSV file with stable header ordering.

    Args:
        path: Destination CSV path.
        rows: List of dictionaries representing rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write("")
        return

    fieldnames: List[str] = []
    seen: set = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_computed_results(
    agg: AggregateResults,
    out_dir: Path,
) -> List[Path]:
    """Export every computed metric object to machine-readable files.

    Args:
        agg: Aggregate analysis results.
        out_dir: Root output directory.

    Returns:
        List of written file paths.
    """
    export_dir = out_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    nav_rows: List[Dict[str, Any]] = []
    planner_rows: List[Dict[str, Any]] = []
    sensor_rows: List[Dict[str, Any]] = []
    multi_rows: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    for er in agg.episodes:
        base_episode = {
            "run": er.data.run,
            "env": er.data.env,
            "episode": er.data.episode,
            "label": er.data.label,
            "n_robots": er.data.n_robots,
            "n_steps": er.data.n_steps,
        }
        manifest_rows.append(base_episode.copy())

        if er.multi_agent is not None:
            row = base_episode.copy()
            for k, v in asdict(er.multi_agent).items():
                row[k] = _serialize_scalar(v)
            multi_rows.append(row)

        for ridx in range(er.data.n_robots):
            robot_base = {
                **base_episode,
                "robot_idx": ridx,
            }

            nm = er.nav_metrics[ridx]
            nav_row = robot_base.copy()
            for k, v in asdict(nm).items():
                nav_row[k] = _serialize_scalar(v)
            nav_rows.append(nav_row)

            pm = er.planner_metrics[ridx] if ridx < len(er.planner_metrics) else None
            if pm is not None:
                planner_row = robot_base.copy()
                for k, v in asdict(pm).items():
                    planner_row[k] = _serialize_scalar(v)
                planner_rows.append(planner_row)

            sm = er.sensor_metrics[ridx] if ridx < len(er.sensor_metrics) else None
            if sm is not None:
                sensor_row = robot_base.copy()
                for k, v in asdict(sm).items():
                    sensor_row[k] = _serialize_scalar(v)
                sensor_rows.append(sensor_row)

    nav_fields = [f.name for f in fields(NavigationMetrics)]
    nav_values = {
        name: [float(getattr(nm, name)) for er in agg.episodes for nm in er.nav_metrics
               if isinstance(getattr(nm, name), (int, float, np.integer, np.floating))
               and np.isfinite(float(getattr(nm, name)))]
        for name in nav_fields
    }
    summary = {
        "counts": {
            "runs": len(agg.runs),
            "envs": len(agg.envs),
            "episodes": len(agg.episodes),
            "robot_episodes": len(nav_rows),
            "multi_agent_episodes": len(multi_rows),
            "planner_metric_rows": len(planner_rows),
            "sensor_metric_rows": len(sensor_rows),
        },
        "global_navigation_means": {
            name: (_safe_mean(np.asarray(vals, dtype=np.float64)) if vals else None)
            for name, vals in nav_values.items()
        },
    }

    run_summary_rows: List[Dict[str, Any]] = []
    env_summary_rows: List[Dict[str, Any]] = []
    for run in agg.runs:
        run_eps = agg.filter(run=run)
        nm_list = [nm for er in run_eps for nm in er.nav_metrics]
        row = {
            "run": run,
            "episodes": len(run_eps),
            "robot_episodes": len(nm_list),
        }
        for name in nav_fields:
            vals = [float(getattr(nm, name)) for nm in nm_list
                    if isinstance(getattr(nm, name), (int, float, np.integer, np.floating))
                    and np.isfinite(float(getattr(nm, name)))]
            row[f"mean_{name}"] = _safe_mean(np.asarray(vals, dtype=np.float64)) if vals else None
        run_summary_rows.append(row)

    for env in agg.envs:
        env_eps = agg.filter(env=env)
        nm_list = [nm for er in env_eps for nm in er.nav_metrics]
        row = {
            "env": env,
            "episodes": len(env_eps),
            "robot_episodes": len(nm_list),
        }
        for name in nav_fields:
            vals = [float(getattr(nm, name)) for nm in nm_list
                    if isinstance(getattr(nm, name), (int, float, np.integer, np.floating))
                    and np.isfinite(float(getattr(nm, name)))]
            row[f"mean_{name}"] = _safe_mean(np.asarray(vals, dtype=np.float64)) if vals else None
        env_summary_rows.append(row)

    written: List[Path] = []
    csv_targets = [
        (export_dir / "episodes_manifest.csv", manifest_rows),
        (export_dir / "navigation_metrics.csv", nav_rows),
        (export_dir / "planner_metrics.csv", planner_rows),
        (export_dir / "sensor_metrics.csv", sensor_rows),
        (export_dir / "multi_agent_metrics.csv", multi_rows),
        (export_dir / "run_summary.csv", run_summary_rows),
        (export_dir / "env_summary.csv", env_summary_rows),
    ]
    for path, rows in csv_targets:
        _write_csv_rows(path, rows)
        written.append(path)

    summary_path = export_dir / "global_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    written.append(summary_path)

    return written


# =====================================================================
#  §12  REPORT GENERATION (Markdown)
# =====================================================================

def generate_report(
    agg: AggregateResults,
    out_dir: Path,
) -> str:
    """Generate a Markdown analysis report with figure links.

    Args:
        agg: Aggregate analysis results.
        out_dir: Output directory containing figures.

    Returns:
        Full Markdown report text that is also written to report.md.
    """
    lines: List[str] = []

    def _add(text: str = ""):
        """Append one Markdown line to the accumulating report buffer.

        Args:
            text: Line text to append.
        """
        lines.append(text)

    _add("# Data Trace Analysis Report")
    _add()
    _add(f"**Traces root:** `{agg.config.traces_root}`  ")
    _add(f"**Runs:** {len(agg.runs)}  |  **Environments:** {len(agg.envs)}  "
         f"|  **Episodes:** {len(agg.episodes)}")
    _add()

    # ── Global summary table ─────────────────────────────────────────
    _add("## 1. Global Summary")
    _add()
    all_nm = [nm for er in agg.episodes for nm in er.nav_metrics]
    if all_nm:
        def _stat(fld):
            """Compute mean and bootstrap CI strings for a metric field.

            Args:
                fld: NavigationMetrics attribute name.

            Returns:
                Tuple of preformatted strings (mean, ci_low, ci_high).
            """
            vals = [getattr(nm, fld, 0) for nm in all_nm
                    if np.isfinite(getattr(nm, fld, float("nan")))]
            if not vals:
                return "-", "-", "-"
            m, lo, hi = _bootstrap_ci(np.array(vals),
                                      agg.config.n_bootstrap,
                                      agg.config.confidence_level)
            return f"{m:.4f}", f"{lo:.4f}", f"{hi:.4f}"

        metrics_table = [
            ("Success Rate", "success"),
            ("Collision Rate", "collision_rate"),
            ("SPL", "spl"),
            ("Path Efficiency", "path_efficiency"),
            ("Total Reward", "total_reward"),
            ("Path Length (m)", "path_length"),
            ("Mean Speed (m/s)", "mean_speed"),
            ("Energy Proxy", "energy_proxy"),
            ("Time to Goal (s)", "time_to_goal_s"),
            ("Final Dist. to Target (m)", "final_distance_to_target"),
            ("Min Clearance (m)", "min_clearance"),
            ("Mean Acceleration (m/s2)", "mean_acceleration"),
            ("Mean Jerk", "mean_jerk"),
            ("Heading Change Rate (rad/s)", "heading_change_rate"),
        ]

        ci_pct = int(agg.config.confidence_level * 100)
        _add(f"| Metric | Mean | {ci_pct}% CI Low | {ci_pct}% CI High |")
        _add("|--------|------|----------|---------|")
        for label, fld in metrics_table:
            m, lo, hi = _stat(fld)
            _add(f"| {label} | {m} | {lo} | {hi} |")
        _add()

    # ── Per-run breakdown ────────────────────────────────────────────
    _add("## 2. Per-Run Breakdown")
    _add()
    _add("| Run | Episodes | Robots | Success% | Collision% | Mean Reward | Mean SPL | Mean Path (m) |")
    _add("|-----|----------|--------|----------|------------|-------------|----------|---------------|")
    for run in agg.runs:
        eps = agg.filter(run=run)
        nm_list = [nm for er in eps for nm in er.nav_metrics]
        n_eps = len(eps)
        n_rob = eps[0].data.n_robots if eps else 0
        suc = f"{np.mean([nm.success for nm in nm_list]) * 100:.1f}" if nm_list else "-"
        col = f"{np.mean([nm.collision_occurred for nm in nm_list]) * 100:.1f}" if nm_list else "-"
        rew = f"{np.mean([nm.total_reward for nm in nm_list]):.2f}" if nm_list else "-"
        s = f"{np.mean([nm.spl for nm in nm_list]):.3f}" if nm_list else "-"
        pl = f"{np.mean([nm.path_length for nm in nm_list]):.2f}" if nm_list else "-"
        _add(f"| {run} | {n_eps} | {n_rob} | {suc} | {col} | {rew} | {s} | {pl} |")
    _add()

    # ── Multi-agent section ──────────────────────────────────────────
    multi_results = [er for er in agg.episodes if er.multi_agent is not None]
    if multi_results:
        _add("## 3. Multi-Agent Metrics")
        _add()
        _add("| Episode | Robots | Mean Inter-Dist (m) | Min Inter-Dist (m) | "
             "Near-Collisions | Jain Fairness (Reward) |")
        _add("|---------|--------|---------------------|--------------------"
             "|-----------------|------------------------|")
        for er in multi_results:
            ma = er.multi_agent
            _add(f"| {er.data.label} | {ma.n_robots} | "
                 f"{ma.mean_inter_robot_distance:.3f} | "
                 f"{ma.min_inter_robot_distance:.3f} | "
                 f"{ma.inter_robot_collision_steps} | "
                 f"{ma.jain_fairness_reward:.4f} |")
        _add()

    # ── Planner section ──────────────────────────────────────────────
    planner_results = [(er, pm) for er in agg.episodes
                       for pm in er.planner_metrics if pm is not None]
    if planner_results:
        _add("## 4. Planner Quality")
        _add()
        _add("| Episode | Mean CTE (m) | Max CTE (m) | Plan/Exec Ratio | DWA Horizon (m) | Coverage |")
        _add("|---------|-------------|-------------|-----------------|-----------------|----------|")
        for er, pm in planner_results:
            _add(f"| {er.data.label} | {pm.mean_cross_track_error:.3f} | "
                 f"{pm.max_cross_track_error:.3f} | {pm.plan_vs_executed_ratio:.3f} | "
                 f"{pm.mean_dwa_horizon_length:.3f} | {pm.planning_coverage:.2%} |")
        _add()

    # ── Figures ──────────────────────────────────────────────────────
    _add("## 5. Figures")
    _add()
    ext = agg.config.fig_format
    figures_root = out_dir / "figures"
    figure_files = sorted(figures_root.rglob(f"*.{ext}")) if figures_root.exists() else []
    for fname in figure_files:
        rel = fname.relative_to(out_dir).as_posix()
        _add(f"### {fname.stem.replace('_', ' ').title()}")
        _add(f"*Path:* `{rel}`")
        _add(f"![{fname.stem}]({rel})")
        _add()

    exports_dir = out_dir / "exports"
    if exports_dir.exists():
        _add("## 6. Exported Tables")
        _add()
        for fname in sorted(exports_dir.glob("*")):
            if fname.is_file():
                rel = f"exports/{fname.name}"
                _add(f"- [{fname.name}]({rel})")
        _add()

    report_text = "\n".join(lines)
    report_path = out_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Report saved to %s", report_path)
    return report_text


# =====================================================================
#  §13  MAIN ENTRY POINT
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    p = argparse.ArgumentParser(
        description="Comprehensive analysis of recorded robot navigation traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--traces", type=Path, default=Path("traces"),
                   help="Root directory containing run folders.")
    p.add_argument("--output", type=Path, default=Path("trace_analysis_output"),
                   help="Directory for figures and report.")
    p.add_argument("--runs", nargs="*", default=None,
                   help="Filter to specific run names.")
    p.add_argument("--envs", nargs="*", default=None,
                   help="Filter to specific env names.")
    p.add_argument("--dt", type=float, default=0.04,
                   help="Simulation timestep (fallback when wall_time unavailable).")
    p.add_argument("--no-sensor", action="store_true",
                   help="Skip sensor-level analysis (faster).")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--no-report", action="store_true",
                   help="Skip Markdown report generation.")
    p.add_argument("--fig-format", choices=["png", "pdf", "svg"], default="png",
                   help="Output figure format.")
    p.add_argument("--confidence", type=float, default=0.95,
                   help="Confidence level for bootstrap CIs.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable debug logging.")
    return p


def main(argv: Optional[Sequence[str]] = None):
    """CLI entrypoint for running trace analysis and artifact generation.

    Args:
        argv: Optional argument vector. Uses sys.argv when None.
    """
    args = build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = AnalysisConfig(
        traces_root=args.traces.resolve(),
        output_dir=args.output.resolve(),
        run_filter=args.runs,
        env_filter=args.envs,
        dt=args.dt,
        enable_sensor_analysis=not args.no_sensor,
        enable_state_analysis=not args.no_sensor,
        enable_plots=not args.no_plots,
        enable_report=not args.no_report,
        fig_format=args.fig_format,
        confidence_level=args.confidence,
    )

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  VTPRL Data Trace Analysis")
    print("=" * 70)

    # ── Run analysis ─────────────────────────────────────────────────
    agg = run_analysis(cfg)
    if not agg.episodes:
        print("\n  No episodes found. Check --traces path.\n")
        return

    n_total = len(agg.episodes)
    n_robots_total = sum(er.data.n_robots for er in agg.episodes)
    print(f"\n  Analysed {n_total} episodes, {n_robots_total} robot-episodes "
          f"across {len(agg.runs)} run(s).\n")

    # ── Export all computed metric artifacts ────────────────────────
    exported_files = export_computed_results(agg, cfg.output_dir)
    print(f"  Exported {len(exported_files)} result file(s) to {cfg.output_dir / 'exports'}/\n")

    # ── Console summary ──────────────────────────────────────────────
    all_nm = [nm for er in agg.episodes for nm in er.nav_metrics]

    print("  +-- Global Navigation Metrics ----------------------------+")
    for label, fld, fmt in [
        ("Success Rate",        "success",                  ".1%"),
        ("Collision Rate",      "collision_rate",           ".1%"),
        ("SPL",                 "spl",                      ".4f"),
        ("Path Efficiency",     "path_efficiency",          ".4f"),
        ("Mean Total Reward",   "total_reward",             "+.2f"),
        ("Mean Path Length",    "path_length",              ".2f"),
        ("Mean Speed",          "mean_speed",               ".3f"),
        ("Mean Jerk",           "mean_jerk",                ".4f"),
        ("Mean Energy Proxy",   "energy_proxy",             ".2f"),
        ("Mean Final Dist.",    "final_distance_to_target", ".3f"),
        ("Mean Min Clearance",  "min_clearance",            ".3f"),
    ]:
        vals = [getattr(nm, fld) for nm in all_nm if np.isfinite(getattr(nm, fld))]
        if vals:
            m = np.mean(vals)
            print(f"  |  {label:25s} = {m:{fmt}}")
    print("  +--------------------------------------------------------+\n")

    # ── Per-run console table ────────────────────────────────────────
    header = f"  {'Run':<12} {'Eps':>4}  {'Suc%':>5}  {'Col%':>5}  {'SPL':>7}  {'Reward':>8}  {'Path(m)':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for run in agg.runs:
        eps = agg.filter(run=run)
        nms = [nm for er in eps for nm in er.nav_metrics]
        n_e = len(eps)
        suc = np.mean([nm.success for nm in nms]) * 100 if nms else 0
        col = np.mean([nm.collision_rate for nm in nms]) * 100 if nms else 0
        spl_v = np.mean([nm.spl for nm in nms]) if nms else 0
        rew = np.mean([nm.total_reward for nm in nms]) if nms else 0
        pl = np.mean([nm.path_length for nm in nms]) if nms else 0
        print(f"  {run:<12} {n_e:>4}  {suc:>5.1f}  {col:>5.1f}  {spl_v:>7.4f}  {rew:>+8.2f}  {pl:>8.2f}")
    print()

    # ── Multi-agent summary ──────────────────────────────────────────
    multi_eps = [er for er in agg.episodes if er.multi_agent is not None]
    if multi_eps:
        print("  +-- Multi-Agent Metrics ---------------------------------+")
        ma_list = [er.multi_agent for er in multi_eps]
        print(f"  |  Mean inter-robot dist.  = {np.mean([m.mean_inter_robot_distance for m in ma_list]):.3f} m")
        print(f"  |  Min  inter-robot dist.  = {np.min([m.min_inter_robot_distance for m in ma_list]):.3f} m")
        print(f"  |  Mean Jain fairness(rew) = {np.mean([m.jain_fairness_reward for m in ma_list]):.4f}")
        print(f"  |  Near-collisions (steps) = {np.sum([m.inter_robot_collision_steps for m in ma_list])}")
        print("  +--------------------------------------------------------+\n")

    # ── Generate plots ───────────────────────────────────────────────
    if cfg.enable_plots:
        print("  Generating figures ...")
        od = cfg.output_dir / "figures"
        general_dir = od / "general"
        detailed_dir = od / "detailed"

        # General result folders
        g_overview = general_dir / "overview"
        g_boxplots = general_dir / "boxplots"
        g_comparison = general_dir / "comparison"

        # Detailed result folders
        d_trajectory = detailed_dir / "trajectory"
        d_kinematics = detailed_dir / "kinematics"
        d_planner = detailed_dir / "planner_snapshot"
        d_sensor = detailed_dir / "laser_clearance"
        d_multi = detailed_dir / "inter_robot_heatmap"

        if agg.episodes:
            plot_trajectory_map(agg.episodes, "Trajectories - All Episodes",
                                cfg, g_overview / "trajectories_all")

        for er in agg.episodes:
            ep_tag = f"{er.data.run}_{er.data.env}_{er.data.episode}"
            plot_trajectory_map([er], f"Trajectory - {er.data.label}",
                                cfg, d_trajectory / f"trajectory_{ep_tag}")

        for metric, ylabel in [
            ("total_reward", "Total Reward"),
            ("spl", "SPL"),
            ("path_length", "Path Length (m)"),
            ("path_efficiency", "Path Efficiency"),
            ("energy_proxy", "Energy Proxy"),
        ]:
            plot_metric_boxplot(agg, metric, ylabel,
                                f"{ylabel} Across Runs", g_boxplots / f"boxplot_{metric}")

        plot_learning_curves(agg, g_comparison / "learning_curves")
        plot_success_collision_rates(agg, g_comparison / "success_collision_rates")
        plot_spl_distribution(agg, g_comparison / "spl_distribution")

        if len(agg.runs) >= 2:
            plot_radar_comparison(agg, g_comparison / "radar_comparison")

        for er in agg.episodes:
            ep_tag = f"{er.data.run}_{er.data.env}_{er.data.episode}"

            for ridx, ts in enumerate(er.robot_ts):
                if ts.n >= 3:
                    plot_kinematics_profile(
                        er,
                        robot_idx=ridx,
                        out=d_kinematics / f"kinematics_{ep_tag}_r{ridx}",
                        cfg=cfg,
                    )

                if er.data.planner_rows and ts.n > 0:
                    mid = int(ts.step[ts.n // 2])
                    plot_planner_snapshot(
                        er,
                        step=mid,
                        robot_idx=ridx,
                        out=d_planner / f"planner_snapshot_{ep_tag}_r{ridx}",
                        cfg=cfg,
                    )

                if cfg.enable_sensor_analysis and er.data.laser_npz is not None:
                    plot_laser_clearance_over_time(
                        er,
                        robot_idx=ridx,
                        out=d_sensor / f"laser_clearance_{ep_tag}_r{ridx}",
                        cfg=cfg,
                    )

            if er.multi_agent is not None:
                plot_inter_robot_distance_heatmap(
                    er,
                    out=d_multi / f"inter_robot_heatmap_{ep_tag}",
                    cfg=cfg,
                )

        print(f"  Figures saved to {cfg.output_dir / 'figures'}/\n")

    # ── Generate report ──────────────────────────────────────────────
    if cfg.enable_report:
        generate_report(agg, cfg.output_dir)
        print(f"  Report saved to {cfg.output_dir / 'report.md'}\n")

    print("=" * 70)
    print("  Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
