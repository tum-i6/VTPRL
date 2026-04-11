"""Offline data-trace player for simulator environments.

This module reads recordings produced by :class:`~utils.data_trace_recorder.DataTraceRecorder`
and reconstructs :class:`~utils.telemetry.MonitorPayload` objects step by step.
The payloads can then be pushed to either the Qt or Web task-monitor backend
for visualisation — **without** running the Unity simulator or the gym
environment.

Usage
-----
.. code-block:: python

    from utils.data_trace_player import DataTracePlayer

    player = DataTracePlayer("./my_traces", monitor_backend="web")
    for env_id, episode_idx in player.episodes():
        player.play_episode(env_id, episode_idx, speed=1.0)
    player.close()

Architecture
------------
The player is built on the same :class:`~utils.task_monitor_proxy.TaskMonitorController`
(or its web variant) that the live simulator uses.  It instantiates the
monitor subprocess, registers a synthetic environment spec, then iterates
over the stored steps and feeds each reconstructed ``MonitorPayload``
dict through the existing ``update_environment`` IPC path.

Playback speed is realised by sleeping between successive steps using the
``wall_time`` field recorded in ``robots_payload.jsonl``.  A *speed*
multiplier of ``2.0`` replays at twice the original wall-clock rate;
``0`` (or ``None``) replays as fast as the monitor can consume.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


from .data_trace_schema import RecordingConfig
from .telemetry import (
    CostmapData,
    LaserPoints,
    LaserScanData,
    MonitorPayload,
    NavmeshData,
    OccupancyGridData,
    PlannerData,
    PlannerPaths,
    Pose2D,
)

logger = logging.getLogger(__name__)


# =====================================================================
#  Trace index helpers
# =====================================================================


def _load_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a ``.npz`` archive as a dict of arrays.

    ``np.savez`` appends ``.npz`` automatically when the suffix is missing,
    so only the canonical path is checked.

    Args:
        path: Path to the ``.npz`` file.

    Returns:
        Dictionary mapping array names to numpy arrays, or ``None`` when the
        file does not exist.
    """
    if path.exists():
        return dict(np.load(str(path), allow_pickle=False))
    return None


def _unjsonify(obj: object) -> object:
    """Reverse the ``_jsonify`` encoding used by the recorder.

    Converts ``{"__bytes_b64__": "..."}`` sentinels back to ``bytes``
    and recurses into dicts and lists.

    Args:
        obj: Parsed JSON structure possibly containing encoded byte blobs.

    Returns:
        Equivalent structure with byte blobs restored.
    """
    if isinstance(obj, dict):
        b64 = obj.get("__bytes_b64__")
        if b64 is not None and len(obj) == 1 and isinstance(b64, str):
            return base64.b64decode(b64)
        return {k: _unjsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unjsonify(v) for v in obj]
    return obj


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON-lines file.

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        List of parsed dictionaries (one per line), or empty list if absent.
    """
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(_unjsonify(json.loads(line)))
    return rows


# =====================================================================
#  Step reconstruction
# =====================================================================

def _reconstruct_payload(
    step: int,
    robots_row: Dict[str, Any],
    laser_npz: Optional[Dict[str, np.ndarray]],
    laser_points_npz: Optional[Dict[str, np.ndarray]],
    occ_npz: Optional[Dict[str, np.ndarray]],
    costmap_npz: Optional[Dict[str, np.ndarray]],
    navmesh_npz: Optional[Dict[str, np.ndarray]],
    agent_states_npz: Optional[Dict[str, np.ndarray]],
    images_npz: Optional[Dict[str, np.ndarray]],
    planner_by_step: Dict[Tuple[int, int], Dict[str, Any]],
) -> MonitorPayload:
    """Reassemble a :class:`MonitorPayload` from stored artefacts.

    All dense arrays use per-robot prefixed keys (``r{idx}_``).  Robot-0
    scalar fields (pose, velocity, reward, …) are extracted from the
    first entry in the ``robots_row["robots"]`` list.

    Args:
        step: Zero-based step index.
        robots_row: One row from ``robots_payload.jsonl``.
        laser_npz: Loaded laser-scan ``.npz`` archive (or ``None``).
        laser_points_npz: Loaded laser-points ``.npz`` archive (or ``None``).
        occ_npz: Loaded occupancy-grid ``.npz`` archive (or ``None``).
        costmap_npz: Loaded costmap ``.npz`` archive (or ``None``).
        navmesh_npz: Loaded navmesh ``.npz`` archive (or ``None``).
        agent_states_npz: Loaded agent-state ``.npz`` archive (or ``None``).
        images_npz: Loaded images ``.npz`` archive (or ``None``).
        planner_by_step: Planner-path rows indexed by ``(step, robot_idx)`` tuple.

    Returns:
        Fully populated :class:`MonitorPayload`.
    """
    robots = robots_row.get("robots", [])
    r0 = robots[0] if robots else {}

    # ── scalars from robot 0 ───────────────────────────────────────
    r0_pos = r0.get("robot_position", [0.0, 0.0])
    pose = Pose2D(
        x=float(r0_pos[0]) if len(r0_pos) > 0 else 0.0,
        y=float(r0_pos[1]) if len(r0_pos) > 1 else 0.0,
        yaw=float(r0.get("robot_yaw", 0.0)),
    )
    r0_vel = r0.get("robot_velocity", [0.0, 0.0])
    velocity = (
        float(r0_vel[0]) if len(r0_vel) > 0 else 0.0,
        float(r0_vel[1]) if len(r0_vel) > 1 else 0.0,
    )
    r0_delta = r0.get("target_delta", [0.0, 0.0, 0.0])
    delta = (
        float(r0_delta[0]) if len(r0_delta) > 0 else 0.0,
        float(r0_delta[1]) if len(r0_delta) > 1 else 0.0,
        float(r0_delta[2]) if len(r0_delta) > 2 else 0.0,
    )
    reward = float(r0.get("reward", 0.0))
    success = bool(r0.get("success", False))
    collision = bool(r0.get("collision", False))

    r0_target = r0.get("target_position")
    target_pos = None
    if r0_target is not None and len(r0_target) >= 2:
        target_pos = (float(r0_target[0]), float(r0_target[1]))

    r0_action = r0.get("agent_action")
    agent_action = None
    if r0_action is not None and len(r0_action) >= 2:
        agent_action = np.array([float(r0_action[0]), float(r0_action[1])], dtype=np.float32)

    agent_reward = np.array([reward], dtype=np.float32)

    # ── helper: extract per-robot NPZ array ──────────────────────────
    def _get_npz(npz: Optional[Dict[str, np.ndarray]], key: str) -> Optional[np.ndarray]:
        """Return array from *npz* under *key*, or ``None``."""
        return npz.get(key) if npz is not None else None

    # ── laser scan (top-level = robot 0) ─────────────────────────────
    laser_scan = None
    if laser_npz is not None:
        ranges = _get_npz(laser_npz, f"r0_ranges_{step:06d}")
        if ranges is not None:
            meta = laser_npz.get(f"r0_meta_{step:06d}", np.zeros(5, dtype=np.float32))
            laser_scan = LaserScanData(
                ranges=ranges,
                angle_min=float(meta[0]),
                angle_max=float(meta[1]),
                max_range=float(meta[2]),
                sensor_offset_xy=(float(meta[3]), float(meta[4])),
            )

    # ── laser points (top-level = robot 0) ───────────────────────────
    laser_points = None
    if laser_points_npz is not None:
        pts = _get_npz(laser_points_npz, f"r0_step_{step:06d}")
        if pts is not None:
            laser_points = LaserPoints(
                points_xy=pts,
                origin_xy=(pose.x, pose.y),
            )

    # ── occupancy grid (top-level = robot 0) ─────────────────────────
    occupancy = None
    if occ_npz is not None:
        grid = _get_npz(occ_npz, f"r0_grid_{step:06d}")
        if grid is not None:
            meta_arr = occ_npz.get(f"r0_meta_{step:06d}", np.zeros(3, dtype=np.float32))
            occupancy = OccupancyGridData(
                grid=grid,
                resolution=float(meta_arr[0]),
                origin_xy=(float(meta_arr[1]), float(meta_arr[2])),
            )

    # ── costmap (top-level = robot 0) ────────────────────────────────
    costmap = None
    if costmap_npz is not None:
        grid = _get_npz(costmap_npz, f"r0_grid_{step:06d}")
        if grid is not None:
            cm_meta = costmap_npz.get(f"r0_meta_{step:06d}", np.zeros(3, dtype=np.float32))
            costmap = CostmapData(
                grid=grid,
                resolution=float(cm_meta[0]),
                origin_xy=(float(cm_meta[1]), float(cm_meta[2])),
            )

    # ── navmesh (top-level = robot 0) ────────────────────────────────
    navmesh = None
    if navmesh_npz is not None:
        verts = _get_npz(navmesh_npz, f"r0_verts_{step:06d}")
        if verts is not None:
            navmesh = NavmeshData(
                vertices=verts,
                triangles=navmesh_npz.get(f"r0_tris_{step:06d}", np.zeros((0, 3), dtype=np.int32)),
            )

    # ── agent state (top-level = robot 0) ────────────────────────────
    agent_state = _get_npz(agent_states_npz, f"r0_step_{step:06d}")

    # ── planner paths (top-level = robot 0) ──────────────────────────
    planner = None
    planner_row = planner_by_step.get((step, 0))
    has_spatial = occupancy is not None or costmap is not None or laser_points is not None or navmesh is not None
    if planner_row is not None or has_spatial:
        if planner_row is not None:
            paths = PlannerPaths(
                global_path=[tuple(pt) for pt in planner_row.get("global_path", [])],
                dwa_traj=[tuple(pt) for pt in planner_row.get("dwa_traj", [])],
                p_traj=[tuple(pt) for pt in planner_row.get("p_traj", [])],
            )
        else:
            paths = PlannerPaths(global_path=[], dwa_traj=[], p_traj=[])
        planner = PlannerData(
            paths=paths,
            occupancy=occupancy,
            costmap=costmap,
            laser_points=laser_points,
            robot_pose=pose,
            target_pos=target_pos,
        )

    # ── images (per-robot keys: r{idx}_{cam}_{step:06d}) ────────────
    agent_image = None
    if images_npz is not None:
        suffix = f"_{step:06d}"
        per_robot_imgs: Dict[int, Dict[str, bytes]] = {}
        for key in sorted(images_npz):
            if not key.endswith(suffix):
                continue
            prefix = key[: -len(suffix)]  # e.g. "r0_Overhead_1"
            parts = prefix.split("_", 1)
            if len(parts) != 2 or not parts[0].startswith("r") or not parts[0][1:].isdigit():
                continue
            ridx = int(parts[0][1:])
            cam_name = parts[1]
            arr = images_npz[key]
            frame = arr.tobytes() if isinstance(arr, np.ndarray) else arr
            per_robot_imgs.setdefault(ridx, {})[cam_name] = frame

        # Top-level agent_image from robot 0
        r0_imgs = per_robot_imgs.get(0, {})
        if r0_imgs:
            agent_image = {
                "frames": list(r0_imgs.values()),
                "labels": list(r0_imgs.keys()),
            }

        # Inject per-robot images back into robot dicts
        for ridx, cam_dict in per_robot_imgs.items():
            if ridx < len(robots):
                robots[ridx]["agent_image"] = {
                    "frames": list(cam_dict.values()),
                    "labels": list(cam_dict.keys()),
                }

    # ── per-robot spatial injection ──────────────────────────────────
    # Restore spatial data into each robot dict so that _merge_robot_data
    # in the task monitor correctly overlays per-robot data when the user
    # switches between robots.
    for ridx in range(len(robots)):
        rdict = robots[ridx]
        if not isinstance(rdict, dict):
            continue
        rpos = rdict.get('robot_position', [0.0, 0.0])

        # laser scan
        if laser_npz is not None:
            rk = f"r{ridx}_ranges_{step:06d}"
            if rk in laser_npz:
                rmeta = laser_npz.get(f"r{ridx}_meta_{step:06d}", np.zeros(5, dtype=np.float32))
                rdict['laser_scan'] = laser_npz[rk]
                rdict['laser_angle_min'] = float(rmeta[0])
                rdict['laser_angle_max'] = float(rmeta[1])
                rdict['laser_max_range'] = float(rmeta[2])
                rdict['laser_sensor_offset'] = [float(rmeta[3]), float(rmeta[4])]

        # laser points
        if laser_points_npz is not None:
            rk = f"r{ridx}_step_{step:06d}"
            if rk in laser_points_npz:
                pts_arr = laser_points_npz[rk]
                rdict['laser_points'] = pts_arr.tolist() if isinstance(pts_arr, np.ndarray) else pts_arr

        # occupancy
        if occ_npz is not None:
            rk = f"r{ridx}_grid_{step:06d}"
            if rk in occ_npz:
                ometa = occ_npz.get(f"r{ridx}_meta_{step:06d}", np.zeros(3, dtype=np.float32))
                rdict['occupancy'] = {
                    'grid': occ_npz[rk],
                    'resolution': float(ometa[0]),
                    'origin': [float(ometa[1]), float(ometa[2])],
                }

        # costmap
        if costmap_npz is not None:
            rk = f"r{ridx}_grid_{step:06d}"
            if rk in costmap_npz:
                cmeta = costmap_npz.get(f"r{ridx}_meta_{step:06d}", np.zeros(3, dtype=np.float32))
                rdict['costmap'] = {
                    'costmap': costmap_npz[rk],
                    'resolution': float(cmeta[0]),
                    'origin': [float(cmeta[1]), float(cmeta[2])],
                }

        # navmesh
        if navmesh_npz is not None:
            vk = f"r{ridx}_verts_{step:06d}"
            if vk in navmesh_npz:
                rdict['navmesh'] = {
                    'vertices': navmesh_npz[vk],
                    'indices': navmesh_npz.get(f"r{ridx}_tris_{step:06d}", np.zeros((0,), dtype=np.int32)),
                }

        # agent state
        if agent_states_npz is not None:
            rk = f"r{ridx}_step_{step:06d}"
            if rk in agent_states_npz:
                rdict['agent_state'] = agent_states_npz[rk]

        # Synthesize agent_reward (numpy array) from scalar reward for panel compatibility
        reward_val = rdict.get('reward')
        if reward_val is not None and 'agent_reward' not in rdict:
            rdict['agent_reward'] = np.array([float(reward_val)], dtype=np.float32)

        # planner map (composite dict expected by _merge_robot_data)
        rpr = planner_by_step.get((step, ridx))
        r_occ = rdict.get('occupancy')
        r_cm = rdict.get('costmap')
        r_nm = rdict.get('navmesh')
        r_lp = rdict.get('laser_points')
        if rpr is not None or r_occ or r_cm or r_nm or r_lp:
            pm: Dict[str, Any] = {
                'robot_position': rpos,
                'robot_yaw': rdict.get('robot_yaw', 0.0),
                'target_position': rdict.get('target_position'),
            }
            if rpr is not None:
                pm['global_path'] = rpr.get('global_path', [])
                pm['dwa_traj'] = rpr.get('dwa_traj', [])
                pm['p_traj'] = rpr.get('p_traj', [])
            else:
                pm['global_path'] = []
                pm['dwa_traj'] = []
                pm['p_traj'] = []
            if r_occ:
                pm['occupancy'] = r_occ
            if r_cm:
                pm['costmap'] = r_cm
            if r_nm:
                pm['navmesh'] = r_nm
            if r_lp:
                pm['laser_points'] = r_lp
            rdict['planner_map'] = pm

    return MonitorPayload(
        robot_pose=pose,
        robot_velocity=velocity,
        target_delta=delta,
        reward=reward,
        success=success,
        collision=collision,
        laser_scan=laser_scan,
        laser_points=laser_points,
        occupancy=occupancy,
        costmap=costmap,
        navmesh=navmesh,
        planner=planner,
        target_pos=target_pos,
        agent_state=agent_state,
        agent_action=agent_action,
        agent_reward=agent_reward,
        agent_image=agent_image,
        robots=robots,
    )


# =====================================================================
#  Public player class
# =====================================================================

class DataTracePlayer:
    """Offline replayer that streams recorded traces to the task monitor.

    The player loads a trace directory produced by
    :class:`~utils.data_trace_recorder.DataTraceRecorder`, reconstructs each
    step's ``MonitorPayload``, and pushes it through the standard task-monitor
    IPC — letting you visualise historical runs without a live simulator.

    Args:
        trace_root: Path to the trace output directory (containing
            ``metadata.json``).
        monitor_backend: ``"qt"`` for the desktop Qt monitor or ``"web"``
            for the browser-based dashboard.  Defaults to ``"web"``.
        playback_env_id: Virtual environment id to register with the monitor.
            Defaults to ``0``.
    """

    def __init__(
        self,
        trace_root: str,
        monitor_backend: str = "web",
        playback_env_id: int = 0,
    ) -> None:
        self._root = Path(trace_root)
        self._backend = monitor_backend
        self._playback_env_id = playback_env_id
        self._monitor: Any = None
        self._registered = False

        # Load run metadata
        meta_path = self._root / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                self._metadata: Dict[str, Any] = json.load(fh)
        else:
            self._metadata = {}

        self._config = RecordingConfig.from_dict(self._metadata.get("config", {}))

    # ── discovery ────────────────────────────────────────────────────

    def episodes(self) -> List[Tuple[int, int]]:
        """List all ``(env_id, episode_index)`` pairs present in the trace.

        Returns:
            Sorted list of ``(env_id, episode_index)`` tuples.
        """
        result: List[Tuple[int, int]] = []
        for env_dir in sorted(self._root.glob("env_*")):
            if not env_dir.is_dir():
                continue
            env_id = int(env_dir.name.split("_")[1])
            for ep_dir in sorted(env_dir.glob("episode_*")):
                if not ep_dir.is_dir():
                    continue
                ep_idx = int(ep_dir.name.split("_")[1])
                result.append((env_id, ep_idx))
        return result

    def episode_dir(self, env_id: int, episode_index: int) -> Path:
        """Resolve the filesystem path for a specific episode.

        Args:
            env_id: Environment identifier.
            episode_index: Zero-based episode index.

        Returns:
            :class:`pathlib.Path` to the episode directory.
        """
        return self._root / f"env_{env_id:03d}" / f"episode_{episode_index:04d}"

    def episode_step_count(self, env_id: int, episode_index: int) -> int:
        """Return the number of recorded steps in an episode.

        Args:
            env_id: Environment identifier.
            episode_index: Zero-based episode index.

        Returns:
            Number of rows (steps) found in robots_payload.jsonl.
        """
        ep_dir = self.episode_dir(env_id, episode_index)
        return len(_load_jsonl(ep_dir / "robots_payload.jsonl"))

    # ── loading ──────────────────────────────────────────────────────

    def load_episode(
        self,
        env_id: int,
        episode_index: int,
    ) -> List[MonitorPayload]:
        """Load an entire episode and return the list of payloads.

        This is useful for programmatic analysis — every step is
        reconstructed into a :class:`MonitorPayload` in memory.

        Args:
            env_id: Environment identifier.
            episode_index: Episode index within that environment.

        Returns:
            Ordered list of :class:`MonitorPayload`, one per recorded step.
        """
        ep_dir = self.episode_dir(env_id, episode_index)

        robots_rows = _load_jsonl(ep_dir / "robots_payload.jsonl")
        if not robots_rows:
            logger.warning("No robots payload data in %s", ep_dir)
            return []

        # Load all bulk archives once (they are memory-mapped by numpy)
        laser_npz = _load_npz(ep_dir / "laser_scans.npz")
        laser_pts_npz = _load_npz(ep_dir / "laser_points.npz")
        occ_npz = _load_npz(ep_dir / "occupancy_grids.npz")
        cost_npz = _load_npz(ep_dir / "costmaps.npz")
        nav_npz = _load_npz(ep_dir / "navmeshes.npz")
        state_npz = _load_npz(ep_dir / "agent_states.npz")
        img_npz = _load_npz(ep_dir / "images.npz")

        planner_rows = _load_jsonl(ep_dir / "planner_paths.jsonl")
        planner_by_step: Dict[Tuple[int, int], Dict[str, Any]] = {
            (r["step"], r.get("robot_idx", 0)): r for r in planner_rows
        }

        payloads: List[MonitorPayload] = []
        for row in robots_rows:
            step = int(row.get("step", len(payloads)))
            payloads.append(
                _reconstruct_payload(
                    step, row,
                    laser_npz, laser_pts_npz, occ_npz, cost_npz,
                    nav_npz, state_npz, img_npz,
                    planner_by_step,
                )
            )
        return payloads

    # ── playback to task monitor ─────────────────────────────────────

    def play_episode(
        self,
        env_id: int,
        episode_index: int,
        speed: Optional[float] = 1.0,
        step_callback: Optional[Any] = None,
    ) -> int:
        """Replay an episode through the task monitor for visualisation.

        Args:
            env_id: Source environment id in the trace.
            episode_index: Episode index to replay.
            speed: Playback speed multiplier relative to original wall-clock
                timing.  ``1.0`` = real-time, ``2.0`` = double speed,
                ``None`` or ``0`` = as fast as possible.
            step_callback: Optional callable invoked as
                ``step_callback(step_index, payload_dict)`` after each step
                is sent to the monitor.  Useful for external progress bars.

        Returns:
            Total number of steps played.
        """
        self._ensure_monitor()

        ep_dir = self.episode_dir(env_id, episode_index)

        robots_rows = _load_jsonl(ep_dir / "robots_payload.jsonl")
        if not robots_rows:
            logger.warning("Nothing to play in %s", ep_dir)
            return 0

        # Pre-load all bulk archives
        laser_npz = _load_npz(ep_dir / "laser_scans.npz")
        laser_pts_npz = _load_npz(ep_dir / "laser_points.npz")
        occ_npz = _load_npz(ep_dir / "occupancy_grids.npz")
        cost_npz = _load_npz(ep_dir / "costmaps.npz")
        nav_npz = _load_npz(ep_dir / "navmeshes.npz")
        state_npz = _load_npz(ep_dir / "agent_states.npz")
        img_npz = _load_npz(ep_dir / "images.npz")

        planner_rows = _load_jsonl(ep_dir / "planner_paths.jsonl")
        planner_by_step: Dict[Tuple[int, int], Dict[str, Any]] = {
            (r["step"], r.get("robot_idx", 0)): r for r in planner_rows
        }

        prev_wall: Optional[float] = None
        count = 0
        carry_forward: Dict[str, Any] = {}

        for row in robots_rows:
            step = int(row.get("step", count))
            payload = _reconstruct_payload(
                step, row,
                laser_npz, laser_pts_npz, occ_npz, cost_npz,
                nav_npz, state_npz, img_npz,
                planner_by_step,
            )

            # Carry forward slowly-changing data so panels remain stable
            # across steps where the env didn't emit a fresh value.
            for field in ('occupancy', 'costmap', 'navmesh',
                          'laser_scan', 'laser_points', 'robots',
                          'agent_image'):
                current = getattr(payload, field)
                if current is not None:
                    carry_forward[field] = current
                elif field in carry_forward:
                    setattr(payload, field, carry_forward[field])

            # Keep planner spatial overlays in sync with carried-forward data
            if payload.planner is not None:
                for pf in ('occupancy', 'costmap', 'laser_points'):
                    if getattr(payload.planner, pf) is None and pf in carry_forward:
                        setattr(payload.planner, pf, carry_forward[pf])
            elif any(f in carry_forward for f in ('occupancy', 'costmap',
                                                   'laser_points', 'navmesh')):
                payload.planner = PlannerData(
                    paths=PlannerPaths(),
                    occupancy=carry_forward.get('occupancy'),
                    costmap=carry_forward.get('costmap'),
                    laser_points=carry_forward.get('laser_points'),
                    robot_pose=payload.robot_pose,
                    target_pos=payload.target_pos,
                )

            # Timing: honour original inter-step intervals
            cur_wall = float(row.get("wall_time", 0.0))
            if prev_wall is not None and speed and speed > 0 and cur_wall > prev_wall:
                dt = (cur_wall - prev_wall) / speed
                time.sleep(dt)
            prev_wall = cur_wall

            # Push to monitor — always use the single playback env id
            data_dict = payload.to_dict()
            self._enrich_data(data_dict, env_id)
            self._monitor.update_environment(self._playback_env_id, data_dict)

            if step_callback is not None:
                step_callback(step, data_dict)

            count += 1

        logger.info("Played %d steps for env %d episode %d", count, env_id, episode_index)
        return count

    def play_all(self, speed: Optional[float] = 1.0) -> int:
        """Replay every episode in the trace sequentially.

        Args:
            speed: Playback speed multiplier (see :meth:`play_episode`).

        Returns:
            Total number of steps played across all episodes.
        """
        total = 0
        for env_id, ep_idx in self.episodes():
            total += self.play_episode(env_id, ep_idx, speed=speed)
        return total

    # ── monitor lifecycle ────────────────────────────────────────────

    def _enrich_data(self, data: Dict[str, Any], env_id: int = 0) -> None:
        """Synthesise missing panel fields from available scalars in-place.

        Mirrors the web monitor's ``_enrich_warehouse_data`` so that the
        Qt backend (which lacks that enrichment) also gets ``agent_state``
        and ``agent_reward`` when the recorded payload omitted them.
        """
        saved_specs = self._metadata.get("specs", {})
        spec_info = saved_specs.get(str(env_id), {})
        config = spec_info.get("config", {}) if spec_info else {}
        if not config and saved_specs:
            first_info = next(iter(saved_specs.values()))
            config = first_info.get("config", {})
        if not config:
            config = {}

        if data.get("agent_state") is None:
            p = config.get("param_agent_state")
            if p:
                dim = int(p.get("dim", 0))
                pos = data.get("robot_position")
                yaw = data.get("robot_yaw")
                vel = data.get("robot_velocity")
                delta = data.get("target_delta")
                parts: List[float] = []
                if pos is not None:
                    parts.extend(
                        float(v)
                        for v in (pos if hasattr(pos, "__iter__") else [pos])
                    )
                if yaw is not None:
                    parts.append(float(yaw))
                if vel is not None:
                    parts.extend(
                        float(v)
                        for v in (vel if hasattr(vel, "__iter__") else [vel])
                    )
                if delta is not None:
                    parts.extend(
                        float(v)
                        for v in (delta if hasattr(delta, "__iter__") else [delta])
                    )
                if parts:
                    arr = np.array(parts, dtype=float)
                    if arr.size < dim:
                        arr = np.pad(arr, (0, dim - arr.size))
                    data["agent_state"] = arr[:dim]

        if data.get("agent_reward") is None and data.get("reward") is not None:
            data["agent_reward"] = np.array([float(data["reward"])], dtype=float)

    def _ensure_monitor(self) -> None:
        """Lazily start the task-monitor subprocess and register a single
        playback environment.

        All episodes are played sequentially through one environment slot
        (``self._playback_env_id``), so the environment selector in the
        monitor simply shows *Data Trace Replay* and is not switchable.
        """
        if self._monitor is not None:
            return

        if self._backend == "qt":
            from .task_monitor_proxy import TaskMonitorController
            self._monitor = TaskMonitorController()
        else:
            from .task_monitor_web_proxy import TaskMonitorWebController
            self._monitor = TaskMonitorWebController()

        from .task_monitor import MonitorSpec

        saved_specs = self._metadata.get("specs", {})

        # Pick config from the first available saved spec
        spec_config: Dict[str, Any] = {}
        spec_type = "warehouse"
        if saved_specs:
            first_info = next(iter(saved_specs.values()))
            spec_config = dict(first_info.get("config", {}))
            spec_type = first_info.get("type", "warehouse")
        if not spec_config:
            spec_config = {"observation": {"laser_count": 0}}

        spec_config["replay"] = True

        spec = MonitorSpec(
            env_id=self._playback_env_id,
            name="Data Trace Replay",
            type=spec_type,
            config=spec_config,
        )
        self._monitor.register_environment(spec)

        self._registered = True
        logger.info(
            "Task monitor (%s) started for replay — env %d.",
            self._backend, self._playback_env_id,
        )

    def close(self) -> None:
        """Shut down the task-monitor subprocess.

        Safe to call multiple times.
        """
        if self._monitor is not None:
            if self._registered:
                try:
                    self._monitor.remove_environment(self._playback_env_id)
                except Exception:
                    pass
            try:
                self._monitor.close()
            except Exception:
                pass
            self._monitor = None
            self._registered = False

    # ── context-manager support ──────────────────────────────────────

    def __enter__(self) -> DataTracePlayer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# =====================================================================
#  Standalone CLI entry-point
# =====================================================================

def main() -> None:
    """Command-line interface for replaying a data trace.

    Usage::

        python -m utils.data_trace_player <trace_dir> [--speed 1.0] [--backend web]
    """
    import argparse

    parser = argparse.ArgumentParser(description="Replay a data trace recording.")
    parser.add_argument("trace_dir", help="Path to the trace root directory.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--backend", choices=["qt", "web"], default="web",
        help="Task monitor backend (default: web).",
    )
    parser.add_argument("--env", type=int, default=None, help="Only replay this env_id.")
    parser.add_argument("--episode", type=int, default=None, help="Only replay this episode index.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    with DataTracePlayer(args.trace_dir, monitor_backend=args.backend) as player:
        episodes = player.episodes()
        if not episodes:
            logger.warning("No episodes found in '%s'.", args.trace_dir)
            return
        logger.info(
            "Found %d episode(s) in '%s'. Backend: %s, speed: %s",
            len(episodes), args.trace_dir, args.backend, args.speed,
        )
        if args.env is not None and args.episode is not None:
            player.play_episode(args.env, args.episode, speed=args.speed)
        elif args.env is not None:
            # Play all episodes for one env
            for env_id, ep_idx in player.episodes():
                if env_id == args.env:
                    player.play_episode(env_id, ep_idx, speed=args.speed)
        else:
            player.play_all(speed=args.speed)


if __name__ == "__main__":
    main()
