"""High-performance data-trace recorder for simulator environments.

The recorder sits between the environment ``step`` / ``reset`` boundary and
the optional task-monitor IPC layer.  It intercepts the same
:class:`~utils.telemetry.MonitorPayload` that the monitor consumes and
writes every enabled channel to disk:

* **Dense arrays** (laser scans, grids, images, navmeshes, agent states) →
  NumPy ``.npz`` archives with per-robot key prefixes (``r{idx}_``)
  in a streaming append pattern (one key per step).
* **Per-robot scalar telemetry & planner paths** → JSON-lines files.

Usage
-----
.. code-block:: python

    from utils.data_trace_schema import RecordingConfig, Channel
    from utils.data_trace_recorder import DataTraceRecorder

    cfg = RecordingConfig(
        enabled_channels={Channel.AGENT_STATE, Channel.LASER_SCAN, Channel.IMAGES},
        trace_root="./my_traces",
    )
    recorder = DataTraceRecorder(cfg)

    # Inside training / evaluation loop
    recorder.begin_episode(env_id=0)
    for step in range(max_steps):
        obs, reward, done, info = env.step(action)
        payload = env.get_monitor_payload()
        recorder.record_step(env_id=0, payload=payload, step_index=step)
        if done:
            break
    recorder.end_episode(env_id=0)

    # When finished
    recorder.close()
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


from .data_trace_schema import (
    ALL_CHANNELS,
    Channel,
    RecordingConfig,
)
from .telemetry import MonitorPayload

logger = logging.getLogger(__name__)

# Heavy keys stripped from per-robot dicts before JSONL serialisation.
# These modalities are either saved in dedicated NPZ/JSONL files already
# or are too large for JSON encoding (binary arrays, image bytes, grids).
_ROBOTS_STRIP_KEYS = frozenset({
    'agent_image',
    'agent_state',
    'laser_scan',
    'laser_points',
    'occupancy',
    'costmap',
    'navmesh',
    'planner_paths',
    'planner_map',
    'item_positions',
    'item_orientations',
    'item_yaws',
    'obstacle_positions',
    'obstacle_orientations',
    'obstacle_is_dynamic',
})


# =====================================================================
#  Bulk (numpy) array helpers
# =====================================================================

class _NpzStreamWriter:
    """Accumulates named numpy arrays and writes a single ``.npz`` on close.

    Keeps data in memory (lists of arrays keyed by step) and flushes to a
    single compressed or uncompressed ``.npz`` archive in one shot.  For
    very long episodes the :meth:`flush` call can be used to dump an
    intermediate split-file.

    Args:
        path: Target ``.npz`` file path.
        compressed: Use ``np.savez_compressed`` when ``True``.
    """

    def __init__(self, path: Path, compressed: bool = True) -> None:
        self._path = path
        self._compressed = compressed
        self._arrays: Dict[str, np.ndarray] = {}

    def append(self, key: str, arr: np.ndarray) -> None:
        """Store one array under *key*.

        Args:
            key: Unique identifier (typically ``"step_NNNNN"`` or a named sub-key).
            arr: Numpy array to record.
        """
        self._arrays[key] = arr

    def flush(self, final: bool = False) -> None:
        """Write all accumulated arrays to the ``.npz`` file.

        When *final* is ``True`` the in-memory buffer is cleared (used
        during episode finalisation).  Otherwise the buffer is kept so
        that the next periodic checkpoint writes a superset of the data
        — safe incremental checkpointing without read-merge overhead.

        Args:
            final: If ``True`` clear the buffer after writing.
        """
        if not self._arrays:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        save_fn = np.savez_compressed if self._compressed else np.savez
        save_fn(str(self._path), **self._arrays)
        if final:
            self._arrays.clear()

    @property
    def pending(self) -> int:
        """Number of arrays currently buffered.

        Returns:
            Count of arrays awaiting flush.
        """
        return len(self._arrays)


# =====================================================================
#  Per-episode writer
# =====================================================================

class _EpisodeWriter:
    """Manages all open file handles for a single episode of one environment.

    Args:
        episode_dir: Filesystem directory for this episode.
        config: Recording session configuration.
    """

    def __init__(self, episode_dir: Path, config: RecordingConfig) -> None:
        self._dir = episode_dir
        self._cfg = config

        # Bulk array writers — created lazily on first write
        self._laser_writer: Optional[_NpzStreamWriter] = None
        self._laser_points_writer: Optional[_NpzStreamWriter] = None
        self._occupancy_writer: Optional[_NpzStreamWriter] = None
        self._costmap_writer: Optional[_NpzStreamWriter] = None
        self._navmesh_writer: Optional[_NpzStreamWriter] = None
        self._image_writer: Optional[_NpzStreamWriter] = None
        self._agent_state_writer: Optional[_NpzStreamWriter] = None
        self._planner_rows: List[Dict[str, object]] = []
        self._item_rows: List[Dict[str, object]] = []
        self._obstacle_rows: List[Dict[str, object]] = []
        self._robots_rows: List[Dict[str, object]] = []

        self._step_count = 0

    # ── lazy constructors ────────────────────────────────────────────

    def _npz(self, name: str) -> _NpzStreamWriter:
        """Create or retrieve a :class:`_NpzStreamWriter` for *name*.

        Args:
            name: Base filename (without extension).

        Returns:
            Writer instance for the given array stream.
        """
        attr = f"_{name}_writer"
        writer = getattr(self, attr, None)
        if writer is None:
            writer = _NpzStreamWriter(self._dir / f"{name}.npz", self._cfg.compress_arrays)
            setattr(self, attr, writer)
        return writer

    # ── recording ────────────────────────────────────────────────────

    def record(self, step: int, payload: MonitorPayload) -> None:
        """Ingest one step of telemetry from *payload*.

        Only channels enabled in the session config are processed; disabled
        channels are silently skipped with no allocation overhead.

        Args:
            step: Zero-based step index within the current episode.
            payload: Telemetry bundle from the environment.
        """
        wall = time.monotonic()
        cfg = self._cfg

        # ── per-robot spatial data ────────────────────────────────────
        # All spatial channels are stored per-robot with an ``r{idx}_``
        # key prefix in their respective NPZ archives so the player can
        # reconstruct full per-robot dicts for the task-monitor merge.
        if payload.robots is not None:
            for ridx, robot in enumerate(payload.robots):
                if not isinstance(robot, dict):
                    continue

                # agent state
                if cfg.is_enabled(Channel.AGENT_STATE):
                    astate = robot.get('agent_state')
                    if astate is not None:
                        self._npz("agent_states").append(
                            f"r{ridx}_step_{step:06d}",
                            np.asarray(astate, dtype=np.float32),
                        )

                # laser scan
                if cfg.is_enabled(Channel.LASER_SCAN):
                    ls = robot.get('laser_scan')
                    if ls is not None:
                        offset = robot.get('laser_sensor_offset', [0.0, 0.0])
                        meta = np.array([
                            robot.get('laser_angle_min', 0.0),
                            robot.get('laser_angle_max', 0.0),
                            robot.get('laser_max_range', 0.0),
                            offset[0], offset[1],
                        ], dtype=np.float32)
                        self._npz("laser_scans").append(
                            f"r{ridx}_ranges_{step:06d}",
                            np.asarray(ls, dtype=np.float32),
                        )
                        self._npz("laser_scans").append(
                            f"r{ridx}_meta_{step:06d}", meta,
                        )

                # laser points
                if cfg.is_enabled(Channel.LASER_POINTS):
                    lp = robot.get('laser_points')
                    if lp is not None and len(lp) > 0:
                        self._npz("laser_points").append(
                            f"r{ridx}_step_{step:06d}",
                            np.asarray(lp, dtype=np.float32),
                        )

                # occupancy grid
                if cfg.is_enabled(Channel.OCCUPANCY_GRID):
                    occ = robot.get('occupancy')
                    if isinstance(occ, dict) and 'grid' in occ:
                        occ_origin = occ.get('origin', [0.0, 0.0])
                        self._npz("occupancy_grids").append(
                            f"r{ridx}_grid_{step:06d}",
                            np.asarray(occ['grid'], dtype=np.int8),
                        )
                        self._npz("occupancy_grids").append(
                            f"r{ridx}_meta_{step:06d}",
                            np.array([
                                occ.get('resolution', 0.0),
                                occ_origin[0], occ_origin[1],
                            ], dtype=np.float32),
                        )

                # costmap
                if cfg.is_enabled(Channel.COSTMAP):
                    cm = robot.get('costmap')
                    if isinstance(cm, dict) and 'costmap' in cm:
                        cm_origin = cm.get('origin', [0.0, 0.0])
                        self._npz("costmaps").append(
                            f"r{ridx}_grid_{step:06d}",
                            np.asarray(cm['costmap'], dtype=np.float32),
                        )
                        self._npz("costmaps").append(
                            f"r{ridx}_meta_{step:06d}",
                            np.array([
                                cm.get('resolution', 0.0),
                                cm_origin[0], cm_origin[1],
                            ], dtype=np.float32),
                        )

                # navmesh
                if cfg.is_enabled(Channel.NAVMESH):
                    nm = robot.get('navmesh')
                    if isinstance(nm, dict) and 'vertices' in nm:
                        self._npz("navmeshes").append(
                            f"r{ridx}_verts_{step:06d}",
                            np.asarray(nm['vertices'], dtype=np.float32),
                        )
                        self._npz("navmeshes").append(
                            f"r{ridx}_tris_{step:06d}",
                            np.asarray(
                                nm.get('indices', np.zeros((0,), dtype=np.int32)),
                                dtype=np.int32,
                            ),
                        )

                # planner paths
                if cfg.is_enabled(Channel.PLANNER_PATHS):
                    pp = robot.get('planner_paths')
                    if isinstance(pp, dict):
                        self._planner_rows.append({
                            "step": step,
                            "robot_idx": ridx,
                            "global_path": [list(pt) for pt in pp.get('global_path', [])],
                            "dwa_traj": [list(pt) for pt in pp.get('dwa_traj', [])],
                            "p_traj": [list(pt) for pt in pp.get('p_traj', [])],
                        })

                # images
                if cfg.is_enabled(Channel.IMAGES):
                    img = robot.get('agent_image')
                    if isinstance(img, dict):
                        for cam_name, frame in zip(
                            img.get('labels', []), img.get('frames', []),
                        ):
                            key = f"r{ridx}_{cam_name}_{step:06d}"
                            if isinstance(frame, (bytes, bytearray)):
                                self._npz("images").append(
                                    key, np.frombuffer(frame, dtype=np.uint8),
                                )
                            elif isinstance(frame, np.ndarray):
                                self._npz("images").append(key, frame)

        # ── item poses ───────────────────────────────────────────────
        if cfg.is_enabled(Channel.ITEM_POSES) and payload.robots is not None:
            # Collect per-item pose data from the first robot entry that
            # carries item arrays (all robots share the same item list).
            for r in payload.robots:
                if not isinstance(r, dict):
                    continue
                positions = r.get('item_positions')
                orientations = r.get('item_orientations')
                yaws = r.get('item_yaws')
                if positions is not None and len(positions) > 0:
                    item_list = []
                    for i, pos in enumerate(positions):
                        entry: Dict[str, object] = {'position': list(pos)}
                        if orientations is not None and i < len(orientations):
                            entry['orientation_quat'] = list(orientations[i])
                        if yaws is not None and i < len(yaws):
                            entry['yaw'] = float(yaws[i])
                        item_list.append(entry)
                    self._item_rows.append({"step": step, "items": item_list})
                    break  # same data for every robot in this env

        # ── obstacle poses ───────────────────────────────────────────
        if cfg.is_enabled(Channel.OBSTACLE_POSES) and payload.robots is not None:
            for r in payload.robots:
                if not isinstance(r, dict):
                    continue
                obs_positions = r.get('obstacle_positions')
                obs_orientations = r.get('obstacle_orientations')
                obs_dynamic = r.get('obstacle_is_dynamic')
                if obs_positions is not None and len(obs_positions) > 0:
                    obs_list = []
                    for i, pos in enumerate(obs_positions):
                        entry_o: Dict[str, object] = {'position': list(pos)}
                        if obs_orientations is not None and i < len(obs_orientations):
                            entry_o['orientation_quat'] = list(obs_orientations[i])
                        if obs_dynamic is not None and i < len(obs_dynamic):
                            entry_o['is_dynamic'] = bool(obs_dynamic[i])
                        obs_list.append(entry_o)
                    self._obstacle_rows.append({"step": step, "obstacles": obs_list})
                    break  # same obstacle data for every robot in this env

        # ── robots payload (stripped of heavy fields) ────────────────
        if cfg.is_enabled(Channel.ROBOTS_PAYLOAD) and payload.robots is not None:
            stripped: List[Dict[str, object]] = []
            for robot in payload.robots:
                if isinstance(robot, dict):
                    stripped.append({
                        k: v for k, v in robot.items()
                        if k not in _ROBOTS_STRIP_KEYS
                    })
                else:
                    stripped.append(robot)
            self._robots_rows.append({
                "step": step,
                "wall_time": wall,
                "robots": _jsonify(stripped),
            })

        self._step_count += 1

        # periodic flush for long episodes — all bulk arrays
        if self._step_count % self._cfg.flush_interval_steps == 0:
            self._flush_all_npz()

    # ── flushing / finalisation ──────────────────────────────────────

    def _flush_all_npz(self) -> None:
        """Checkpoint all NPZ writers to disk without clearing buffers."""
        for attr_name in dir(self):
            if attr_name.endswith("_writer"):
                writer = getattr(self, attr_name, None)
                if isinstance(writer, _NpzStreamWriter):
                    writer.flush(final=False)

    def finalise(self) -> int:
        """Flush all remaining buffers and close file handles.

        Returns:
            Total number of steps recorded in this episode.
        """
        # Flush all npz writers (final — clear buffers)
        for attr_name in dir(self):
            if attr_name.endswith("_writer"):
                writer = getattr(self, attr_name, None)
                if isinstance(writer, _NpzStreamWriter):
                    writer.flush(final=True)

        # Write JSON-line files for variable-length data
        self._write_jsonl("planner_paths.jsonl", self._planner_rows)
        self._write_jsonl("item_poses.jsonl", self._item_rows)
        self._write_jsonl("obstacle_poses.jsonl", self._obstacle_rows)
        self._write_jsonl("robots_payload.jsonl", self._robots_rows)

        return self._step_count

    def _write_jsonl(self, name: str, rows: List[Dict[str, object]]) -> None:
        """Write *rows* to a JSON-lines file if non-empty.

        Args:
            name: Filename relative to episode directory.
            rows: List of dictionaries to serialise (one per line).
        """
        if not rows:
            return
        path = self._dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=_json_default) + "\n")
        rows.clear()


# =====================================================================
#  Public recorder interface
# =====================================================================

class DataTraceRecorder:
    """Top-level recorder managing episodes across multiple environments.

    The recorder is designed to be attached to the vectorised environment
    loop.  It DOES NOT interfere with the normal training or evaluation
    pipeline — recording is purely observational and append-only.

    Args:
        config: Recording configuration (channels, output directory, …).
    """

    def __init__(self, config: RecordingConfig) -> None:
        self._cfg = config
        self._root = Path(config.trace_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._writers: Dict[int, _EpisodeWriter] = {}
        self._episode_counters: Dict[int, int] = {}
        self._closed = False
        self._run_start = time.monotonic()
        self._meta_path = self._root / "metadata.json"

        # Persist run-level metadata
        self._meta: Dict[str, Any] = {
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": config.to_dict(),
        }
        self._flush_metadata()

        logger.info("DataTraceRecorder initialised → %s", self._root)

    def _flush_metadata(self) -> None:
        """Write the current metadata dict to disk."""
        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._meta, fh, indent=2, default=_json_default)

    def register_spec(self, env_id: int, spec_config: Dict[str, Any]) -> None:
        """Persist a monitor spec configuration for an environment.

        The spec is stored in ``metadata.json`` under the ``"specs"`` key
        so that :class:`~utils.data_trace_player.DataTracePlayer` can
        reconstruct the full :class:`MonitorSpec` during offline replay.

        Args:
            env_id: Environment identifier.
            spec_config: The ``MonitorSpec.config`` dict.
        """
        if self._closed:
            return
        specs = self._meta.setdefault("specs", {})
        specs[str(env_id)] = spec_config
        self._flush_metadata()

    # ── episode lifecycle ────────────────────────────────────────────

    def begin_episode(self, env_id: int) -> None:
        """Open a new episode recording for environment *env_id*.

        If a previous episode for the same *env_id* is still open it is
        finalised automatically (guards against missing ``end_episode``
        calls during auto-reset).

        Args:
            env_id: Integer identifier of the environment instance.
        """
        if self._closed:
            return
        if self._cfg.record_env_ids is not None and env_id not in self._cfg.record_env_ids:
            return

        # Auto-close stale episode
        if env_id in self._writers:
            self.end_episode(env_id)

        ep_idx = self._episode_counters.get(env_id, 0)

        # Honour max-episodes cap
        if self._cfg.max_episodes is not None and ep_idx >= self._cfg.max_episodes:
            return

        ep_dir = self._root / f"env_{env_id:03d}" / f"episode_{ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._writers[env_id] = _EpisodeWriter(ep_dir, self._cfg)
        logger.debug("Episode %d started for env %d → %s", ep_idx, env_id, ep_dir)

    def record_step(
        self,
        env_id: int,
        payload: MonitorPayload,
        step_index: int,
    ) -> None:
        """Record one simulation step for *env_id*.

        Args:
            env_id: Environment instance identifier.
            payload: Telemetry bundle from the environment.
            step_index: Zero-based step counter within the current episode.
        """
        if self._closed:
            return
        writer = self._writers.get(env_id)
        if writer is None:
            return
        writer.record(step_index, payload)

    def end_episode(self, env_id: int) -> None:
        """Finalise and close the current episode for *env_id*.

        Flushes all open file handles and increments the episode counter
        so the next :meth:`begin_episode` call creates a fresh directory.

        Args:
            env_id: Environment instance identifier.
        """
        writer = self._writers.pop(env_id, None)
        if writer is None:
            return
        n_steps = writer.finalise()
        ep_idx = self._episode_counters.get(env_id, 0)
        self._episode_counters[env_id] = ep_idx + 1
        logger.info("Episode %d env %d closed — %d steps", ep_idx, env_id, n_steps)

    # ── shutdown ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close all open episodes.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        for env_id in list(self._writers):
            self.end_episode(env_id)
        logger.info("DataTraceRecorder closed.")

    # ── context-manager support ──────────────────────────────────────

    def __enter__(self) -> DataTraceRecorder:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# =====================================================================
#  Utility helpers
# =====================================================================

def _json_default(obj: object) -> object:
    """JSON-serialise numpy and dataclass types gracefully.

    Args:
        obj: Object that the default JSON encoder cannot handle.

    Returns:
        A JSON-compatible representation.

    Raises:
        TypeError: When *obj* has no known conversion.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bytes, bytearray)):
        import base64
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)  # type: ignore[arg-type]
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _jsonify(obj: object) -> object:
    """Recursively convert an object tree to JSON-safe types.

    Args:
        obj: Arbitrary nested structure of dicts, lists, and numpy types.

    Returns:
        Equivalent structure with all numpy types replaced by Python builtins.
    """
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (bytes, bytearray)):
        import base64
        return {"__bytes_b64__": base64.b64encode(obj).decode("ascii")}
    return obj
