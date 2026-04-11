"""Data-trace channel definitions and recording configuration.

This module defines the set of recordable *channels* (data modalities)
produced by the simulator environments (warehouse AMR, manipulator, etc.)
together with a lightweight configuration object that controls which
channels are captured at runtime.

Storage format
--------------
Each episode is written into a self-contained directory.  Dense arrays
(images, laser scans, grids, observation vectors) are stored as NumPy
``.npz`` archives with per-robot key prefixes (``r{idx}_``).  Scalar and
variable-length telemetry (robot state dicts, planner paths, item poses)
are stored as JSON-lines (``.jsonl``) files.

.. code-block:: text

    <trace_root>/
        metadata.json                   # run-level metadata & env specs
        env_000/
            episode_0000/
                robots_payload.jsonl    # per-step per-robot scalar dicts
                planner_paths.jsonl     # per-step planner trajectories
                item_poses.jsonl        # per-step movable item poses
                obstacle_poses.jsonl    # per-step obstacle poses (static + dynamic)
                laser_scans.npz         # per-robot laser range arrays
                laser_points.npz        # per-robot projected laser points
                occupancy_grids.npz     # per-robot occupancy grids
                costmaps.npz            # per-robot costmaps
                navmeshes.npz           # per-robot navmesh vertex/triangle
                agent_states.npz        # per-robot observation vectors
                images.npz              # per-robot camera frames
            episode_0001/
                ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, Optional, Set


class Channel(Enum):
    """Identifiers for every recordable data modality.

    Each member maps to one data stream produced by the environment's
    ``get_monitor_payload()`` method.  The recorder checks which channels
    are enabled before writing any artefact.
    """

    # ── Observation vector ───────────────────────────────────────────
    AGENT_STATE = auto()
    """Full observation / state vector fed to the policy (per-robot)."""

    # ── Dense sensor arrays (warehouse-specific) ─────────────────────
    LASER_SCAN = auto()
    """Raw laser-range array plus scan metadata (per-robot)."""

    LASER_POINTS = auto()
    """Laser points projected into world frame — Nx2 (per-robot)."""

    # ── Spatial maps (warehouse-specific) ────────────────────────────
    OCCUPANCY_GRID = auto()
    """Rasterised occupancy grid from the navmesh (per-robot)."""

    COSTMAP = auto()
    """Floating-point planner costmap (per-robot)."""

    NAVMESH = auto()
    """2-D projected navigation mesh — vertices + triangles (per-robot)."""

    # ── Planner data (warehouse-specific) ────────────────────────────
    PLANNER_PATHS = auto()
    """Global path, DWA trajectory and P-controller trajectory (per-robot)."""

    # ── Camera / images ──────────────────────────────────────────────
    IMAGES = auto()
    """Camera image frames — overhead and/or per-robot."""

    # ── Item poses ───────────────────────────────────────────────────
    ITEM_POSES = auto()
    """Poses of all movable items in the environment."""

    # ── Obstacle poses ───────────────────────────────────────────────
    OBSTACLE_POSES = auto()
    """Poses of all obstacles (static and dynamic) in the environment."""

    # ── Multi-robot bundle ───────────────────────────────────────────
    ROBOTS_PAYLOAD = auto()
    """Per-robot scalar telemetry dictionaries (JSONL)."""


# ── Channel groups ───────────────────────────────────────────────────

ALL_CHANNELS: FrozenSet[Channel] = frozenset(Channel)
"""Every available channel."""


@dataclass
class RecordingConfig:
    """Runtime configuration for a data-trace recording session.

    Args:
        enabled_channels: Set of channels to capture.  Defaults to all.
        trace_root: Filesystem directory for trace output.
        compress_arrays: Whether to compress bulk ``.npz`` files.
        max_episodes: Optional cap on number of episodes to record.
        flush_interval_steps: Flush buffered NPZ arrays every *N* steps.
        record_env_ids: Optional set of environment ids to record.
            When ``None`` every environment is recorded.
    """

    enabled_channels: Set[Channel] = field(default_factory=lambda: set(ALL_CHANNELS))
    trace_root: str = "traces"
    compress_arrays: bool = True
    max_episodes: Optional[int] = None
    flush_interval_steps: int = 200
    record_env_ids: Optional[Set[int]] = None

    # ── helpers ──────────────────────────────────────────────────────

    def is_enabled(self, channel: Channel) -> bool:
        """Check whether *channel* is selected for recording.

        Args:
            channel: The data channel to query.

        Returns:
            ``True`` when the channel should be captured.
        """
        return channel in self.enabled_channels

    def to_dict(self) -> Dict[str, object]:
        """Serialise to a JSON-safe dictionary for metadata persistence.

        Returns:
            Dictionary containing all configuration fields with enum names
            converted to strings.
        """
        return {
            "enabled_channels": sorted(ch.name for ch in self.enabled_channels),
            "trace_root": self.trace_root,
            "compress_arrays": self.compress_arrays,
            "max_episodes": self.max_episodes,
            "flush_interval_steps": self.flush_interval_steps,
            "record_env_ids": sorted(self.record_env_ids) if self.record_env_ids is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> RecordingConfig:
        """Reconstruct from a dictionary (e.g. loaded from metadata JSON).

        Args:
            data: Dictionary with the same keys produced by :meth:`to_dict`.

        Returns:
            A new :class:`RecordingConfig` instance.
        """
        channels = {Channel[name] for name in data.get("enabled_channels", [ch.name for ch in ALL_CHANNELS])}
        env_ids = data.get("record_env_ids")
        return cls(
            enabled_channels=channels,
            trace_root=str(data.get("trace_root", "traces")),
            compress_arrays=bool(data.get("compress_arrays", True)),
            max_episodes=data.get("max_episodes"),  # type: ignore[arg-type]
            flush_interval_steps=int(data.get("flush_interval_steps", 200)),
            record_env_ids=set(env_ids) if env_ids is not None else None,
        )
