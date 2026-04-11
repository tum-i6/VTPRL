"""Utilities for segmented configuration expansion and pose resolution.

This module centralizes shared config helpers used by manipulator and
warehouse environments to avoid duplicated expansion logic.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence
import numpy as np


def _safe_count(value: Any, default: int) -> int:
    """Convert an arbitrary value to a non-negative integer count.

    Args:
        value: Input value that should represent a count (for example, a
            numeric value or numeric string).
        default: Fallback count used when ``value`` cannot be converted.

    Returns:
        A non-negative integer count. Negative values are clamped to ``0``.
    """
    try:
        count = int(value)
    except Exception:
        count = int(default)
    return max(count, 0)


def expand_segmented_entries(
    segments: Any,
    default_count: int = 1,
) -> List[Dict[str, Any]]:
    """Expand segmented configuration entries into per-instance dictionaries.

    Each segment is expected to be a dict with optional `count` and other
    per-segment fields. The returned list contains one dict per expanded
    instance and does not include the `count` key.

    Args:
        segments: List-like segmented configuration where each valid segment is
            a dictionary that may include a ``count`` field.
        default_count: Fallback number of instances to expand for a segment
            when ``count`` is missing or invalid.

    Returns:
        A list of expanded dictionaries (one dictionary per instance). Any
        ``count`` key is removed from returned dictionaries.
    """
    expanded: List[Dict[str, Any]] = []
    if not isinstance(segments, list):
        return expanded

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        count = _safe_count(segment.get("count", default_count), default_count)
        entry = dict(segment)
        entry.pop("count", None)
        for _ in range(count):
            expanded.append(dict(entry))

    return expanded


def expand_manipulator_instances(manipulator_config: Any) -> List[Dict[str, Any]]:
    """Expand manipulator segment configuration into per-robot instances.

    Args:
        manipulator_config: Manipulator configuration dictionary expected to
            contain a ``manipulators`` list of segmented entries.

    Returns:
        A list of expanded manipulator instance dictionaries. Returns an empty
        list when ``manipulator_config`` is not a dictionary or contains no
        valid manipulator segments.
    """
    if not isinstance(manipulator_config, dict):
        return []
    return expand_segmented_entries(manipulator_config.get("manipulators", []), default_count=1)


def expand_item_instances(manipulator_config: Any) -> List[Dict[str, Any]]:
    """Expand item segment configuration into per-item instances.

    Args:
        manipulator_config: Manipulator configuration dictionary expected to
            contain an ``items`` list of segmented entries.

    Returns:
        A list of expanded item instance dictionaries. Returns an empty list
        when ``manipulator_config`` is not a dictionary or contains no valid
        item segments.
    """
    if not isinstance(manipulator_config, dict):
        return []
    return expand_segmented_entries(manipulator_config.get("items", []), default_count=1)


def _coerce_pose6(pose: Any, default_pose: Sequence[float]) -> List[float]:
    """Convert a pose value into a 6-element Unity pose list.

    Args:
        pose: Candidate pose value. If it is a list/tuple, its values are used;
            otherwise ``default_pose`` is used.
        default_pose: Fallback pose values used when ``pose`` is invalid.

    Returns:
        A 6-element list of floats in Unity pose order
        ``[x, y, z, rx, ry, rz]``. Short inputs are padded with zeros.
    """
    values = list(pose) if isinstance(pose, (list, tuple)) else list(default_pose)
    if len(values) < 6:
        values.extend([0.0] * (6 - len(values)))
    return [float(values[0]), float(values[1]), float(values[2]), float(values[3]), float(values[4]), float(values[5])]


def resolve_robot_poses_unity(
    manipulator_config: Any,
    default_pose: Optional[Sequence[float]] = None,
) -> List[List[float]]:
    """Resolve one Unity robot base pose per expanded manipulator instance.

    Pose priority per robot:
    1) Manipulator instance ``base_pose``
    2) Manipulator instance ``robot_pose``
    3) Global ``robot_poses[index]``
    4) ``default_pose`` fallback

    Args:
        manipulator_config: Manipulator configuration dictionary containing
            segmented manipulator entries and optional global ``robot_poses``.
        default_pose: Fallback pose used when no robot-specific pose is
            available. Defaults to ``[0, 0, 0, 0, 0, 0]``.

    Returns:
        A list of 6D Unity robot base poses (one per resolved robot instance).
        When no instances are found, returns global poses if available,
        otherwise a single default pose.
    """
    if default_pose is None:
        default_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    instances = expand_manipulator_instances(manipulator_config)
    global_robot_poses = []
    if isinstance(manipulator_config, dict):
        raw = manipulator_config.get("robot_poses")
        if isinstance(raw, list):
            global_robot_poses = raw

    if not instances:
        if len(global_robot_poses) > 0:
            return [_coerce_pose6(pose, default_pose) for pose in global_robot_poses]
        return [_coerce_pose6(default_pose, default_pose)]

    poses: List[List[float]] = []
    for idx, instance in enumerate(instances):
        pose = instance.get("base_pose")
        if pose is None:
            pose = instance.get("robot_pose")
        if pose is None and idx < len(global_robot_poses):
            pose = global_robot_poses[idx]
        poses.append(_coerce_pose6(pose, default_pose))

    return poses


def resolve_target_poses_unity(
    manipulator_config: Any,
    robot_count: int,
    default_pose: Sequence[float],
) -> List[List[float]]:
    """Resolve target poses with one entry per robot.

    Priority per robot:
    1) manipulator instance `target_pose`
    2) global `target_poses[index]`
    3) provided default pose

    Args:
        manipulator_config: Manipulator configuration dictionary containing
            segmented manipulator entries and optional global ``target_poses``.
        robot_count: Number of robot target poses to resolve.
        default_pose: Fallback pose. Can be a single pose sequence applied to
            all robots or a per-robot list of pose sequences.

    Returns:
        A list of resolved 6D target poses with length ``max(robot_count, 1)``.
    """
    robot_count = max(int(robot_count), 1)
    instances = expand_manipulator_instances(manipulator_config)
    global_target_poses = []
    if isinstance(manipulator_config, dict):
        raw = manipulator_config.get("target_poses")
        if isinstance(raw, list):
            global_target_poses = raw

    default_pose_is_per_robot = (
        isinstance(default_pose, list)
        and len(default_pose) > 0
        and isinstance(default_pose[0], (list, tuple))
    )

    poses: List[List[float]] = []
    for idx in range(robot_count):
        pose = None
        if idx < len(instances):
            pose = instances[idx].get("target_pose")
        if pose is None and idx < len(global_target_poses):
            pose = global_target_poses[idx]
        fallback_default = default_pose[idx] if default_pose_is_per_robot and idx < len(default_pose) else default_pose
        poses.append(_coerce_pose6(pose, fallback_default))
    return poses


def resolve_item_poses_unity(
    manipulator_config: Any,
    default_pose: Sequence[float],
) -> List[List[float]]:
    """Resolve Unity item poses with one entry per expanded item instance.

    Pose priority per item:
    1) Item instance ``item_pose``
    2) Global ``item_poses[index]``
    3) ``default_pose`` fallback

    Args:
        manipulator_config: Manipulator configuration dictionary containing
            segmented item entries and optional global ``item_poses``.
        default_pose: Fallback pose used when item-specific pose data is
            missing.

    Returns:
        A list of resolved 6D Unity item poses. Returns an empty list when
        there are no item instances and no global item poses.
    """
    instances = expand_item_instances(manipulator_config)
    global_item_poses = []
    if isinstance(manipulator_config, dict):
        raw = manipulator_config.get("item_poses")
        if isinstance(raw, list):
            global_item_poses = raw

    if not instances:
        if len(global_item_poses) > 0:
            return [_coerce_pose6(pose, default_pose) for pose in global_item_poses]
        return []

    poses: List[List[float]] = []
    for idx, instance in enumerate(instances):
        pose = instance.get("item_pose")
        if pose is None and idx < len(global_item_poses):
            pose = global_item_poses[idx]
        poses.append(_coerce_pose6(pose, default_pose))

    return poses


def first_manipulator_instance(manipulator_config: Any) -> Dict[str, Any]:
    """Return the first expanded manipulator instance.

    Args:
        manipulator_config: Manipulator configuration dictionary containing
            segmented manipulator entries.

    Returns:
        The first expanded manipulator instance dictionary, or an empty
        dictionary when no instances are available.
    """
    instances = expand_manipulator_instances(manipulator_config)
    return instances[0] if instances else {}


def _angle_axis_to_rotation_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from an angle-axis (rotation vector).

    The direction ``[ax, ay, az]`` is the rotation axis and the magnitude
    ``theta = ||[ax, ay, az]||`` is the rotation angle in radians.  Uses the
    Rodrigues formula.

    Args:
        ax: X component of the angle-axis rotation vector in radians.
        ay: Y component of the angle-axis rotation vector in radians.
        az: Z component of the angle-axis rotation vector in radians.

    Returns:
        A 3x3 rotation matrix as a NumPy array.
    """
    theta = math.sqrt(ax * ax + ay * ay + az * az)
    if theta < 1e-12:
        return np.eye(3)
    kx, ky, kz = ax / theta, ay / theta, az / theta
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + kx * kx * C,      kx * ky * C - kz * s, kx * kz * C + ky * s],
        [ky * kx * C + kz * s, c + ky * ky * C,      ky * kz * C - kx * s],
        [kz * kx * C - ky * s, kz * ky * C + kx * s, c + kz * kz * C     ],
    ])


def angle_axis_to_euler_xyz(ax: float, ay: float, az: float) -> List[float]:
    """Convert an angle-axis (rotation vector) to extrinsic XYZ Euler angles.

    Chains the Rodrigues formula with the Euler-XYZ decomposition so that
    downstream code that expects Euler angles receives the correct values.

    Args:
        ax: X component of the angle-axis rotation vector in radians.
        ay: Y component of the angle-axis rotation vector in radians.
        az: Z component of the angle-axis rotation vector in radians.

    Returns:
        A list ``[rx, ry, rz]`` of extrinsic XYZ Euler angles in radians.
    """
    R = _angle_axis_to_rotation_matrix(ax, ay, az)
    return _rotation_matrix_to_euler_xyz(R)


def euler_xyz_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3x3 rotation matrix from extrinsic XYZ Euler angles.

    Args:
        rx: Rotation around the X axis in radians.
        ry: Rotation around the Y axis in radians.
        rz: Rotation around the Z axis in radians.

    Returns:
        A 3x3 rotation matrix as a NumPy array.
    """
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    # R = Rz * Ry * Rx  (extrinsic XYZ == intrinsic ZYX)
    return np.array([
        [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
        [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
        [-sy,     sx * cy,                cx * cy               ],
    ])


def _rotation_matrix_to_euler_xyz(R: np.ndarray) -> List[float]:
    """Extract extrinsic XYZ Euler angles from a 3x3 rotation matrix.

    Args:
        R: A 3x3 rotation matrix.

    Returns:
        A list ``[rx, ry, rz]`` of extrinsic XYZ Euler angles in radians.
    """
    sy = -R[2, 0]
    cy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if cy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(sy, cy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(sy, cy)
        rz = 0.0

    return [rx, ry, rz]


def transform_local_pose_to_world(
    robot_pose: Sequence[float],
    local_pose: Sequence[float],
) -> List[float]:
    """Transform a pose from robot-local frame to the world frame.

    Both poses are 6D: ``[x, y, z, rx, ry, rz]`` with Euler XYZ angles in
    radians.  The *local_pose* is defined relative to *robot_pose*.  The
    function applies the full homogeneous transformation so that both
    position and orientation are correctly composed.

    Args:
        robot_pose: The 6D world pose of the robot base.
        local_pose: The 6D pose expressed in the robot base’s local frame.

    Returns:
        A 6D world pose ``[x, y, z, rx, ry, rz]``.
    """
    rp = [float(v) for v in robot_pose]
    lp = [float(v) for v in local_pose]

    # Pad to length 6 if needed
    while len(rp) < 6:
        rp.append(0.0)
    while len(lp) < 6:
        lp.append(0.0)

    R_robot = euler_xyz_to_rotation_matrix(rp[3], rp[4], rp[5])
    t_robot = np.array(rp[:3])

    R_local = euler_xyz_to_rotation_matrix(lp[3], lp[4], lp[5])
    t_local = np.array(lp[:3])

    # World position: R_robot * t_local + t_robot
    t_world = R_robot @ t_local + t_robot

    # World orientation: R_robot * R_local
    R_world = R_robot @ R_local
    euler_world = _rotation_matrix_to_euler_xyz(R_world)

    return [t_world[0], t_world[1], t_world[2],
            euler_world[0], euler_world[1], euler_world[2]]
