from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict
import numpy as np


@dataclass
class Pose2D:
    """Planar pose in the ROS frame.

    Args:
        x: Position along +X (meters).
        y: Position along +Y (meters).
        yaw: Heading about +Z in radians (right-handed, ROS convention).
    """

    x: float
    y: float
    yaw: float


@dataclass
class LaserScanData:
    """Raw laser scan polar data in ROS frame.

    Args:
        ranges: Range measurements (meters).
        angle_min: Start angle of the scan (radians, ROS frame).
        angle_max: End angle of the scan (radians, ROS frame).
        max_range: Maximum valid range (meters).
        sensor_offset_xy: Sensor offset from base in base frame (x, y meters).
        frame_id: Optional frame identifier string.
    """

    ranges: np.ndarray
    angle_min: float
    angle_max: float
    max_range: float
    sensor_offset_xy: Tuple[float, float] = (0.0, 0.0)
    frame_id: str = ""


@dataclass
class LaserPoints:
    """Cartesian projection of a laser scan in world frame.

    Args:
        points_xy: Nx2 array of points (meters) in world frame.
        origin_xy: Sensor origin (x, y meters) in world frame.
        angle_offset_applied: Angular offset used during projection (radians).
    """

    points_xy: np.ndarray
    origin_xy: Tuple[float, float]
    angle_offset_applied: float = 0.0

    def as_list(self) -> List[List[float]]:
        """Return points as a Python list for serialization.

        Returns:
            List of [x, y] pairs; empty list if no points.
        """

        if self.points_xy is None or self.points_xy.size == 0:
            return []
        return [[float(x), float(y)] for x, y in self.points_xy]


@dataclass
class OccupancyGridData:
    """Occupancy grid payload (ROS costmap-style).

    Args:
        grid: HxW grid; values in [0,100] or -1 for unknown.
        resolution: Cell size in meters.
        origin_xy: Origin (x, y meters) of the grid in world frame.
    """

    grid: np.ndarray
    resolution: float
    origin_xy: Tuple[float, float] = (0.0, 0.0)


@dataclass
class CostmapData:
    """Planner costmap payload.

    Args:
        grid: HxW floating-point cost grid.
        resolution: Cell size in meters.
        origin_xy: Origin (x, y meters) of the grid in world frame.
    """

    grid: np.ndarray
    resolution: float
    origin_xy: Tuple[float, float] = (0.0, 0.0)


@dataclass
class NavmeshData:
    """2D projection of a navigation mesh.

    Args:
        vertices: Nx2 vertex array (meters) in world frame.
        triangles: Mx3 triangle indices referencing ``vertices``.
    """

    vertices: np.ndarray
    triangles: np.ndarray


@dataclass
class PlannerPaths:
    """Planner trajectory overlays.

    Args:
        global_path: Waypoints of the global path (x, y meters).
        dwa_traj: DWA rollout states (x, y, yaw radians).
        p_traj: P-controller preview states (x, y, yaw radians).
    """

    global_path: List[Tuple[float, float]] = field(default_factory=list)
    dwa_traj: List[Tuple[float, float, float]] = field(default_factory=list)
    p_traj: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class PlannerData:
    """Bundle of planner inputs/outputs for monitoring.

    Args:
        paths: Global and local planner paths.
        occupancy: Occupancy grid used by the planner.
        costmap: Optional costmap overlay.
        laser_points: Projected laser points used by the planner.
        robot_pose: Current robot pose in planner frame/world.
        target_pos: Target position (x, y meters) in world frame.
    """

    paths: PlannerPaths = field(default_factory=PlannerPaths)
    occupancy: Optional[OccupancyGridData] = None
    costmap: Optional[CostmapData] = None
    laser_points: Optional[LaserPoints] = None
    robot_pose: Optional[Pose2D] = None
    target_pos: Optional[Tuple[float, float]] = None


@dataclass
class MonitorPayload:
    """Unified telemetry bundle for the task monitor.

    Args:
        robot_pose: Current robot pose in world frame.
        robot_velocity: Linear and angular velocity (v, omega).
        target_delta: Delta from robot to target (dx, dy, dyaw) in world frame.
        reward: Latest reward value.
        success: Episode success flag.
        collision: Collision flag.
        laser_scan: Optional raw laser scan.
        laser_points: Optional projected laser points.
        occupancy: Optional occupancy grid.
        costmap: Optional planner costmap.
        navmesh: Optional navmesh snapshot (projected to 2D).
        planner: Optional planner overlay data.
        target_pos: Optional target position (x, y meters).
        agent_state: Optional agent state vector.
        agent_action: Optional last agent action.
        agent_reward: Optional reward array for plotting.
        agent_image: Optional image payload (bytes or list of bytes).
    """

    robot_pose: Pose2D
    robot_velocity: Tuple[float, float]
    target_delta: Tuple[float, float, float]
    reward: float
    success: bool
    collision: bool
    laser_scan: Optional[LaserScanData] = None
    laser_points: Optional[LaserPoints] = None
    occupancy: Optional[OccupancyGridData] = None
    costmap: Optional[CostmapData] = None
    navmesh: Optional[NavmeshData] = None
    planner: Optional[PlannerData] = None
    target_pos: Optional[Tuple[float, float]] = None
    agent_state: Optional[np.ndarray] = None
    agent_action: Optional[np.ndarray] = None
    agent_reward: Optional[np.ndarray] = None
    agent_image: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert payload to a dict compatible with the task monitor.

        Returns:
            Dictionary keyed to legacy monitor field names with numpy arrays preserved where useful.
        """

        data: Dict[str, Any] = {
            'robot_position': [self.robot_pose.x, self.robot_pose.y],
            'robot_yaw': self.robot_pose.yaw,
            'robot_velocity': list(self.robot_velocity),
            'target_delta': list(self.target_delta),
            'reward': self.reward,
            'success': self.success,
            'collision': self.collision,
        }
        if self.target_pos is not None:
            data['target_position'] = list(self.target_pos)
        if self.agent_state is not None:
            data['agent_state'] = np.asarray(self.agent_state)
        if self.agent_action is not None:
            data['agent_action'] = np.asarray(self.agent_action)
        if self.agent_reward is not None:
            data['agent_reward'] = np.asarray(self.agent_reward)
        if self.agent_image is not None:
            data['agent_image'] = self.agent_image
        if self.laser_scan is not None:
            data['laser_scan'] = np.asarray(self.laser_scan.ranges)
            data['laser_angle_min'] = float(self.laser_scan.angle_min)
            data['laser_angle_max'] = float(self.laser_scan.angle_max)
            data['laser_max_range'] = float(self.laser_scan.max_range)
            data['laser_sensor_offset'] = list(self.laser_scan.sensor_offset_xy)
        if self.laser_points is not None:
            data['laser_points'] = self.laser_points.as_list()
            data['laser_angle_offset'] = float(self.laser_points.angle_offset_applied)
            data['laser_origin'] = list(self.laser_points.origin_xy)
        if self.occupancy is not None:
            occ_payload = {
                'grid': self.occupancy.grid,
                'resolution': self.occupancy.resolution,
                'origin': list(self.occupancy.origin_xy),
            }
            data['occupancy'] = occ_payload
        if self.costmap is not None:
            cost_payload = {
                'costmap': self.costmap.grid,
                'resolution': self.costmap.resolution,
                'origin': list(self.costmap.origin_xy),
            }
            data['costmap'] = cost_payload
        if self.navmesh is not None:
            data['navmesh'] = {
                'vertices': self.navmesh.vertices,
                'indices': self.navmesh.triangles,
            }
        if self.planner is not None:
            planner_payload: Dict[str, Any] = {
                'global_path': list(self.planner.paths.global_path),
                'dwa_traj': list(self.planner.paths.dwa_traj),
                'p_traj': list(self.planner.paths.p_traj),
                'robot_position': [self.robot_pose.x, self.robot_pose.y],
                'robot_yaw': self.robot_pose.yaw,
                'target_position': list(self.planner.target_pos) if self.planner.target_pos is not None else None,
            }
            if self.planner.laser_points is not None:
                planner_payload['laser_points'] = self.planner.laser_points.as_list()
            if self.planner.occupancy is not None:
                planner_payload['occupancy'] = {
                    'grid': self.planner.occupancy.grid,
                    'resolution': self.planner.occupancy.resolution,
                    'origin': list(self.planner.occupancy.origin_xy),
                }
            if self.planner.costmap is not None:
                planner_payload['costmap'] = {
                    'costmap': self.planner.costmap.grid,
                    'resolution': self.planner.costmap.resolution,
                    'origin': list(self.planner.costmap.origin_xy),
                }
            data['planner_map'] = planner_payload
        return data
