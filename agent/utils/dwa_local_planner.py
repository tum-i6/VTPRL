"""
Dynamic Window Approach (DWA) local planner.

Consumes robot state, global path, and obstacle points; produces (linear_vel, angular_vel)
commands plus the best trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple
import math
import numpy as np


@dataclass
class DWAConfig:
    """Tunable parameters for the Dynamic Window Approach controller.

    Attributes:
        freq: Control loop frequency (Hz).
        lookahead: Simulation horizon in seconds for rollouts.
        min_linear_vel: Minimum forward speed (m/s).
        max_linear_vel: Maximum forward speed (m/s).
        min_angular_vel: Minimum yaw rate (rad/s).
        max_angular_vel: Maximum yaw rate (rad/s).
        max_acc: Maximum linear acceleration (m/s^2).
        max_dec: Maximum linear deceleration (m/s^2).
        robot_radius: Robot radius in metres for clearance checks.
        safety_distance: Extra clearance margin in metres.
        min_dist_goal: Goal tolerance in metres.
        res_lin_vel_space: Samples in the linear velocity search space.
        res_ang_vel_space: Samples in the angular velocity search space.
        gain_glob_path: Weight for distance-to-path cost.
        gain_angle_to_goal: Weight for heading-to-goal cost.
        gain_vel: Weight for velocity preference cost.
        gain_prox_to_obst: Weight for obstacle proximity cost.
        debug_mode: Enable verbose logging when True.
    """
    lookahead: float = 1.0
    min_linear_vel: float = 0.0
    max_linear_vel: float = 1.0
    min_angular_vel: float = -1.0
    max_angular_vel: float = 1.0
    max_acc: float = 0.5
    max_dec: float = 0.5
    robot_radius: float = 0.35
    safety_distance: float = 0.3
    min_dist_goal: float = 0.1
    res_lin_vel_space: int = 11
    res_ang_vel_space: int = 11
    gain_glob_path: float = 1.0
    gain_angle_to_goal: float = 1.0
    gain_vel: float = 1.0
    gain_prox_to_obst: float = 0.0
    debug_mode: bool = False


@dataclass
class DWAResult:
    """Outcome of a DWA step.

    Attributes:
        linear_vel: Chosen forward velocity (m/s).
        angular_vel: Chosen yaw rate (rad/s).
        trajectory: Simulated (x, y, yaw) samples for the best command.
        cost: Total scalar cost of that command.
        debug: Human-readable breakdown of costs.
    """
    linear_vel: float
    angular_vel: float
    trajectory: List[Tuple[float, float, float]]
    cost: float
    debug: str


class DWALocalPlanner:
    """Dynamic Window Approach local planner with obstacle and goal costs."""

    def __init__(self, config: DWAConfig):
        """Initialize the planner.

        Args:
            config: DWA configuration parameters.
        """
        self.cfg = config

    def run(
        self,
        current_twist: Tuple[float, float],
        current_pose: Tuple[float, float, float],
        global_path: Sequence[Tuple[float, float]],
        obstacles: np.ndarray,
        force_follow_plan: bool = True,
    ) -> Optional[DWAResult]:
        """Compute the best (linear_vel, angular_vel) using DWA.

        Args:
            current_twist: Current velocities as (linear_x, angular_z).
            current_pose: Current pose (x, y, yaw) in world frame.
            global_path: Sequence of waypoints [(x, y), ...]; must be non-empty.
            obstacles: Nx2 array of obstacle points in world frame (can be empty).
            force_follow_plan: When False, skip planning and return None.

        Returns:
            DWAResult if a command is found; otherwise None.
        """
        if not force_follow_plan:
            return None
        if global_path is None or len(global_path) == 0:
            return None

        lin_vel, ang_vel = float(current_twist[0]), float(current_twist[1])
        robot_state = (float(current_pose[0]), float(current_pose[1]), float(current_pose[2]))
        path_arr = np.asarray(global_path, dtype=float)
        if path_arr.ndim != 2 or path_arr.shape[1] < 2:
            return None

        # choose closest point to start from
        dists = np.hypot(path_arr[:, 0] - robot_state[0], path_arr[:, 1] - robot_state[1])
        idx = int(np.argmin(dists))
        path_use = path_arr[idx:]
        if path_use.shape[0] == 0:
            return None
        goal = path_use[-1]

        # Obstacle cost gain — use the configured weight directly.
        gain_prox_to_obst = self.cfg.gain_prox_to_obst

        Vd = self._dynamic_window(lin_vel, ang_vel)

        if self.cfg.debug_mode:
            print(
                "[DWA] state=({:.2f},{:.2f},{:.2f}) v={:.2f} w={:.2f} path_len={} start_idx={} goal=({:.2f},{:.2f}) gain_obs={:.2f} window_lin=[{:.2f},{:.2f}] window_ang=[{:.2f},{:.2f}]".format(
                    robot_state[0],
                    robot_state[1],
                    robot_state[2],
                    lin_vel,
                    ang_vel,
                    path_arr.shape[0],
                    idx,
                    goal[0],
                    goal[1],
                    gain_prox_to_obst,
                    float(Vd[:, 0, 0].min()),
                    float(Vd[:, 0, 0].max()),
                    float(Vd[0, :, 1].min()),
                    float(Vd[0, :, 1].max()),
                )
            )

        lowest_cost = math.inf
        best_pair = (0.0, 0.0)
        best_traj: np.ndarray = np.empty((0, 3))
        best_debug = ""

        for i in range(Vd.shape[0]):
            for j in range(Vd.shape[1]):
                v_lin, w_ang = Vd[i, j]
                cost, traj, dbg = self._trajectory_cost((v_lin, w_ang), robot_state, path_use, obstacles, gain_prox_to_obst)
                if cost < lowest_cost:
                    lowest_cost = cost
                    best_pair = (v_lin, w_ang)
                    best_traj = traj
                    best_debug = dbg

        # If every sampled trajectory is infeasible (all costs infinite),
        # signal failure so the caller can activate recovery behaviors.
        if lowest_cost >= math.inf:
            if self.cfg.debug_mode:
                print("[DWA] ALL trajectories infeasible — signalling failure")
            return None

        traj_list = [tuple(row) for row in best_traj] if best_traj.size > 0 else []

        # Goal-reached check
        if self._goal_reached(robot_state, goal):
            return DWAResult(linear_vel=0.0, angular_vel=0.0, trajectory=traj_list, cost=lowest_cost, debug="goal reached")

        if self.cfg.debug_mode:
            print(
                "[DWA] best v={:.3f} w={:.3f} cost={:.3f} dbg={}".format(
                    best_pair[0], best_pair[1], lowest_cost, best_debug
                )
            )

        return DWAResult(linear_vel=best_pair[0], angular_vel=best_pair[1], trajectory=traj_list, cost=lowest_cost, debug=best_debug)

    def _dynamic_window(self, lin_vel: float, ang_vel: float) -> np.ndarray:
        """Sample feasible (linear, angular) velocities from the dynamic window.

        Args:
            lin_vel: Current linear velocity (m/s).
            ang_vel: Current angular velocity (rad/s).

        Returns:
            np.ndarray: Grid of shape (res_lin_vel_space, res_ang_vel_space, 2) with
            (linear, angular) pairs.
        """
        dt = self.cfg.lookahead
        lin_min = lin_vel - self.cfg.max_acc * dt
        lin_max = lin_vel + self.cfg.max_acc * dt
        ang_min = ang_vel - self.cfg.max_acc * dt
        ang_max = ang_vel + self.cfg.max_acc * dt

        lin_space = np.linspace(max(self.cfg.min_linear_vel, lin_min), min(self.cfg.max_linear_vel, lin_max), self.cfg.res_lin_vel_space)
        ang_space = np.linspace(max(self.cfg.min_angular_vel, ang_min), min(self.cfg.max_angular_vel, ang_max), self.cfg.res_ang_vel_space)

        yv, xv = np.meshgrid(lin_space, ang_space, indexing="ij")
        return np.stack([yv, xv], axis=-1)

    def _trajectory_cost(
        self,
        control_pair: Tuple[float, float],
        robot_state: Tuple[float, float, float],
        path: np.ndarray,
        obstacles: Optional[np.ndarray],
        gain_prox_to_obst: float,
    ) -> Tuple[float, np.ndarray, str]:
        """Evaluate a control pair and return its total cost and trajectory.

        Args:
            control_pair: (linear_vel, angular_vel) command to evaluate.
            robot_state: Current state (x, y, yaw).
            path: Global path waypoints as Nx2 array.
            obstacles: Optional obstacle points as Nx2 array.
            gain_prox_to_obst: Obstacle cost weight to apply.

        Returns:
            Tuple of (total_cost, trajectory_array, debug_string).
        """
        new_state, traj = self._motion_update(robot_state, control_pair)
        lin_vel = control_pair[0]
        cost_vel = self._vel_cost(lin_vel)
        cost_angle = self._angle_to_goal_cost(new_state, path[-1])
        cost_path = self._path_cost(new_state, path)
        cost_obst = self._obst_cost(traj, control_pair, obstacles) if gain_prox_to_obst > 0.0 else 0.0
        total = self.cfg.gain_vel * cost_vel + self.cfg.gain_glob_path * cost_path + self.cfg.gain_angle_to_goal * cost_angle + gain_prox_to_obst * cost_obst
        if self.cfg.debug_mode:
            debug = f"vel={cost_vel:.3f}, angle={cost_angle:.3f}, path={cost_path:.3f}, obst={cost_obst:.3f}"
        else:
            debug = ""
        return total, traj, debug

    def _motion_update(self, robot_state: Tuple[float, float, float], control_pair: Tuple[float, float], traj_resolution: int = 10) -> Tuple[Tuple[float, float, float], np.ndarray]:
        """Simulate motion over the lookahead horizon for a control pair.

        Args:
            robot_state: Current (x, y, yaw).
            control_pair: (linear_vel, angular_vel) to roll out.
            traj_resolution: Number of discrete steps in the rollout.

        Returns:
            Tuple of (new_state, trajectory) where new_state is (x, y, yaw) at
            horizon end, and trajectory is an (N, 3) array of intermediate samples.
        """
        x, y, yaw = robot_state
        v, w = control_pair
        dt = self.cfg.lookahead / max(traj_resolution, 1)
        steps = np.arange(1, traj_resolution + 1)

        if abs(w) < 1e-3:
            dx = v * math.cos(yaw) * dt
            dy = v * math.sin(yaw) * dt
            yawn = yaw + self.cfg.lookahead * w
            xs = x + dx * steps
            ys = y + dy * steps
            traj = np.column_stack([xs, ys, np.full(traj_resolution, yaw)])
            return (float(xs[-1]), float(ys[-1]), yawn), traj

        r = v / w if w != 0 else 0.0
        dx_partial = -r * math.sin(yaw)
        dy_partial = r * math.cos(yaw)
        yawn = yaw + self.cfg.lookahead * w
        yawn_arr = yaw + w * dt * steps
        xs = x + dx_partial + r * np.sin(yawn_arr)
        ys = y + dy_partial - r * np.cos(yawn_arr)
        traj = np.column_stack([xs, ys, yawn_arr])
        return (float(xs[-1]), float(ys[-1]), yawn), traj

    def _goal_reached(self, robot_state: Tuple[float, float, float], goal: np.ndarray) -> bool:
        """Return True when the robot is within the goal tolerance.

        Args:
            robot_state: Current (x, y, yaw).
            goal: Goal point (x, y).

        Returns:
            bool: True if within min_dist_goal.
        """
        return self._euclidean(robot_state[:2], goal) < self.cfg.min_dist_goal

    def _vel_cost(self, lin_vel: float) -> float:
        """Penalty that grows as linear velocity deviates from the max.

        Args:
            lin_vel: Linear velocity (m/s).

        Returns:
            float: Cost in [0, inf).
        """
        span = max(self.cfg.max_linear_vel - self.cfg.min_linear_vel, 1e-6)
        raw = ((self.cfg.max_linear_vel - self.cfg.min_linear_vel) - lin_vel) / span
        return max(raw, 0.0)

    def _angle_to_goal_cost(self, new_state: Tuple[float, float, float], goal: np.ndarray) -> float:
        """Normalized angular error to the goal direction (0..1).

        Args:
            new_state: (x, y, yaw) after rollout.
            goal: Goal point (x, y).

        Returns:
            float: Normalized heading error in [0, 1].
        """
        xn, yn, yawn = new_state
        gx, gy = goal
        angle = math.atan2(gy - yn, gx - xn) - yawn
        return abs(math.atan2(math.sin(angle), math.cos(angle))) / math.pi

    def _path_cost(self, new_state: Tuple[float, float, float], path: np.ndarray) -> float:
        """Distance from the new state to the nearest point on the path.

        Args:
            new_state: (x, y, yaw) after rollout.
            path: Global path waypoints as Nx2 array.

        Returns:
            float: Minimum Euclidean distance to the path.
        """
        dists = np.hypot(path[:, 0] - new_state[0], path[:, 1] - new_state[1])
        return float(np.min(dists)) if dists.size else math.inf

    def _obst_cost(self, traj: np.ndarray, control_pair: Tuple[float, float], obstacles: Optional[np.ndarray]) -> float:
        """ROS1-style obstacle proximity cost.

        Uses a simple two-zone model consistent with the ROS1 DWA local
        planner's costmap-based scoring:

        1. **Hard collision** (cost = inf): trajectory point enters the robot
           footprint (``min_dist < robot_radius``).
        2. **Linear proximity** (cost in [0, 1]): linear decay from 1.0 at
           the robot surface to 0.0 at the safety distance edge.
        3. **Free space** (cost = 0): trajectory is farther than
           ``robot_radius + safety_distance`` from all obstacles.

        Args:
            traj: Rollout trajectory as (N, 3) ndarray of (x, y, yaw) samples.
            control_pair: (linear_vel, angular_vel) used to generate the
                trajectory.  Unused in ROS1 model (kept for interface
                compatibility).
            obstacles: Optional obstacle points as Nx2 array.

        Returns:
            float: Linear proximity cost in [0, 1], or ``math.inf`` on
            collision.
        """
        if obstacles is None or len(obstacles) == 0:
            return 0.0
        traj_xy = traj[:, :2]
        if obstacles.ndim != 2 or obstacles.shape[1] < 2:
            return 0.0
        min_dist = self._min_distance(traj_xy, obstacles[:, :2])

        # Hard collision: trajectory enters the physical robot footprint
        if min_dist < self.cfg.robot_radius:
            return math.inf

        # Linear proximity penalty within the safety zone
        clearance = min_dist - self.cfg.robot_radius
        if clearance >= self.cfg.safety_distance:
            return 0.0
        return 1.0 - clearance / max(self.cfg.safety_distance, 1e-6)

    @staticmethod
    def _euclidean(p1: Sequence[float], p2: Sequence[float]) -> float:
        """Euclidean distance between two (x, y) points.

        Args:
            p1: First point (x, y).
            p2: Second point (x, y).

        Returns:
            float: Euclidean distance.
        """
        return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))

    @staticmethod
    def _min_distance(points: np.ndarray, obstacles: np.ndarray) -> float:
        """Minimum pairwise distance between point set and obstacle set.

        Args:
            points: Waypoints array (N,2).
            obstacles: Obstacles array (M,2).

        Returns:
            float: Minimum pairwise Euclidean distance or inf when empty.
        """
        if points.size == 0 or obstacles.size == 0:
            return math.inf
        pts = points[:, :2]
        obs = obstacles[:, :2]
        dists = np.hypot(pts[:, None, 0] - obs[None, :, 0], pts[:, None, 1] - obs[None, :, 1])
        return float(np.min(dists)) if dists.size else math.inf
