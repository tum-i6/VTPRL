"""
Dynamic Window B (DWB) local planner — Nav2-compatible critic-based controller.

This module implements a local planner modelled after the ROS 2 Nav2 ``dwb_core``
controller.  The key architectural difference from the classic DWA (ROS 1) planner
is the **critic pipeline**: each candidate trajectory is scored by a chain of
independent critic functions, each with its own weight (``scale``).  A trajectory
is immediately rejected (short-circuited) if any critic returns a lethal score,
which prevents the stuck-in-corridor failure mode that plagues monolithic DWA.

Nav2 DWB critics implemented here
----------------------------------
* **PathDist** — minimum Euclidean distance from the rollout endpoint to the
  global path.
* **GoalDist** — Euclidean distance from the rollout endpoint to the goal.
* **PathAlign** — angular deviation of the trajectory heading from the local
  path direction.  This is the critical addition that helps in corridors: DWA
  only measures *distance* to the path; DWB also measures *alignment*.
* **GoalAlign** — angular deviation of the trajectory heading from the direction
  toward the goal.
* **ObstacleFootprint** — two-tier obstacle cost: hard collision at robot
  radius, soft quadratic decay inside an influence zone.
* **PreferForward** — penalises backward motion and pure-rotation commands.
* **RotateToGoal** — when the robot is very close to the goal, discard
  translational commands and score only by final-heading error.
* **Oscillation** — detects repeated sign-flipping in angular velocity and
  forces the robot to commit to the last non-oscillating direction.

Trajectory generation
---------------------
Velocity samples are generated using a ``StandardTrajectoryGenerator`` that
linearly spaces both linear and angular velocities inside the intersection of
the dynamic window and the robot's kinematic limits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

@dataclass
class DWBConfig:
    """Tunable parameters for the DWB controller (Nav2 dwb_core-compatible).

    Parameters shared with DWA use the same naming convention (e.g.
    ``max_linear_vel``) for easier cross-algorithm configuration.  DWB-only
    parameters keep their own names.

    Attributes:
        lookahead: Trajectory simulation horizon in seconds.
        traj_resolution: Number of discrete samples per trajectory rollout.
        min_linear_vel: Hard lower bound on forward velocity (m/s).
        max_linear_vel: Hard upper bound on forward velocity (m/s).
        min_angular_vel: Hard lower bound on angular velocity (rad/s).
        max_angular_vel: Hard upper bound on angular velocity (rad/s).
        max_acc: Maximum linear acceleration (m/s²).
        max_dec: Maximum linear deceleration (m/s²).  Positive value.
        max_ang_acc: Maximum angular acceleration (rad/s²).
        robot_radius: Inscribed robot radius (m) for collision checking.
        safety_distance: Extra clearance beyond robot_radius (m).
        min_dist_goal: Distance tolerance to consider position-goal reached (m).
        yaw_goal_tolerance: Heading tolerance at the goal (rad).
        res_lin_vel_space: Number of linear velocity samples.
        res_ang_vel_space: Number of angular velocity samples.
        short_circuit_on_lethal: Stop evaluating critics once a lethal score is
            returned.
        oscillation_reset_dist: Forward travel (m) that resets oscillation flags.
        oscillation_reset_angle: Rotation (rad) that resets oscillation flags.
        scale_path_dist: Weight for the PathDist critic.
        scale_goal_dist: Weight for the GoalDist critic.
        scale_path_align: Weight for the PathAlign critic.
        scale_goal_align: Weight for the GoalAlign critic.
        scale_obstacle: Weight for the ObstacleFootprint critic.
        scale_prefer_forward: Weight for the PreferForward critic.
        scale_rotate_to_goal: Weight for the RotateToGoal critic (only active
            inside min_dist_goal).
        debug_mode: Print verbose per-trajectory cost breakdown.
    """

    # Trajectory generator
    lookahead: float = 1.0
    traj_resolution: int = 10
    res_lin_vel_space: int = 11
    res_ang_vel_space: int = 11

    # Kinematic limits
    min_linear_vel: float = 0.0
    max_linear_vel: float = 0.8
    min_angular_vel: float = -0.5
    max_angular_vel: float = 0.5
    max_acc: float = 1.0
    max_dec: float = 1.0
    max_ang_acc: float = 1.0

    # Footprint
    robot_radius: float = 0.4
    safety_distance: float = 0.3

    # Goal tolerance
    min_dist_goal: float = 0.12
    yaw_goal_tolerance: float = 0.05

    # Behaviour flags
    short_circuit_on_lethal: bool = True
    oscillation_reset_dist: float = 0.15
    oscillation_reset_angle: float = 0.25

    # Critic weights (names match Nav2 ``dwb_core`` parameters)
    scale_path_dist: float = 32.0
    scale_goal_dist: float = 24.0
    scale_path_align: float = 32.0
    scale_goal_align: float = 24.0
    scale_obstacle: float = 2.0
    scale_prefer_forward: float = 5.0
    scale_rotate_to_goal: float = 32.0

    debug_mode: bool = False


# ---------------------------------------------------------------------------
#  Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DWBResult:
    """Outcome of a single DWB planning step.

    Attributes:
        linear_vel: Chosen forward velocity (m/s).
        angular_vel: Chosen yaw rate (rad/s).
        trajectory: Simulated (x, y, yaw) rollout samples for the winning
            trajectory.
        cost: Aggregate weighted critic cost of the best trajectory.
        debug: Human-readable critic breakdown string.
    """
    linear_vel: float
    angular_vel: float
    trajectory: List[Tuple[float, float, float]]
    cost: float
    debug: str


# ---------------------------------------------------------------------------
#  Oscillation detector
# ---------------------------------------------------------------------------

class _OscillationDetector:
    """Track angular-velocity sign changes and flag oscillation.

    This mirrors Nav2's ``OscillationCritic``.  When the angular command flips
    sign more than ``_MAX_FLIPS`` times within ``_WINDOW`` consecutive steps
    without the robot having travelled ``reset_dist`` or rotated
    ``reset_angle``, the detector locks the sign of the angular command.

    Attributes:
        locked_sign: When non-zero, only angular commands matching this sign
            are permitted.
    """

    _MAX_FLIPS: int = 3
    _WINDOW: int = 6

    def __init__(self, reset_dist: float, reset_angle: float) -> None:
        """Initialise the oscillation detector.

        Args:
            reset_dist: Forward travel distance (m) that resets oscillation state.
            reset_angle: Rotation amount (rad) that resets oscillation state.
        """
        self._reset_dist = max(reset_dist, 1e-6)
        self._reset_angle = max(reset_angle, 1e-6)
        self._history: List[int] = []
        self._accum_dist: float = 0.0
        self._accum_angle: float = 0.0
        self.locked_sign: int = 0  # -1, 0, or +1

    def update(self, ang_vel: float, lin_vel: float, dt: float) -> None:
        """Record a command and update oscillation state.

        Args:
            ang_vel: Angular command that was actually sent.
            lin_vel: Linear command that was actually sent.
            dt: Time step duration.
        """
        self._accum_dist += abs(lin_vel) * dt
        self._accum_angle += abs(ang_vel) * dt

        # Reset when robot has made meaningful progress
        if self._accum_dist >= self._reset_dist or self._accum_angle >= self._reset_angle:
            self._history.clear()
            self._accum_dist = 0.0
            self._accum_angle = 0.0
            self.locked_sign = 0
            return

        sign = 1 if ang_vel > 0.01 else (-1 if ang_vel < -0.01 else 0)
        if sign != 0:
            self._history.append(sign)

        # Keep only the most recent window
        if len(self._history) > self._WINDOW:
            self._history = self._history[-self._WINDOW:]

        # Count sign changes
        flips = 0
        for i in range(1, len(self._history)):
            if self._history[i] != self._history[i - 1]:
                flips += 1

        if flips >= self._MAX_FLIPS and len(self._history) >= self._WINDOW:
            # Lock to the sign of the most recent non-zero command
            self.locked_sign = self._history[-1]

    def is_oscillating(self, ang_vel: float) -> bool:
        """Return True if the proposed angular velocity violates the lock.

        Args:
            ang_vel: Candidate angular velocity.

        Returns:
            True when the candidate sign opposes the locked direction.
        """
        if self.locked_sign == 0:
            return False
        if ang_vel > 0.01 and self.locked_sign < 0:
            return True
        if ang_vel < -0.01 and self.locked_sign > 0:
            return True
        return False

    def reset(self) -> None:
        """Fully reset oscillation state."""
        self._history.clear()
        self._accum_dist = 0.0
        self._accum_angle = 0.0
        self.locked_sign = 0


# ---------------------------------------------------------------------------
#  DWB Local Planner
# ---------------------------------------------------------------------------

class DWBLocalPlanner:
    """Nav2-compatible Dynamic Window B (DWB) local planner.

    The planner generates candidate (linear, angular) velocity pairs from the
    intersection of the dynamic window and the robot's kinematic limits, rolls
    each pair out over ``lookahead`` seconds, scores the resulting trajectory
    through a chain of critics, and returns the lowest-cost feasible command.

    Unlike DWA (ROS 1), DWB:

    * Short-circuits trajectory evaluation when any critic returns a lethal cost.
    * Detects oscillation and locks the angular sign to break deadlocks.
    * Uses explicit **PathAlign** and **GoalAlign** critics that measure heading
      congruence with the path/goal direction — this is the key feature that
      prevents the "stuck in corridor" failure mode where DWA's distance-only
      cost surface has no gradient.
    """

    def __init__(self, config: DWBConfig) -> None:
        """Initialise the planner.

        Args:
            config: DWB configuration dataclass instance.
        """
        self.cfg = config
        self._oscillation = _OscillationDetector(
            reset_dist=config.oscillation_reset_dist,
            reset_angle=config.oscillation_reset_angle,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(
        self,
        current_twist: Tuple[float, float],
        current_pose: Tuple[float, float, float],
        global_path: Sequence[Tuple[float, float]],
        obstacles: Optional[np.ndarray] = None,
        force_follow_plan: bool = True,
    ) -> Optional[DWBResult]:
        """Compute the best (linear_vel, angular_vel) using DWB critics.

        Args:
            current_twist: Current velocities (linear_x, angular_z).
            current_pose: Current pose (x, y, yaw) in world frame.
            global_path: Non-empty list of (x, y) waypoints.
            obstacles: Optional Nx2 array of obstacle points.
            force_follow_plan: When False, skip planning and return None.

        Returns:
            DWBResult when a feasible command exists; ``None`` on total failure.
        """
        if not force_follow_plan:
            return None
        if global_path is None or len(global_path) == 0:
            return None

        lin_vel = float(current_twist[0])
        ang_vel = float(current_twist[1])
        x, y, yaw = float(current_pose[0]), float(current_pose[1]), float(current_pose[2])

        path_arr = np.asarray(global_path, dtype=np.float64)
        if path_arr.ndim != 2 or path_arr.shape[1] < 2:
            return None

        # Trim the path to start from the closest point
        dists = np.hypot(path_arr[:, 0] - x, path_arr[:, 1] - y)
        closest_idx = int(np.argmin(dists))
        path_use = path_arr[closest_idx:]
        if path_use.shape[0] == 0:
            return None

        goal = path_use[-1]
        dist_to_goal = float(np.hypot(goal[0] - x, goal[1] - y))

        # ---- Goal reached → RotateToGoal mode ----
        if dist_to_goal < self.cfg.min_dist_goal:
            return self._rotate_to_goal(x, y, yaw, goal, ang_vel)

        # ---- Generate velocity samples ----
        candidates = self._generate_trajectories(lin_vel, ang_vel, x, y, yaw)

        # ---- Pre-compute local path heading for PathAlign / GoalAlign ----
        local_heading = self._local_path_heading(path_use, x, y)
        goal_heading = math.atan2(goal[1] - y, goal[0] - x)

        obs = obstacles if obstacles is not None and len(obstacles) > 0 else None

        best_cost = math.inf
        best_cmd: Tuple[float, float] = (0.0, 0.0)
        best_traj = np.empty((0, 3))
        best_dbg = ""

        for v_lin, w_ang, end_state, traj in candidates:
            # Oscillation filter: reject commands that flip against the lock
            if self._oscillation.is_oscillating(w_ang):
                continue

            cost, dbg, lethal = self._evaluate_critics(
                v_lin, w_ang, end_state, traj,
                path_use, goal, goal_heading, local_heading,
                obs, dist_to_goal,
            )
            if lethal:
                continue
            if cost < best_cost:
                best_cost = cost
                best_cmd = (v_lin, w_ang)
                best_traj = traj
                best_dbg = dbg

        if best_cost >= math.inf:
            if self.cfg.debug_mode:
                print("[DWB] ALL trajectories infeasible — signalling failure")
            return None

        # Update oscillation tracker with the selected command
        self._oscillation.update(best_cmd[1], best_cmd[0], self.cfg.lookahead)

        traj_list = [tuple(row) for row in best_traj] if best_traj.size > 0 else []

        if self.cfg.debug_mode:
            print(
                "[DWB] best v={:.3f} w={:.3f} cost={:.3f} {}".format(
                    best_cmd[0], best_cmd[1], best_cost, best_dbg,
                )
            )

        return DWBResult(
            linear_vel=best_cmd[0],
            angular_vel=best_cmd[1],
            trajectory=traj_list,
            cost=best_cost,
            debug=best_dbg,
        )

    def reset(self) -> None:
        """Reset internal state (oscillation memory). Call on new goal."""
        self._oscillation.reset()

    # ------------------------------------------------------------------
    #  Trajectory generation (StandardTrajectoryGenerator)
    # ------------------------------------------------------------------

    def _generate_trajectories(
        self,
        lin_vel: float,
        ang_vel: float,
        x: float,
        y: float,
        yaw: float,
    ) -> List[Tuple[float, float, Tuple[float, float, float], np.ndarray]]:
        """Sample velocity pairs from the dynamic window and roll them out.

        Args:
            lin_vel: Current linear velocity (m/s).
            ang_vel: Current angular velocity (rad/s).
            x: Robot x position in world frame.
            y: Robot y position in world frame.
            yaw: Robot heading in world frame (rad).

        Returns:
            List of (v_lin, w_ang, end_state, traj_ndarray) tuples where
            end_state is (x, y, yaw) at rollout end.
        """
        dt_total = self.cfg.lookahead

        # Dynamic window — intersection of kinematic limits and acceleration
        v_min = max(self.cfg.min_linear_vel, lin_vel - self.cfg.max_dec * dt_total)
        v_max = min(self.cfg.max_linear_vel, lin_vel + self.cfg.max_acc * dt_total)
        w_min = max(self.cfg.min_angular_vel, ang_vel - self.cfg.max_ang_acc * dt_total)
        w_max = min(self.cfg.max_angular_vel, ang_vel + self.cfg.max_ang_acc * dt_total)

        v_space = np.linspace(v_min, v_max, self.cfg.res_lin_vel_space)
        w_space = np.linspace(w_min, w_max, self.cfg.res_ang_vel_space)

        results: List[Tuple[float, float, Tuple[float, float, float], np.ndarray]] = []
        n = max(self.cfg.traj_resolution, 1)
        dt = dt_total / n
        steps = np.arange(1, n + 1)

        for v in v_space:
            for w in w_space:
                vf = float(v)
                wf = float(w)
                if abs(wf) < 1e-3:
                    dx = vf * math.cos(yaw) * dt
                    dy = vf * math.sin(yaw) * dt
                    xs = x + dx * steps
                    ys = y + dy * steps
                    yawn = yaw + dt_total * wf
                    traj = np.column_stack([xs, ys, np.full(n, yaw)])
                else:
                    r = vf / wf
                    dx_p = -r * math.sin(yaw)
                    dy_p = r * math.cos(yaw)
                    yawn_arr = yaw + wf * dt * steps
                    xs = x + dx_p + r * np.sin(yawn_arr)
                    ys = y + dy_p - r * np.cos(yawn_arr)
                    yawn = yaw + dt_total * wf
                    traj = np.column_stack([xs, ys, yawn_arr])
                end = (float(xs[-1]), float(ys[-1]), float(yawn))
                results.append((vf, wf, end, traj))

        return results

    # ------------------------------------------------------------------
    #  Critic pipeline
    # ------------------------------------------------------------------

    def _evaluate_critics(
        self,
        v_lin: float,
        w_ang: float,
        end_state: Tuple[float, float, float],
        traj: np.ndarray,
        path: np.ndarray,
        goal: np.ndarray,
        goal_heading: float,
        local_heading: float,
        obstacles: Optional[np.ndarray],
        dist_to_goal: float,
    ) -> Tuple[float, str, bool]:
        """Run the critic chain and return aggregate cost.

        Args:
            v_lin: Candidate linear velocity (m/s).
            w_ang: Candidate angular velocity (rad/s).
            end_state: Final (x, y, yaw) after rollout.
            traj: Full trajectory as (N, 3) array of (x, y, yaw).
            path: Trimmed global path as Nx2 array.
            goal: Goal point as (x, y) array.
            goal_heading: Heading angle from robot to goal (rad).
            local_heading: Heading of the closest path segment (rad).
            obstacles: Optional Nx2 obstacle points.
            dist_to_goal: Distance from robot to goal (m).

        Returns:
            Tuple of (total_cost, debug_string, is_lethal).
        """
        total = 0.0
        parts: List[str] = []

        # 1. ObstacleFootprint
        c_obs = self._critic_obstacle(traj, v_lin, obstacles)
        if c_obs >= math.inf:
            return math.inf, "obs=LETHAL", True
        total += self.cfg.scale_obstacle * c_obs
        if self.cfg.debug_mode:
            parts.append(f"obs={c_obs:.3f}")

        # 2. PathDist
        c_pdist = self._critic_path_dist(end_state, path)
        total += self.cfg.scale_path_dist * c_pdist
        if self.cfg.debug_mode:
            parts.append(f"pdist={c_pdist:.3f}")

        # 3. GoalDist
        c_gdist = self._critic_goal_dist(end_state, goal)
        total += self.cfg.scale_goal_dist * c_gdist
        if self.cfg.debug_mode:
            parts.append(f"gdist={c_gdist:.3f}")

        # 4. PathAlign
        c_palign = self._critic_path_align(end_state, local_heading)
        total += self.cfg.scale_path_align * c_palign
        if self.cfg.debug_mode:
            parts.append(f"palign={c_palign:.3f}")

        # 5. GoalAlign
        c_galign = self._critic_goal_align(end_state, goal_heading)
        total += self.cfg.scale_goal_align * c_galign
        if self.cfg.debug_mode:
            parts.append(f"galign={c_galign:.3f}")

        # 6. PreferForward
        c_fwd = self._critic_prefer_forward(v_lin, w_ang)
        total += self.cfg.scale_prefer_forward * c_fwd
        if self.cfg.debug_mode:
            parts.append(f"fwd={c_fwd:.3f}")

        dbg = ", ".join(parts) if self.cfg.debug_mode else ""
        return total, dbg, False

    # ------------------------------------------------------------------
    #  Individual critics
    # ------------------------------------------------------------------

    def _critic_path_dist(self, end_state: Tuple[float, float, float], path: np.ndarray) -> float:
        """PathDist critic — minimum distance from rollout endpoint to the path.

        Args:
            end_state: Final (x, y, yaw) after trajectory rollout.
            path: Trimmed global path as Nx2 array.

        Returns:
            float: Minimum Euclidean distance to the path, or inf if empty.
        """
        dists = np.hypot(path[:, 0] - end_state[0], path[:, 1] - end_state[1])
        return float(np.min(dists)) if dists.size else math.inf

    def _critic_goal_dist(self, end_state: Tuple[float, float, float], goal: np.ndarray) -> float:
        """GoalDist critic — Euclidean distance from rollout endpoint to the goal.

        Args:
            end_state: Final (x, y, yaw) after trajectory rollout.
            goal: Goal point as (x, y) array.

        Returns:
            float: Euclidean distance to the goal.
        """
        return float(math.hypot(end_state[0] - goal[0], end_state[1] - goal[1]))

    def _critic_path_align(self, end_state: Tuple[float, float, float], local_heading: float) -> float:
        """PathAlign critic — heading deviation from local path direction.

        This is the key critic that prevents corridor sticking by scoring
        heading *alignment* with the path direction, not just distance.

        Args:
            end_state: Final (x, y, yaw) after trajectory rollout.
            local_heading: Heading of the closest path segment (rad).

        Returns:
            float: Normalised angular error in [0, 1].
        """
        err = _wrap_angle(end_state[2] - local_heading)
        return abs(err) / math.pi

    def _critic_goal_align(self, end_state: Tuple[float, float, float], goal_heading: float) -> float:
        """GoalAlign critic — heading deviation from the direction to goal.

        Args:
            end_state: Final (x, y, yaw) after trajectory rollout.
            goal_heading: Heading angle from robot to goal (rad).

        Returns:
            float: Normalised angular error in [0, 1].
        """
        err = _wrap_angle(end_state[2] - goal_heading)
        return abs(err) / math.pi

    def _critic_obstacle(self, traj: np.ndarray, v_lin: float, obstacles: Optional[np.ndarray]) -> float:
        """ObstacleFootprint critic — Nav2-style two-tier obstacle cost.

        Hard collision inside ``robot_radius`` returns infinity.  Within the
        influence zone (``robot_radius + safety_distance + braking_distance``)
        a quadratic proximity cost is applied.

        Args:
            traj: Rollout trajectory as (N, 3) array of (x, y, yaw).
            v_lin: Candidate linear velocity (used for braking distance).
            obstacles: Optional Nx2 array of obstacle points.

        Returns:
            float: Obstacle cost in [0, 1], or ``math.inf`` on collision.
        """
        if obstacles is None or len(obstacles) == 0:
            return 0.0
        traj_xy = traj[:, :2]
        if obstacles.ndim != 2 or obstacles.shape[1] < 2:
            return 0.0
        min_dist = _min_distance(traj_xy, obstacles[:, :2])

        if min_dist < self.cfg.robot_radius:
            return math.inf

        braking = (v_lin ** 2) / max(2.0 * self.cfg.max_dec, 1e-6)
        influence = self.cfg.robot_radius + self.cfg.safety_distance + braking
        if min_dist >= influence:
            return 0.0
        clearance = min_dist - self.cfg.robot_radius
        max_clearance = influence - self.cfg.robot_radius
        normalised = 1.0 - clearance / max(max_clearance, 1e-6)
        return normalised * normalised

    @staticmethod
    def _critic_prefer_forward(v_lin: float, w_ang: float) -> float:
        """PreferForward critic — penalise backward motion and pure rotation.

        Args:
            v_lin: Candidate linear velocity (m/s).
            w_ang: Candidate angular velocity (rad/s).

        Returns:
            float: Cost in [0, 1] (1.0 for backward, 0.4 for pure rotation,
            0.0 otherwise).
        """
        if v_lin < 0.0:
            return 1.0
        if abs(v_lin) < 0.01:
            # Pure rotation is slightly penalized to prefer mixed commands
            return 0.4
        return 0.0

    # ------------------------------------------------------------------
    #  RotateToGoal (separate mode)
    # ------------------------------------------------------------------

    def _rotate_to_goal(
        self,
        x: float,
        y: float,
        yaw: float,
        goal: np.ndarray,
        ang_vel: float,
    ) -> Optional[DWBResult]:
        """RotateToGoal critic — pure rotation at the goal position.

        Scores only by heading error to the goal direction.

        Args:
            x: Robot x position.
            y: Robot y position.
            yaw: Robot heading.
            goal: Goal (x, y).
            ang_vel: Current angular velocity.

        Returns:
            DWBResult with zero linear velocity and best angular command;
            ``None`` when already aligned within tolerance.
        """
        goal_heading = math.atan2(goal[1] - y, goal[0] - x)
        yaw_err = _wrap_angle(goal_heading - yaw)

        if abs(yaw_err) < self.cfg.yaw_goal_tolerance:
            return DWBResult(
                linear_vel=0.0,
                angular_vel=0.0,
                trajectory=[(x, y, yaw)],
                cost=0.0,
                debug="goal_reached",
            )

        # Sample angular velocities within dynamic window
        dt = self.cfg.lookahead
        w_min = max(self.cfg.min_angular_vel, ang_vel - self.cfg.max_ang_acc * dt)
        w_max = min(self.cfg.max_angular_vel, ang_vel + self.cfg.max_ang_acc * dt)
        w_space = np.linspace(w_min, w_max, self.cfg.res_ang_vel_space)

        best_cost = math.inf
        best_w = 0.0
        for w in w_space:
            wf = float(w)
            # Skip oscillating commands
            if self._oscillation.is_oscillating(wf):
                continue
            new_yaw = yaw + wf * dt
            err = abs(_wrap_angle(goal_heading - new_yaw))
            cost = self.cfg.scale_rotate_to_goal * (err / math.pi)
            if cost < best_cost:
                best_cost = cost
                best_w = wf

        if best_cost >= math.inf:
            return None

        self._oscillation.update(best_w, 0.0, dt)

        return DWBResult(
            linear_vel=0.0,
            angular_vel=best_w,
            trajectory=[(x, y, yaw)],
            cost=best_cost,
            debug=f"rotate_to_goal yaw_err={yaw_err:.3f}",
        )

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _local_path_heading(self, path: np.ndarray, x: float, y: float) -> float:
        """Compute the heading of the path segment closest to the robot.

        Args:
            path: Nx2 array of waypoints (already trimmed to closest).
            x: Robot x.
            y: Robot y.

        Returns:
            Heading angle (rad) of the closest path segment.
        """
        if path.shape[0] < 2:
            return math.atan2(path[0, 1] - y, path[0, 0] - x)
        # Use the first segment of the remaining path
        dx = path[1, 0] - path[0, 0]
        dy = path[1, 1] - path[0, 1]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return math.atan2(path[-1, 1] - y, path[-1, 0] - x)
        return math.atan2(dy, dx)


# ---------------------------------------------------------------------------
#  Module-level helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Normalize an angle to the interval [-pi, pi).

    Args:
        a: Input angle in radians.

    Returns:
        Wrapped angle in radians.
    """
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _min_distance(points: np.ndarray, obstacles: np.ndarray) -> float:
    """Minimum pairwise Euclidean distance between two point sets.

    Args:
        points: Nx2 array.
        obstacles: Mx2 array.

    Returns:
        Minimum distance or inf if either set is empty.
    """
    if points.size == 0 or obstacles.size == 0:
        return math.inf
    diffs_x = points[:, 0, np.newaxis] - obstacles[np.newaxis, :, 0]
    diffs_y = points[:, 1, np.newaxis] - obstacles[np.newaxis, :, 1]
    dists = np.hypot(diffs_x, diffs_y)
    return float(np.min(dists))
