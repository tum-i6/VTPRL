"""
Warehouse Unity Gym environment compatible with SimulatorVecEnv.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import gym
from gym import spaces
import base64
import math

from utils.navmesh_occupancy import navmesh_to_occupancy_grid, OccupancyGridResult
from utils.telemetry import (
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
from utils.astar_planner import astar, AStarResult
from utils.dwa_local_planner import DWALocalPlanner, DWAConfig, DWAResult


class WarehouseUnityEnv(gym.Env):
    """Gym-compatible warehouse navigation environment backed by Unity.

    The environment exposes a continuous action space (linear and angular velocity)
    and builds observations from Unity-provided robot/target state plus optional
    laser scans. Planning helpers expose navmesh-to-occupancy conversion and DWA
    local planning for use by higher-level controllers.
    """
    metadata = {
        'render.modes': ['human']
    }

    def __init__(
        self,
        max_time_steps: int,
        env_id: int,
        gym_config: Dict[str, Any],
        warehouse_config: Dict[str, Any],
        observation_config: Dict[str, Any],
    ):
        """Initialize the environment with configuration pulled from config files.

        Args:
            max_time_steps: Maximum simulation steps per episode before auto-termination.
            env_id: Index for multi-env setups when running vectorized environments.
            gym_config: Agent-side gym configuration dict (warehouse-specific sub-dict expected).
            warehouse_config: Unity warehouse environment settings (e.g., chassis limits).
            observation_config: Observation settings (laser, images, etc.).
        Returns:
            None. Sets up action/observation spaces and planner parameters.
        """
        super().__init__()
        self.id = env_id
        self.max_time_steps = int(max_time_steps)
        self.config = gym_config
        self.warehouse_config = warehouse_config
        self.observation_config = observation_config
        self.randomize_spawn_poses = bool(self.config['randomize_spawn_poses'])
        # Per-step caches; navmesh/occupancy reused across steps when no dynamic obstacles are present
        self._navmesh_occ_cache: Dict[Tuple[float, int, float], Optional[OccupancyGridResult]] = {}
        self._step_cache: Dict[str, Any] = {}

        # Planner defaults (Python-side; values supplied by config.py)
        self.navmesh_occ_resolution = float(self.config['navmesh_occupancy_resolution'])
        self.navmesh_occ_padding = int(self.config['navmesh_occupancy_padding_cells'])
        self.navmesh_occ_rotation = float(self.config['navmesh_occupancy_rotation_deg'])

        # NavMesh dynamics: recompute/rasterize every step only when dynamic obstacles carve the mesh
        dyn_cfg = self.warehouse_config.get('dynamic_obstacles') if isinstance(self.warehouse_config, dict) else {}
        dyn_count = dyn_cfg.get('dynamic_obstacle_count', dyn_cfg.get('DynamicObstacleCount', 0)) if isinstance(dyn_cfg, dict) else 0
        enable_obstacle_mgr = self.warehouse_config['enable_obstacle_manager']
        self.enable_transport = self.warehouse_config['enable_transport']
        self.has_dynamic_obstacles = bool((enable_obstacle_mgr and int(dyn_count)) > 0 or self.enable_transport)

        # DWA configuration (all supplied by config.py)
        self.dwa_freq = float(self.config['dwa_freq'])
        self.dwa_lookahead = float(self.config['dwa_lookahead'])
        self.dwa_min_linear_vel = float(self.config['dwa_min_linear_vel'])
        self.dwa_max_linear_vel = float(self.config['dwa_max_linear_vel'])
        self.dwa_min_angular_vel = float(self.config['dwa_min_angular_vel'])
        self.dwa_max_angular_vel = float(self.config['dwa_max_angular_vel'])
        self.dwa_max_acc = float(self.config['dwa_max_acc'])
        self.dwa_max_dec = float(self.config['dwa_max_dec'])
        self.dwa_robot_radius = float(self.config['dwa_robot_radius'])
        self.dwa_safety_distance = float(self.config['dwa_safety_distance'])
        self.dwa_min_dist_goal = float(self.config['dwa_min_dist_goal'])
        self.dwa_res_lin_vel_space = int(self.config['dwa_res_lin_vel_space'])
        self.dwa_res_ang_vel_space = int(self.config['dwa_res_ang_vel_space'])
        self.dwa_gain_glob_path = float(self.config['dwa_gain_glob_path'])
        self.dwa_gain_angle_to_goal = float(self.config['dwa_gain_angle_to_goal'])
        self.dwa_gain_vel = float(self.config['dwa_gain_vel'])
        self.dwa_gain_prox_to_obst = float(self.config['dwa_gain_prox_to_obst'])

        # Action space: [v, omega]
        self.max_linear_velocity = float(self.warehouse_config['max_chassis_linear_speed'])
        self.max_angular_velocity = float(self.warehouse_config['max_chassis_angular_speed'])
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_velocity, -self.max_angular_velocity], dtype=np.float32),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: [x, y, yaw, v, omega, dx, dy, dyaw] + optional laser[N]
        self.use_laser_scan = bool(self.observation_config['enable_laser_scan'])
        laser_cfg = self.observation_config['laser_scan']
        raw_laser_count = int(laser_cfg['num_measurements_per_scan'])
        self.laser_count = raw_laser_count if self.use_laser_scan else 0
        self.laser_max_range = float(laser_cfg['range_meters_max'])
        start_angle_deg = float(laser_cfg['scan_angle_start_degrees'])
        end_angle_deg = float(laser_cfg['scan_angle_end_degrees'])
        start_rad = float(np.deg2rad(start_angle_deg))
        end_rad = float(np.deg2rad(end_angle_deg))
        self.laser_angle_min = min(start_rad, end_rad)
        self.laser_angle_max = max(start_rad, end_rad)
        offset_x, offset_z = self.config['laser_sensor_offset']
        self.laser_sensor_offset = (float(offset_x), float(offset_z))
        self.laser_scan = (
            np.zeros(self.laser_count, dtype=np.float32)
            if self.use_laser_scan and self.laser_count > 0
            else np.zeros(0, dtype=np.float32)
        )
        self.base_observation_dim = 8
        self.observation_dim = self.base_observation_dim + (self.laser_count if self.use_laser_scan else 0)
        obs_high = np.array([np.inf] * self.observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.reward_range = (-2.0, 2.0)

        # Episode state
        self.time_step = 0
        self._last_observation = np.zeros(self.observation_dim, dtype=np.float32)
        self._done = False
        self._rng = np.random.default_rng(seed=0)

        # Per-episode flags
        self.success = False
        self.collision = False
        self.collided_env = 0

        # Normalization/scales and thresholds
        self.normalize_observation = bool(self.config['normalize_obs'])
        self.position_normalization = float(self.config['pos_norm'])
        self.yaw_normalization = float(self.config['yaw_norm'])
        self.success_distance_threshold = float(self.config['success_distance_threshold'])
        self.success_yaw_threshold = float(self.config['success_yaw_threshold'])

        # Unity observation cache
        self.unity_observation = {}
        self._unity_image_bytes_list: List[bytes] = []
        self._navmesh_snapshot: Dict[str, Any] = {}
        self._monitor_global_path: List[Tuple[float, float]] = []
        self._monitor_dwa_traj: List[Tuple[float, float, float]] = []
        self._monitor_p_traj: List[Tuple[float, float, float]] = []
        self._dwa_reorient_done: bool = False

    def seed(self, seed: int = None):
        """Seed the action/observation spaces and internal RNG.

        Args:
            seed: Optional integer seed for reproducibility.
        Returns:
            List containing the seed value for gym compatibility.
        """
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            self._rng = np.random.default_rng(seed=seed)
        return [seed]

    def reset(self):
        """Reset per-episode state and return the initial observation placeholder.

        Returns:
            np.ndarray: Zero observation array matching observation_space shape.
        """
        
        # For RESET command (Unity expects a flat array like IiwaDartUnityEnv):
        # reset_state = active_joints + joint_positions + joint_velocities + target_positions_mapped + object_positions_mapped + robot_positions_mapped
        # Where lengths are: n_joints (active), n_joints (positions), n_joints (velocities), 6 (target pose xyz/rotation), 6 (object pose xyz/rotation), 6 (robot pose xyz/rotation)
        self.n_joints = int(self.config['num_joints'])

        # Active joints: mark all as active by default
        active_joints = [1.0] * self.n_joints

        # Initial joint positions of the AMR [rad or m depending on joint]
        init_pos = [0.0] * self.n_joints

        # Initial joint velocities of the AMR
        init_vel = [0.0] * self.n_joints

        # Convert ROS-style planar poses (x, y, yaw about +z) to Unity coords (x, y, z, rx, ry, rz).
        if self.randomize_spawn_poses:
            target_pose_ros, object_pose_ros, robot_pose_ros = self._sample_spawn_poses()
        else:
            target_pose_ros = list(self.config['target_pose'])
            object_pose_ros = list(self.config['object_pose'])
            robot_pose_ros = list(self.config['robot_pose'])

        object_size = [1.0, 1.0, 1.0]
        items_pool = self.warehouse_config.get('items') if isinstance(self.warehouse_config, dict) else None
        if isinstance(items_pool, (list, tuple)) and items_pool:
            candidate_size = items_pool[0].get('item_size') if isinstance(items_pool[0], dict) else None
            if candidate_size is not None and len(candidate_size) >= 2:
                object_size = candidate_size
        object_height = 0.5 * float(object_size[1]) if len(object_size) >= 2 else 0.5
        ground_surface = 0.5 * self.warehouse_config['ground']['ground_size'][1]

        target_pose_unity = self._ros_planar_pose_to_unity(target_pose_ros, height=ground_surface)
        object_pose_unity = self._ros_planar_pose_to_unity(object_pose_ros, height=ground_surface+object_height)
        robot_pose_unity = self._ros_planar_pose_to_unity(robot_pose_ros, height=ground_surface)

        self.reset_state = [
            *active_joints,
            *init_pos,
            *init_vel,
            *target_pose_unity,
            *object_pose_unity,
            *robot_pose_unity
        ]

        self.time_step = 0
        self._done = False
        self.success = False
        self.collision = False
        self.collided_env = 0
        self._last_observation = np.zeros(self.observation_dim, dtype=np.float32)
        self.unity_observation = {}
        self._navmesh_snapshot = {}
        self._navmesh_occ_cache = {}
        self._step_cache = {}
        self._monitor_global_path = []
        self._monitor_dwa_traj = []
        self._monitor_p_traj = []
        self._dwa_reorient_done = False
        if self.use_laser_scan and self.laser_count > 0:
            zeros = np.zeros(self.laser_count, dtype=np.float32)
            self.unity_observation['laser_scan'] = zeros
            self.laser_scan = zeros
        self._unity_image_bytes_list = []
        return self._last_observation.copy()

    def step(self, action: np.ndarray):
        """Return cached observation, reward, done, and info (VecEnv-compatible).

        Args:
            action: Placeholder; actions are produced by the planner via update().
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: observation array, reward, done flag, info dict with success/collision.
        """
        return self._last_observation.copy(), self._last_reward, self._done, {"success": self.success, "collision": self.collision}

    def update(self, unity_observation_dict: Dict[str, Any], time_step_update: bool = True):
        """Ingest Unity observations, refresh internal state, and compute reward.

        Args:
            unity_observation_dict: Payload from Unity with numeric Observation (required) and optional LaserScan/NavMesh/images.
            time_step_update: Whether to advance the episode timestep counter.
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]: observation array, reward, done flag, info dict with success/collision.
        """
        # Clear per-step caches
        self._step_cache = {}

        image_data = unity_observation_dict.get('ImageData', None)
        if image_data is not None:
            self._unity_retrieve_observation_images(image_data)
        else:
            self._unity_image_bytes_list = []

        navmesh_payload = unity_observation_dict.get('NavMesh', None)
        # Rebuild navmesh snapshot each step only when dynamic obstacles are present; otherwise reuse within an episode
        should_refresh_navmesh = self.has_dynamic_obstacles or self.time_step == 0 or not self._navmesh_snapshot
        if should_refresh_navmesh:
            new_snapshot = self._unity_retrieve_navmesh(navmesh_payload)
            if new_snapshot:
                self._navmesh_snapshot = new_snapshot
                self._navmesh_occ_cache = {}
                self.unity_observation['navmesh'] = self._navmesh_snapshot
            elif 'navmesh' in self.unity_observation:
                # Drop stale navmesh if payload is missing/invalid this step
                self.unity_observation.pop('navmesh', None)

        # Parse numeric observation from Unity
        observation_vector = np.array(unity_observation_dict['Observation'], dtype=np.float32)
        self._unity_retrieve_observation_numeric(observation_vector)

        # Optional obstacle poses
        if 'Obstacles' in unity_observation_dict:
            self.unity_observation['obstacles'] = unity_observation_dict['Obstacles']

        # Build agent observation [x, y, yaw, v, omega, dx, dy, dyaw]
        # Unity ground plane is X/Z; treat Z as the planar "y" axis for the monitor.
        robot_x = float(self.unity_observation['robot_position'][0])
        robot_y = float(self.unity_observation['robot_position'][2])
        robot_yaw = float(self.unity_observation['robot_yaw_ros'])
        robot_linear_velocity = float(self.unity_observation['robot_linear_velocity'])
        robot_angular_velocity = float(self.unity_observation['robot_angular_velocity'])
        target_x = float(self.unity_observation['target_position'][0])
        target_y = float(self.unity_observation['target_position'][2])
        target_yaw = float(self.unity_observation['target_yaw_ros'])

        delta_x = target_x - robot_x
        delta_y = target_y - robot_y
        delta_yaw = self._wrap_angle(target_yaw - robot_yaw)

        observation_base = np.array([robot_x, robot_y, robot_yaw, robot_linear_velocity, robot_angular_velocity,
                                     delta_x, delta_y, delta_yaw], dtype=np.float32)
        if self.normalize_observation:
            observation_base[0] /= self.position_normalization
            observation_base[1] /= self.position_normalization
            observation_base[2] /= self.yaw_normalization
            observation_base[3] /= self.max_linear_velocity
            observation_base[4] /= self.max_angular_velocity
            observation_base[5] /= self.position_normalization
            observation_base[6] /= self.position_normalization
            observation_base[7] /= self.yaw_normalization

        # Laser scan (optional)
        if self.use_laser_scan:
            laser_in = unity_observation_dict.get('LaserScan', None)
            laser_vec = np.array(laser_in if laser_in is not None else [0.0] * self.laser_count, dtype=np.float32)
            if laser_vec.shape[0] != self.laser_count:
                if laser_vec.shape[0] < self.laser_count:
                    laser_vec = np.pad(laser_vec, (0, self.laser_count - laser_vec.shape[0]))
                else:
                    laser_vec = laser_vec[:self.laser_count]
            self._last_observation = np.concatenate([observation_base, laser_vec]).astype(np.float32)
            self.unity_observation['laser_scan'] = laser_vec
            self.laser_scan = laser_vec
        else:
            self._last_observation = observation_base

        # Flags
        self.collision = bool(self.unity_observation['collision_flag'] >= 0.5)
        # self.collided_env = 1 if self.collision else 0

        # Success detection if not explicitly provided
        if not self.success:
            if self.normalize_observation:
                dx_m = float(self._last_observation[5] * self.position_normalization)
                dy_m = float(self._last_observation[6] * self.position_normalization)
                dyaw_r = float(self._last_observation[7] * self.yaw_normalization)
            else:
                dx_m = float(self._last_observation[5])
                dy_m = float(self._last_observation[6])
                dyaw_r = float(self._last_observation[7])
            dist = np.hypot(dx_m, dy_m)
            self.success = bool(dist <= self.success_distance_threshold and abs(dyaw_r) <= self.success_yaw_threshold)

        if time_step_update:
            self.time_step += 1
            # self._done = self.success or self.collision or (self.time_step >= self.max_time_steps)
            self._done = self.time_step >= self.max_time_steps

        reward = self._compute_reward()
        self._last_reward = reward
        return self._last_observation.copy(), self._last_reward, self._done, {"success": self.success, "collision": self.collision}

    def render(self, mode='human'):
        """No-op render stub for gym interface compatibility."""
        return None

    def close(self):
        """No-op cleanup stub for gym interface compatibility."""
        return None

    def get_terminal_reward(self) -> bool:
        """Return success flag for gym wrapper compatibility.

        Returns:
            bool: True if the episode goal was achieved.
        """
        return bool(self.success)

    def random_action(self) -> np.ndarray:
        """Sample a random action from the action space.

        Returns:
            np.ndarray: Random [v, omega] action.
        """
        return self.action_space.sample()

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert normalized [-1, 1] action into physical [m/s, rad/s] range.

        Args:
            action: Normalized action array shaped like [v_norm, w_norm]; values
                outside [-1, 1] are clipped.

        Returns:
            np.ndarray: Physical action [v_mps, w_radps] clipped to chassis limits.
        """
        if action is None:
            return np.zeros(2, dtype=np.float32)
        act = np.asarray(action, dtype=float).flatten()
        v = float(np.clip(act[0], -1.0, 1.0) * self.max_linear_velocity)
        w = float(np.clip(act[1], -1.0, 1.0) * self.max_angular_velocity)
        return np.array([v, w], dtype=np.float32)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Convert physical [m/s, rad/s] action into normalized [-1, 1] range.

        Args:
            action: Physical action array shaped like [v_mps, w_radps]; values
                beyond chassis limits are clipped before scaling.

        Returns:
            np.ndarray: Normalized action [v_norm, w_norm] in [-1, 1].
        """
        if action is None:
            return np.zeros(2, dtype=np.float32)
        act = np.asarray(action, dtype=float).flatten()
        v = float(np.clip(act[0] / self.max_linear_velocity, -1.0, 1.0))
        w = float(np.clip(act[1] / self.max_angular_velocity, -1.0, 1.0))
        return np.array([v, w], dtype=np.float32)

    def build_occupancy_from_navmesh(
        self,
        resolution: float = 0.1,
        padding_cells: int = 1,
        rotation_deg: float = 0.0,
    ) -> Optional[OccupancyGridResult]:
        """Convert cached NavMesh triangles to a 2D occupancy grid.

        Args:
            resolution: Grid cell size in meters.
            padding_cells: Extra padding cells added around bounds.
            rotation_deg: Rotation applied to the rasterization frame.
        Returns:
            OccupancyGridResult if navmesh is available and valid, else None.
        """
        cache_key = (float(resolution), int(padding_cells), float(rotation_deg))
        if cache_key in self._navmesh_occ_cache:
            return self._navmesh_occ_cache[cache_key]
        snapshot = self._navmesh_snapshot or {}
        vertices = snapshot.get('vertices')
        indices = snapshot.get('indices')
        if vertices is None or indices is None:
            return None
        try:
            result = navmesh_to_occupancy_grid(
                vertices=vertices,
                faces=np.reshape(indices, (-1, 3)),
                resolution=resolution,
                padding_cells=padding_cells,
                rotation_deg=rotation_deg,
            )
            self._navmesh_occ_cache[cache_key] = result
            return result
        except Exception:
            return None

    def plan_global_path_astar(
        self,
        resolution: float = 0.1,
        padding_cells: int = 2,
        allow_diagonal: bool = True,
        unknown_is_obstacle: bool = True,
        rotation_deg: float = 0.0,
        cost_weight: float = 3.0,
    ) -> Optional[List[Tuple[float, float]]]:
        """Run A* on the navmesh-derived occupancy grid.

        Args:
            resolution: Grid cell size for rasterization.
            padding_cells: Padding around navmesh bounds in cells.
            allow_diagonal: Whether A* can step diagonally.
            unknown_is_obstacle: Treat unknown cells as obstacles if True.
            rotation_deg: Rotation applied to the occupancy frame.
            cost_weight: Weight for optional costmap if provided.
        Returns:
            List of world-frame waypoints (x, y) or None if no path.
        """
        occ = self.build_occupancy_from_navmesh(
            resolution=resolution,
            padding_cells=padding_cells,
            rotation_deg=rotation_deg,
        )
        if occ is None:
            return None

        robot_x = float(self.unity_observation['robot_position'][0])
        robot_y = float(self.unity_observation['robot_position'][2])
        target_x = float(self.unity_observation['target_position'][0])
        target_y = float(self.unity_observation['target_position'][2])

        def _to_cell(x: float, y: float) -> Tuple[int, int]:
            col = int(np.clip(np.floor((x - occ.origin[0]) / occ.resolution), 0, occ.width - 1))
            row = int(np.clip(np.floor((y - occ.origin[1]) / occ.resolution), 0, occ.height - 1))
            return row, col

        start_cell = _to_cell(robot_x, robot_y)
        goal_cell = _to_cell(target_x, target_y)

        result: Optional[AStarResult] = astar(
            occ.grid,
            start=start_cell,
            goal=goal_cell,
            allow_diagonal=allow_diagonal,
            unknown_is_obstacle=unknown_is_obstacle,
            cost_grid=getattr(occ, "costmap", None),
            cost_weight=cost_weight,
        )
        if result is None or not result.path:
            return None

        def _to_world(cell: Tuple[int, int]) -> Tuple[float, float]:
            r, c = cell
            x = occ.origin[0] + (c + 0.5) * occ.resolution
            y = occ.origin[1] + (r + 0.5) * occ.resolution
            return (x, y)

        return [_to_world(cell) for cell in result.path]

    def _preview_trajectory(
        self,
        pose: Tuple[float, float, float],
        linear: float,
        angular: float,
        horizon: float,
        steps: int = 12,
    ) -> List[Tuple[float, float, float]]:
        """Generate a short rollout for visualization without running full DWA.

        Args:
            pose: (x, y, yaw) start pose.
            linear: Linear velocity to preview.
            angular: Angular velocity to preview.
            horizon: Preview duration in seconds.
            steps: Number of interpolated points.
        Returns:
            List of (x, y, yaw) states along the preview.
        """
        dt = horizon / max(int(steps), 1)
        x, y, yaw = pose
        traj: List[Tuple[float, float, float]] = []
        if abs(angular) < 1e-6:
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            for step in range(1, steps + 1):
                offset = linear * dt * step
                traj.append((x + offset * cos_yaw, y + offset * sin_yaw, yaw))
            return traj

        # If we are turning in place (v≈0), fabricate a small arc so the monitor shows motion.
        if abs(linear) < 1e-6:
            radius = 0.2
        else:
            radius = linear / angular if angular != 0 else 0.0
        for step in range(1, steps + 1):
            yaw_step = yaw + angular * dt * step
            traj.append(
                (
                    x - radius * math.sin(yaw) + radius * math.sin(yaw_step),
                    y + radius * math.cos(yaw) - radius * math.cos(yaw_step),
                    yaw_step,
                )
            )
        return traj

    def compute_dwa_action(
        self,
        global_path_world: Sequence[Tuple[float, float]],
        obstacles_world: Optional[np.ndarray] = None,
        cfg: Optional[DWAConfig] = None,
    ) -> Optional[np.ndarray]:
        """Compute (v, omega) using DWA with a one-time initial reorientation.

        Args:
            global_path_world: Global path waypoints in world frame.
            obstacles_world: Optional obstacle points (x, y) in world frame.
            cfg: Optional override DWAConfig; if None, uses config-driven values.
        Returns:
            np.ndarray [v, omega] in local planner frame, or None if planner failed.
        """
        self._monitor_global_path = [tuple(map(float, p)) for p in global_path_world] if global_path_world else []
        if cfg is None:
            cfg = DWAConfig(
                freq=self.dwa_freq,
                lookahead=self.dwa_lookahead,
                min_linear_vel=self.dwa_min_linear_vel,
                max_linear_vel=self.dwa_max_linear_vel,
                min_angular_vel=self.dwa_min_angular_vel,
                max_angular_vel=self.dwa_max_angular_vel,
                max_acc=self.dwa_max_acc,
                max_dec=self.dwa_max_dec,
                robot_radius=self.dwa_robot_radius,
                safety_distance=self.dwa_safety_distance,
                min_dist_goal=self.dwa_min_dist_goal,
                res_lin_vel_space=self.dwa_res_lin_vel_space,
                res_ang_vel_space=self.dwa_res_ang_vel_space,
                gain_glob_path=self.dwa_gain_glob_path,
                gain_angle_to_goal=self.dwa_gain_angle_to_goal,
                gain_vel=self.dwa_gain_vel,
                gain_prox_to_obst=self.dwa_gain_prox_to_obst,
            )
        planner = DWALocalPlanner(cfg)
        raw_lin = float(self.unity_observation['robot_linear_velocity'])
        raw_ang_unity = float(self.unity_observation['robot_angular_velocity'])
        lin_clip = float(np.clip(raw_lin, -self.max_linear_velocity, self.max_linear_velocity))
        ang_clip_unity = float(np.clip(raw_ang_unity, -self.max_angular_velocity, self.max_angular_velocity))
        ang_clip_local = -ang_clip_unity  # Unity yaw is left-handed; planner uses right-handed
        current_twist = (lin_clip, ang_clip_local)

        yaw_local = float(self.unity_observation['robot_yaw_ros'])
        current_pose = (
            float(self.unity_observation['robot_position'][0]),
            float(self.unity_observation['robot_position'][2]),
            yaw_local,
        )

        path_arr = np.asarray(global_path_world, dtype=float)
        if path_arr.size:
            # Stage 1: one-time reorientation to the initial path segment, then permanently hand off to DWA
            if not self._dwa_reorient_done and path_arr.shape[0] >= 2:
                init_heading = math.atan2(path_arr[1, 1] - path_arr[0, 1], path_arr[1, 0] - path_arr[0, 0])
                heading_err_init = self._wrap_angle(init_heading - current_pose[2])
                turn_exit = 0.3
                if abs(heading_err_init) > turn_exit:
                    w_init = float(np.clip(heading_err_init * 1.5, cfg.min_angular_vel, cfg.max_angular_vel))
                    self._monitor_dwa_traj = [current_pose] + self._preview_trajectory(current_pose, 0.0, w_init, horizon=cfg.lookahead)
                    return np.array([0.0, w_init], dtype=np.float32)
                self._dwa_reorient_done = True

            goal_wp = path_arr[-1]

            dist_to_goal = float(np.hypot(goal_wp[0] - current_pose[0], goal_wp[1] - current_pose[1]))
            # If we are essentially at the goal position, prioritize aligning to the target yaw
            if dist_to_goal < 0.12:
                target_yaw_local = float(self.unity_observation['target_yaw_ros'])
                yaw_err_to_target = self._wrap_angle(target_yaw_local - current_pose[2])
                if abs(yaw_err_to_target) > 0.05:
                    w_align_local = float(np.clip(yaw_err_to_target * 1.5, cfg.min_angular_vel, cfg.max_angular_vel))
                    self._monitor_dwa_traj = [current_pose] + self._preview_trajectory(current_pose, 0.0, w_align_local, horizon=cfg.lookahead)
                    return np.array([0.0, w_align_local], dtype=np.float32)
                # If heading is already aligned, stop
                self._monitor_dwa_traj = []
                return np.array([0.0, 0.0], dtype=np.float32)
        else:
            goal_wp = (None, None)

        obs_array = obstacles_world if obstacles_world is not None else np.zeros((0, 2), dtype=float)
        result: Optional[DWAResult] = planner.run(
            current_twist=current_twist,
            current_pose=current_pose,
            global_path=global_path_world,
            obstacles=obs_array,
            force_follow_plan=True,
        )
        if result is None:
            self._monitor_dwa_traj = []
            return None

        # Stage 2: pure DWA tracking once initial reorientation is done
        self._monitor_dwa_traj = [current_pose] + list(result.trajectory)
        return np.array([result.linear_vel, result.angular_vel], dtype=np.float32)

    def action_by_p_control(self, k_v: float = 1.0, k_w: float = 1.0) -> np.ndarray:
        # 
        """Prefer planner-based control (A* global path + DWA local tracking).
        Fallbacks to P-controller if planning fails.

        Args:
            k_v: Proportional gain of P-controller for linear velocity toward goal.
            k_w: Proportional gain of P-controller for angular velocity toward goal.
        Returns:
            Action array [v, omega] clipped to action space.
        """
        global_path = self.plan_global_path_astar(
            resolution=self.navmesh_occ_resolution,
            padding_cells=self.navmesh_occ_padding,
            allow_diagonal=True,
            unknown_is_obstacle=False,
            rotation_deg=self.navmesh_occ_rotation,
        )
        self._monitor_global_path = global_path or []
        self._monitor_dwa_traj = []
        if global_path is not None and len(global_path) >= 2:
            planner_action = self.compute_dwa_action(global_path_world=global_path)
            if planner_action is not None:
                clipped = np.clip(planner_action, -self.action_space.high, self.action_space.high)
                self._monitor_last_action = clipped.astype(np.float32)
                self._monitor_p_traj = []
                return self._monitor_last_action
        else:
            self._monitor_global_path = []
            self._monitor_dwa_traj = []
        self._monitor_p_traj = []

        # Fallback: proportional controller towards goal delta
        dx = float(self._last_observation[5])
        dy = float(self._last_observation[6])
        robot_pos = self.unity_observation['robot_position']
        rx = float(robot_pos[0])
        ry = float(robot_pos[2])
        robot_yaw = float(self.unity_observation['robot_yaw_ros'])
        goal_heading = float(np.arctan2(dy, dx))
        heading_error = self._wrap_angle(goal_heading - robot_yaw)
        if self.normalize_observation:
            dx *= self.position_normalization
            dy *= self.position_normalization
            heading_error *= 1.0  # already wrapped; keep scale

        distance = float(np.hypot(dx, dy))
        v_cmd = k_v * distance * float(np.cos(heading_error))
        if abs(heading_error) > np.pi / 2:
            v_cmd = 0.0
        w_cmd = k_w * heading_error
        action = np.array([v_cmd, w_cmd], dtype=np.float32)
        action = np.clip(action, -self.action_space.high, self.action_space.high)
        self._monitor_last_action = action
        current_pose = (rx, ry, robot_yaw)
        self._monitor_p_traj = [current_pose] + self._preview_trajectory(current_pose, float(action[0]), float(action[1]), horizon=1.0)
        return action

    def _compute_reward(self) -> float:
        """Compute shaped reward using distance, yaw error, success, and collision.

        Returns:
            float: Shaped reward for the current timestep.
        """
        step_penalty = float(self.config['step_penalty'])
        success_reward = float(self.config['success_reward'])
        collision_penalty = float(self.config['collision_penalty'])
        distance_weight = float(self.config['distance_weight'])
        yaw_weight = float(self.config['yaw_weight'])

        # Use metric units for shaping
        if self.normalize_observation:
            dx_m = float(self._last_observation[5] * self.position_normalization)
            dy_m = float(self._last_observation[6] * self.position_normalization)
            dyaw_r = float(self._last_observation[7] * self.yaw_normalization)
        else:
            dx_m = float(self._last_observation[5])
            dy_m = float(self._last_observation[6])
            dyaw_r = float(self._last_observation[7])

        dist = np.hypot(dx_m, dy_m)
        shaped = -step_penalty - distance_weight * dist - yaw_weight * abs(dyaw_r)
        if self.success:
            shaped += success_reward
        if self.collision:
            shaped -= collision_penalty
        return float(shaped)

    def _sample_spawn_poses(self) -> Tuple[List[float], List[float], List[float]]:
        """Generate non-overlapping planar poses for target, object, and robot within ground bounds.

        Returns:
            Tuple of three [x, y, yaw] lists corresponding to (target, object, robot).
        """
        ground_cfg = self.warehouse_config.get('ground', {}) if isinstance(self.warehouse_config, dict) else {}
        ground_size = ground_cfg.get('ground_size', [12.5, 0.1, 7.5])
        margin = float(self.config['spawn_min_separation']) * 0.5
        half_x = float(ground_size[0]) * 0.5 - margin
        half_y = float(ground_size[2]) * 0.5 - margin
        min_sep = float(self.config['spawn_min_separation'])
        max_attempts = 100
        poses: List[List[float]] = []

        def _valid(candidate: Tuple[float, float]) -> bool:
            for px, py, _ in poses:
                if math.hypot(candidate[0] - px, candidate[1] - py) < min_sep:
                    return False
            return True

        attempts = 0
        while len(poses) < 3 and attempts < max_attempts:
            x = float(self._rng.uniform(-half_x, half_x))
            y = float(self._rng.uniform(-half_y, half_y))
            yaw = float(self._rng.uniform(-math.pi, math.pi))
            if _valid((x, y)):
                poses.append([x, y, yaw])
            attempts += 1

        if len(poses) < 3:
            return (
                list(self.config['target_pose']),
                list(self.config['object_pose']),
                list(self.config['robot_pose']),
            )

        return poses[0], poses[1], poses[2]

    @staticmethod
    def _ros_planar_pose_to_unity(pose_xy_yaw: Sequence[float], height: float) -> List[float]:
        """Map ROS-style planar pose (x, y, yaw about +z) to Unity pose [x,y,z,rx,ry,rz].

        Args:
            pose_xy_yaw: Iterable containing x, y, and yaw in the ROS frame.
            height: Desired Unity height coordinate for the mapped pose.
        Returns:
            List[float]: Unity pose with roll/pitch zeroed and yaw converted.
        """
        try:
            x_ros = float(pose_xy_yaw[0])
            y_ros = float(pose_xy_yaw[1])
            yaw_ros = float(pose_xy_yaw[2])
        except Exception:
            x_ros = 0.0
            y_ros = 0.0
            yaw_ros = 0.0

        # Inverse of _unity_yaw_to_planar: planar_yaw = -yaw_unity + pi/2
        yaw_unity_rad = 0.5 * np.pi - yaw_ros
        yaw_unity_deg = float(np.rad2deg(yaw_unity_rad))
        return [x_ros, float(height), y_ros, 0.0, yaw_unity_deg, 0.0]

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Wrap an input angle to the principal interval [-pi, pi).

        Args:
            a: Angle in radians to normalize.
        Returns:
            float: Wrapped angle in [-pi, pi).
        """
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _unity_yaw_to_planar(yaw_unity: float) -> float:
        """Convert Unity yaw (left-handed, 0 along +Z) to planar yaw (right-handed, 0 along +X).

        Args:
            yaw_unity: Heading reported by Unity in radians.
        Returns:
            float: Planar yaw in a right-handed frame.
        """
        yaw_right_handed = -yaw_unity
        yaw_shifted = yaw_right_handed + 0.5 * np.pi
        return WarehouseUnityEnv._wrap_angle(yaw_shifted)

    def _build_laser_dataclasses(self) -> Tuple[Optional[LaserScanData], Optional[LaserPoints]]:
        """Build laser scan and projected point dataclasses from cached observation.

        Returns:
            Tuple[Optional[LaserScanData], Optional[LaserPoints]]: Parsed laser scan and projected points or (None, None) if disabled.
        """
        if 'laser_dataclasses' in self._step_cache:
            return self._step_cache['laser_dataclasses']
        if not self.use_laser_scan or self.laser_count <= 0:
            return None, None

        ranges = np.asarray(self.unity_observation.get('laser_scan', self.laser_scan), dtype=float).flatten()
        if ranges.size != self.laser_count:
            if ranges.size < self.laser_count:
                ranges = np.pad(ranges, (0, self.laser_count - ranges.size), constant_values=0.0)
            else:
                ranges = ranges[: self.laser_count]

        scan = LaserScanData(
            ranges=ranges,
            angle_min=float(self.laser_angle_min),
            angle_max=float(self.laser_angle_max),
            max_range=float(self.laser_max_range),
            sensor_offset_xy=tuple(self.laser_sensor_offset),
            frame_id="laser",
        )

        # Apply handedness correction (- (min+max)) before projecting points.
        angle_offset = - (scan.angle_min + scan.angle_max)
        angles = np.linspace(scan.angle_min, scan.angle_max, ranges.size, dtype=float) + angle_offset
        yaw = float(self.unity_observation['robot_yaw_ros'])
        theta = yaw + angles

        ox, oy = scan.sensor_offset_xy
        offset_world_x = ox * np.cos(yaw) - oy * np.sin(yaw)
        offset_world_y = ox * np.sin(yaw) + oy * np.cos(yaw)

        dir_x = np.cos(theta)
        dir_y = np.sin(theta)
        robot_pos = self.unity_observation['robot_position']
        robot_x = float(robot_pos[0]) if robot_pos.size >= 1 else 0.0
        robot_y = float(robot_pos[2]) if robot_pos.size >= 3 else 0.0
        sensor_x = robot_x + offset_world_x
        sensor_y = robot_y + offset_world_y

        xs = ranges * dir_x + sensor_x
        ys = ranges * dir_y + sensor_y
        points = np.stack([xs, ys], axis=1) if ranges.size else np.zeros((0, 2), dtype=float)
        laser_points = LaserPoints(points_xy=points, origin_xy=(sensor_x, sensor_y), angle_offset_applied=angle_offset)
        self._step_cache['laser_dataclasses'] = (scan, laser_points)
        return scan, laser_points

    def _navmesh_dataclass(self) -> Optional[NavmeshData]:
        """Convert cached navmesh snapshot into a NavmeshData instance.

        Returns:
            NavmeshData or None if snapshot is unavailable or incomplete.
        """
        if 'navmesh_dc' in self._step_cache:
            return self._step_cache['navmesh_dc']
        if not self._navmesh_snapshot:
            return None
        verts = self._navmesh_snapshot.get('vertices')
        tris = self._navmesh_snapshot.get('indices')
        if verts is None or tris is None:
            return None
        navmesh_dc = NavmeshData(vertices=np.asarray(verts, dtype=float), triangles=np.asarray(tris, dtype=int))
        self._step_cache['navmesh_dc'] = navmesh_dc
        return navmesh_dc

    def _occupancy_dataclass(self) -> Tuple[Optional[OccupancyGridData], Optional[CostmapData]]:
        """Rasterize navmesh into occupancy and optional costmap dataclasses.

        Returns:
            Tuple[Optional[OccupancyGridData], Optional[CostmapData]]: Occupancy grid and costmap (if present) or (None, None).
        """
        if 'occupancy_dc' in self._step_cache:
            return self._step_cache['occupancy_dc']
        occ_result: Optional[OccupancyGridResult] = self.build_occupancy_from_navmesh(
            resolution=self.navmesh_occ_resolution,
            padding_cells=self.navmesh_occ_padding,
            rotation_deg=self.navmesh_occ_rotation,
        )
        if occ_result is None:
            return None, None
        occ = OccupancyGridData(
            grid=np.asarray(occ_result.grid, dtype=float),
            resolution=float(occ_result.resolution),
            origin_xy=tuple(occ_result.origin),
        )
        cost = None
        if getattr(occ_result, 'costmap', None) is not None:
            cost = CostmapData(
                grid=np.asarray(occ_result.costmap, dtype=float),
                resolution=float(occ_result.resolution),
                origin_xy=tuple(occ_result.origin),
            )
        self._step_cache['occupancy_dc'] = (occ, cost)
        return occ, cost

    def get_monitor_payload(self) -> MonitorPayload:
        """Assemble the structured telemetry payload for the task monitor.

        Returns:
            MonitorPayload: All fields required by the monitor (poses, velocities, scans, maps, planner traces, agent state).
        """
        # Poses
        robot_pos = self.unity_observation['robot_position']
        robot_x = float(robot_pos[0]) if robot_pos.size >= 1 else 0.0
        robot_y = float(robot_pos[2]) if robot_pos.size >= 3 else 0.0
        robot_yaw = float(self.unity_observation['robot_yaw_ros'])
        robot_pose = Pose2D(robot_x, robot_y, robot_yaw)

        target_pos_raw = self.unity_observation['target_position']
        target_pos = None
        if target_pos_raw is not None and len(target_pos_raw) >= 3:
            target_pos = (float(target_pos_raw[0]), float(target_pos_raw[2]))

        target_delta = self._last_observation[5:8] if self._last_observation.size >= 8 else [0.0, 0.0, 0.0]
        if self.normalize_observation:
            position_scale = float(self.position_normalization)
            yaw_scale = float(self.yaw_normalization)
            target_delta = (
                float(target_delta[0] * position_scale),
                float(target_delta[1] * position_scale),
                float(target_delta[2] * yaw_scale),
            )
        else:
            target_delta = (float(target_delta[0]), float(target_delta[1]), float(target_delta[2]))

        # Velocities
        linear_velocity = float(self.unity_observation.get('robot_linear_velocity', self._last_observation[3]))
        angular_velocity = float(self.unity_observation.get('robot_angular_velocity', self._last_observation[4]))

        laser_scan_dc, laser_points_dc = self._build_laser_dataclasses()

        occ_dc, cost_dc = self._occupancy_dataclass()
        navmesh_dc = self._navmesh_dataclass()

        planner_paths = PlannerPaths(
            global_path=[(float(p[0]), float(p[1])) for p in self._monitor_global_path],
            dwa_traj=[(float(p[0]), float(p[1]), float(p[2])) for p in self._monitor_dwa_traj],
            p_traj=[(float(p[0]), float(p[1]), float(p[2])) for p in self._monitor_p_traj],
        )
        planner_dc = PlannerData(
            paths=planner_paths,
            occupancy=occ_dc,
            costmap=cost_dc,
            laser_points=laser_points_dc,
            robot_pose=robot_pose,
            target_pos=target_pos,
        )

        return MonitorPayload(
            robot_pose=robot_pose,
            robot_velocity=(linear_velocity, angular_velocity),
            target_delta=target_delta,
            reward=float(getattr(self, '_last_reward', 0.0)),
            success=bool(self.success),
            collision=bool(self.collision),
            laser_scan=laser_scan_dc,
            laser_points=laser_points_dc,
            occupancy=occ_dc,
            costmap=cost_dc,
            navmesh=navmesh_dc,
            planner=planner_dc,
            target_pos=target_pos,
            agent_state=self._last_observation.copy(),
            agent_action=getattr(self, '_monitor_last_action', None),
            agent_reward=np.array([getattr(self, '_last_reward', 0.0)], dtype=float),
            agent_image=self.get_monitor_image(),
        )

    @staticmethod
    def _yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
        """Extract yaw (rotation about Y) from a w,x,y,z quaternion (Unity convention).

        Args:
            w: Quaternion scalar component.
            x: Quaternion x component.
            y: Quaternion y component.
            z: Quaternion z component.
        Returns:
            float: Yaw angle in radians.
        """
        # Standard yaw around Y-axis (roll=X, pitch=Z). Unity stores quaternions as (w,x,y,z).
        # Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _unity_retrieve_observation_numeric(self, observation_vector_unity: np.ndarray):
        """Parse Unity's Observation vector into structured fields.

        Args:
            observation_vector_unity: Flat float array from Unity.
        Returns:
            None. Populates self.unity_observation with robot/target/item poses, velocities, yaw, and collision flag.
        """
        if observation_vector_unity.size < 24:
            self.unity_observation = {
                'robot_position': np.zeros(3, dtype=np.float32),
                'robot_orientation_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'robot_linear_velocity': 0.0,
                'robot_angular_velocity': 0.0,
                'target_position': np.zeros(3, dtype=np.float32),
                'target_orientation_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'item_position': np.zeros(3, dtype=np.float32),
                'item_orientation_quat': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                'collision_flag': 0.0,
                'robot_yaw': 0.0,
                'target_yaw': 0.0,
                'item_yaw': 0.0,
                'robot_yaw_ros': self._unity_yaw_to_planar(0.0),
                'target_yaw_ros': self._unity_yaw_to_planar(0.0),
                'item_yaw_ros': self._unity_yaw_to_planar(0.0),
            }
            return
        base_idx = int(observation_vector_unity.size - 24)
        self.unity_observation['robot_position'] = observation_vector_unity[base_idx + 0: base_idx + 3].astype(np.float32)
        self.unity_observation['robot_orientation_quat'] = observation_vector_unity[base_idx + 3: base_idx + 7].astype(np.float32)
        self.unity_observation['robot_linear_velocity'] = float(observation_vector_unity[base_idx + 7])
        self.unity_observation['robot_angular_velocity'] = float(observation_vector_unity[base_idx + 8])
        self.unity_observation['target_position'] = observation_vector_unity[base_idx + 9: base_idx + 12].astype(np.float32)
        self.unity_observation['target_orientation_quat'] = observation_vector_unity[base_idx + 12: base_idx + 16].astype(np.float32)
        self.unity_observation['item_position'] = observation_vector_unity[base_idx + 16: base_idx + 19].astype(np.float32)
        self.unity_observation['item_orientation_quat'] = observation_vector_unity[base_idx + 19: base_idx + 23].astype(np.float32)
        self.unity_observation['collision_flag'] = float(observation_vector_unity[base_idx + 23])
        robot_yaw_unity = self._yaw_from_wxyz(*self.unity_observation['robot_orientation_quat'])
        target_yaw_unity = self._yaw_from_wxyz(*self.unity_observation['target_orientation_quat'])
        item_yaw_unity = self._yaw_from_wxyz(*self.unity_observation['item_orientation_quat'])
        self.unity_observation['robot_yaw'] = robot_yaw_unity
        self.unity_observation['target_yaw'] = target_yaw_unity
        self.unity_observation['item_yaw'] = item_yaw_unity
        self.unity_observation['robot_yaw_ros'] = self._unity_yaw_to_planar(robot_yaw_unity)
        self.unity_observation['target_yaw_ros'] = self._unity_yaw_to_planar(target_yaw_unity)
        self.unity_observation['item_yaw_ros'] = self._unity_yaw_to_planar(item_yaw_unity)

    def _unity_retrieve_navmesh(self, payload: Any) -> Dict[str, Any]:
        """Parse Unity NavMesh snapshot into vertices/indices for 2D occupancy.

        Args:
            payload: Dict from Unity containing vertices/indices fields.
        Returns:
            Dict with 'vertices' (N x 2) and 'indices' (flattened tris) or empty dict on failure.
        """
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            return {}

        raw_vertices = payload.get('Vertices', None)
        raw_indices = payload.get('Indices', None)

        if raw_vertices is None or raw_indices is None:
            return {}

        try:
            vertices = np.asarray(raw_vertices, dtype=np.float32)
            indices = np.asarray(raw_indices, dtype=np.int32).flatten()
        except Exception:
            return {}

        if vertices.size == 0 or indices.size == 0 or indices.size % 3 != 0:
            return {}

        if vertices.size % 3 == 0:
            vertices = vertices.reshape(-1, 3)
        elif vertices.size % 2 == 0:
            vertices = vertices.reshape(-1, 2)
        else:
            return {}

        return {
            'vertices': vertices,
            'indices': indices,
        }

    def _unity_retrieve_observation_images(self, raw_image_payload: Any) -> List[bytes]:
        """Decode Unity camera payload (base64 strings) into raw image bytes.

        Args:
            raw_image_payload: List/bytes from Unity image stream.
        Returns:
            List of decoded image byte arrays.
        """

        frames: List[bytes] = []

        def _append_candidate(candidate: Any) -> None:
            if candidate in (None, "", b""):
                return
            try:
                base64_bytes = candidate if isinstance(candidate, bytes) else str(candidate).encode('ascii')
            except Exception:
                return
            try:
                decoded = base64.b64decode(base64_bytes)
            except Exception:
                return
            if decoded:
                frames.append(decoded)

        candidates: List[Any] = []
        if isinstance(raw_image_payload, (list, tuple, set)):
            candidates.extend(raw_image_payload)
        else:
            candidates.append(raw_image_payload)

        for candidate in candidates:
            _append_candidate(candidate)

        self._unity_image_bytes_list = frames
        return frames

    def get_wrapper_attr(self, name):
        """Mimic Stable-Baselines3 VecEnvWrapper attribute passthrough.

        Args:
            name: Attribute name to retrieve from the environment.
        Returns:
            Any: Attribute value if present.
        Raises:
            AttributeError: If the attribute is missing.
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_monitor_image(self):
        """Return last captured monitor images, if any.

        Returns:
            Optional[List[bytes]]: Copy of decoded image byte arrays or None if unavailable.
        """
        if not self._unity_image_bytes_list:
            return None
        return list(self._unity_image_bytes_list)
