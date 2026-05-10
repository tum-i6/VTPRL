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
from utils.dwb_local_planner import DWBLocalPlanner, DWBConfig, DWBResult
from utils.config_utils import expand_segmented_entries


class WarehouseUnityEnv(gym.Env):
    """Gym-compatible warehouse navigation environment backed by Unity.

    The environment exposes continuous actions per robot and constructs observations
    from robot, target, laser, and navmesh payloads.
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
    ) -> None:
        """Initialize environment state, planner parameters, and Gym spaces.

        Args:
            max_time_steps: Maximum number of steps per episode.
            env_id: Environment index for vectorized execution.
            gym_config: Agent-side runtime configuration dictionary.
            warehouse_config: Unity warehouse configuration dictionary.
            observation_config: Observation configuration dictionary.
        """
        super().__init__()
        self.id: int = env_id
        self.max_time_steps: int = int(max_time_steps)
        self.config: Dict[str, Any] = gym_config
        self.warehouse_config: Dict[str, Any] = warehouse_config
        self.observation_config: Dict[str, Any] = observation_config
        self.randomize_spawn_poses: bool = bool(self.config['randomize_spawn_poses'])

        self.amr_segments: List[Dict[str, Any]] = self.warehouse_config['amrs']
        self.amr_configs: List[Dict[str, Any]] = expand_segmented_entries(self.amr_segments, default_count=1)
        self.amr_count: int = len(self.amr_configs)
        if self.amr_count <= 0:
            raise ValueError("warehouse_config['amrs'] must define at least one AMR via segment counts.")
        # Keep explicit robot_count alias for VecEnv action-shape checks.
        self.robot_count: int = self.amr_count

        self.item_segments: List[Dict[str, Any]] = self.warehouse_config['items']
        self.item_configs: List[Dict[str, Any]] = expand_segmented_entries(self.item_segments, default_count=1)
        self.item_count: int = len(self.item_configs)
        if self.item_count < 0:
            raise ValueError("warehouse_config['items'] resolved count must be >= 0.")

        # Per-step caches
        self._step_cache: Dict[str, Any] = {}

        # Planner defaults (Python-side; values supplied by config.py)
        self.navmesh_occ_resolution: float = float(self.config['navmesh_occupancy_resolution'])
        self.navmesh_occ_padding: int = int(self.config['navmesh_occupancy_padding_cells'])
        self.navmesh_occ_rotation: float = float(self.config['navmesh_occupancy_rotation_deg'])

        # NavMesh dynamics: recompute/rasterize every step only when dynamic obstacles carve the mesh
        dyn_segments = self.warehouse_config['dynamic_obstacles']
        self.dynamic_obstacle_configs: List[Dict[str, Any]] = expand_segmented_entries(dyn_segments, default_count=0)
        dyn_count = len(self.dynamic_obstacle_configs)
        enable_obstacle_mgr = self.warehouse_config['enable_obstacle_manager']
        self.enable_transport: bool = any(bool(amr['enable_transport']) for amr in self.amr_configs)
        self.has_dynamic_obstacles: bool = bool((enable_obstacle_mgr and int(dyn_count)) > 0 or self.enable_transport)

        # Local controller selection: 'DWA' (ROS1) or 'DWB' (Nav2 critic-based)
        self.local_controller_type: str = str(self.config['local_controller_type']).upper()

        # DWA configuration (all supplied by config.py)
        self.dwa_lookahead: float = float(self.config['dwa_lookahead'])
        self.dwa_min_linear_vel: float = float(self.config['dwa_min_linear_vel'])
        self.dwa_max_linear_vel: float = float(self.config['dwa_max_linear_vel'])
        self.dwa_min_angular_vel: float = float(self.config['dwa_min_angular_vel'])
        self.dwa_max_angular_vel: float = float(self.config['dwa_max_angular_vel'])
        self.dwa_max_acc: float = float(self.config['dwa_max_acc'])
        self.dwa_max_dec: float = float(self.config['dwa_max_dec'])
        self.dwa_robot_radius: float = float(self.config['dwa_robot_radius'])
        self.dwa_safety_distance: float = float(self.config['dwa_safety_distance'])
        self.dwa_min_dist_goal: float = float(self.config['dwa_min_dist_goal'])
        self.dwa_res_lin_vel_space: int = int(self.config['dwa_res_lin_vel_space'])
        self.dwa_res_ang_vel_space: int = int(self.config['dwa_res_ang_vel_space'])
        self.dwa_gain_glob_path: float = float(self.config['dwa_gain_glob_path'])
        self.dwa_gain_angle_to_goal: float = float(self.config['dwa_gain_angle_to_goal'])
        self.dwa_gain_vel: float = float(self.config['dwa_gain_vel'])
        self.dwa_gain_prox_to_obst: float = float(self.config['dwa_gain_prox_to_obst'])

        # DWB configuration (Nav2 critic-based controller, all supplied by config.py)
        self.dwb_lookahead: float = float(self.config['dwb_lookahead'])
        self.dwb_min_linear_vel: float = float(self.config['dwb_min_linear_vel'])
        self.dwb_max_linear_vel: float = float(self.config['dwb_max_linear_vel'])
        self.dwb_min_angular_vel: float = float(self.config['dwb_min_angular_vel'])
        self.dwb_max_angular_vel: float = float(self.config['dwb_max_angular_vel'])
        self.dwb_max_acc: float = float(self.config['dwb_max_acc'])
        self.dwb_max_dec: float = float(self.config['dwb_max_dec'])
        self.dwb_max_ang_acc: float = float(self.config['dwb_max_ang_acc'])
        self.dwb_robot_radius: float = float(self.config['dwb_robot_radius'])
        self.dwb_safety_distance: float = float(self.config['dwb_safety_distance'])
        self.dwb_min_dist_goal: float = float(self.config['dwb_min_dist_goal'])
        self.dwb_yaw_goal_tolerance: float = float(self.config['dwb_yaw_goal_tolerance'])
        self.dwb_res_lin_vel_space: int = int(self.config['dwb_res_lin_vel_space'])
        self.dwb_res_ang_vel_space: int = int(self.config['dwb_res_ang_vel_space'])
        self.dwb_oscillation_reset_dist: float = float(self.config['dwb_oscillation_reset_dist'])
        self.dwb_oscillation_reset_angle: float = float(self.config['dwb_oscillation_reset_angle'])
        self.dwb_scale_path_dist: float = float(self.config['dwb_scale_path_dist'])
        self.dwb_scale_goal_dist: float = float(self.config['dwb_scale_goal_dist'])
        self.dwb_scale_path_align: float = float(self.config['dwb_scale_path_align'])
        self.dwb_scale_goal_align: float = float(self.config['dwb_scale_goal_align'])
        self.dwb_scale_obstacle: float = float(self.config['dwb_scale_obstacle'])
        self.dwb_scale_prefer_forward: float = float(self.config['dwb_scale_prefer_forward'])
        self.dwb_scale_rotate_to_goal: float = float(self.config['dwb_scale_rotate_to_goal'])

        # Global planner obstacle inflation (configuration-space planning)
        # Use a circumscribed footprint radius derived from chassis width/length.
        self.astar_robot_chassis_width: float = float(self.config['astar_robot_chassis_width'])
        self.astar_robot_chassis_length: float = float(self.config['astar_robot_chassis_length'])
        self.astar_obstacle_clearance: float = float(self.config['astar_obstacle_clearance'])
        self.astar_robot_footprint_radius: float = 0.5 * float(
            np.hypot(self.astar_robot_chassis_width, self.astar_robot_chassis_length)
        )
        self.astar_obstacle_inflation_radius: float = max(
            0.0,
            self.astar_robot_footprint_radius + self.astar_obstacle_clearance,
        )
        self.astar_obstacle_inflation_cells: float = self.astar_obstacle_inflation_radius / max(self.navmesh_occ_resolution, 1e-6)
        self.astar_max_projection_distance_cells: int = max(
            0,
            int(self.config['max_projection_distance_cells'])
        )

        # Action space: [v, omega] with per-robot limits
        self.max_linear_velocities: np.ndarray = np.array(
            [float(cfg['max_chassis_linear_speed']) for cfg in self.amr_configs],
            dtype=np.float32,
        )
        self.max_angular_velocities: np.ndarray = np.array(
            [float(cfg['max_chassis_angular_speed']) for cfg in self.amr_configs],
            dtype=np.float32,
        )
        self.max_linear_velocity: float = float(np.max(self.max_linear_velocities))
        self.max_angular_velocity: float = float(np.max(self.max_angular_velocities))
        low_actions = np.stack([-self.max_linear_velocities, -self.max_angular_velocities], axis=1)
        high_actions = np.stack([self.max_linear_velocities, self.max_angular_velocities], axis=1)
        self.action_space = spaces.Box(
            low=low_actions,
            high=high_actions,
            dtype=np.float32,
        )

        # Observation space: [x, y, yaw, v, omega, dx, dy, dyaw] + optional laser[N], derived per AMR
        self.laser_enabled_per_robot: List[bool] = []
        self.laser_counts_per_robot: List[int] = []
        self.laser_max_ranges_per_robot: List[float] = []
        self.laser_angle_min_per_robot: List[float] = []
        self.laser_angle_max_per_robot: List[float] = []
        self.laser_sensor_offsets_per_robot: List[Tuple[float, float]] = []
        for amr_cfg in self.amr_configs:
            enabled = bool(amr_cfg['enable_laser_scan'])
            laser_cfg = amr_cfg['laser_scan']
            count = int(laser_cfg['num_measurements_per_scan']) if enabled else 0
            range_max = float(laser_cfg['range_meters_max']) if enabled else 0.0
            start_deg = float(laser_cfg['scan_angle_start_degrees']) if enabled else 0.0
            end_deg = float(laser_cfg['scan_angle_end_degrees']) if enabled else 0.0
            start_rad = float(np.deg2rad(start_deg))
            end_rad = float(np.deg2rad(end_deg))
            offset_x = float(laser_cfg['sensor_offset_x']) if enabled else 0.0
            offset_y = float(laser_cfg['sensor_offset_y']) if enabled else 0.0

            self.laser_enabled_per_robot.append(enabled)
            self.laser_counts_per_robot.append(max(count, 0))
            self.laser_max_ranges_per_robot.append(range_max)
            self.laser_angle_min_per_robot.append(min(start_rad, end_rad))
            self.laser_angle_max_per_robot.append(max(start_rad, end_rad))
            self.laser_sensor_offsets_per_robot.append((offset_x, offset_y))

        self.use_laser_scan: bool = any(self.laser_enabled_per_robot)
        self.laser_count: int = max(self.laser_counts_per_robot) if self.use_laser_scan else 0
        self.laser_max_range: float = max(self.laser_max_ranges_per_robot) if self.use_laser_scan else 0.0
        self.laser_angle_min: float = min(self.laser_angle_min_per_robot) if self.use_laser_scan else 0.0
        self.laser_angle_max: float = max(self.laser_angle_max_per_robot) if self.use_laser_scan else 0.0
        self.laser_sensor_offset: Tuple[float, float] = self.laser_sensor_offsets_per_robot[0] if self.use_laser_scan else (0.0, 0.0)
        self.laser_scan: np.ndarray = (
            np.zeros(self.laser_count, dtype=np.float32)
            if self.use_laser_scan and self.laser_count > 0
            else np.zeros(0, dtype=np.float32)
        )
        self.base_observation_dim: int = 8
        self.single_observation_dim: int = self.base_observation_dim + (self.laser_count if self.use_laser_scan else 0)
        self.observation_dim: int = self.single_observation_dim * self.amr_count
        obs_high = np.array([np.inf] * self.observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self.reward_range: Tuple[float, float] = (-2.0, 2.0)

        # Episode state
        self.time_step: int = 0
        self._last_observation: np.ndarray = np.zeros(self.observation_dim, dtype=np.float32)
        self._last_reward: float = 0.0
        self._done: bool = False
        self._rng: np.random.Generator = np.random.default_rng(seed=0)

        # Per-episode flags
        self.success: bool = False
        self.collision: bool = False
        self.collided_env: int = 0
        self._last_rewards: np.ndarray = np.zeros(self.amr_count, dtype=np.float32)

        # Normalization/scales and thresholds
        self.normalize_observation: bool = bool(self.config['normalize_obs'])
        self.position_normalization: float = float(self.config['pos_norm'])
        self.yaw_normalization: float = float(self.config['yaw_norm'])
        self.success_distance_threshold: float = float(self.config['success_distance_threshold'])
        self.success_yaw_threshold: float = float(self.config['success_yaw_threshold'])

        # Unity observation cache
        self.unity_observation: Dict[str, Any] = {}
        self._unity_image_bytes_list: List[bytes] = []
        self._unity_image_labels: List[str] = []
        self._unity_robot_payloads: Optional[Any] = None
        self._per_robot_monitor_cache: List[Dict[str, Any]] = []
        self._monitor_last_action: Optional[np.ndarray] = None
        self._monitor_global_path: List[Tuple[float, float]] = []
        self._monitor_dwa_traj: List[Tuple[float, float, float]] = []
        self._monitor_p_traj: List[Tuple[float, float, float]] = []
        self._dwa_reorient_done_per_robot: List[bool] = []
        self._dwa_reorient_done: bool = False

        # DWA config and planner — built once at init, reused every step.
        self._dwa_cfg: DWAConfig = DWAConfig(
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
        self._dwa_planner: DWALocalPlanner = DWALocalPlanner(self._dwa_cfg)

        # DWB config and planner — Nav2-compatible critic-based controller.
        self._dwb_cfg: DWBConfig = DWBConfig(
            lookahead=self.dwb_lookahead,
            res_lin_vel_space=self.dwb_res_lin_vel_space,
            res_ang_vel_space=self.dwb_res_ang_vel_space,
            min_linear_vel=self.dwb_min_linear_vel,
            max_linear_vel=self.dwb_max_linear_vel,
            min_angular_vel=self.dwb_min_angular_vel,
            max_angular_vel=self.dwb_max_angular_vel,
            max_acc=self.dwb_max_acc,
            max_ang_acc=self.dwb_max_ang_acc,
            max_dec=self.dwb_max_dec,
            robot_radius=self.dwb_robot_radius,
            safety_distance=self.dwb_safety_distance,
            min_dist_goal=self.dwb_min_dist_goal,
            yaw_goal_tolerance=self.dwb_yaw_goal_tolerance,
            oscillation_reset_dist=self.dwb_oscillation_reset_dist,
            oscillation_reset_angle=self.dwb_oscillation_reset_angle,
            scale_path_dist=self.dwb_scale_path_dist,
            scale_goal_dist=self.dwb_scale_goal_dist,
            scale_path_align=self.dwb_scale_path_align,
            scale_goal_align=self.dwb_scale_goal_align,
            scale_obstacle=self.dwb_scale_obstacle,
            scale_prefer_forward=self.dwb_scale_prefer_forward,
            scale_rotate_to_goal=self.dwb_scale_rotate_to_goal,
        )
        self._dwb_planner: DWBLocalPlanner = DWBLocalPlanner(self._dwb_cfg)

        # Cached navmesh occupancy grids per robot — keyed by a hash of the
        # navmesh vertex data so we only rebuild when the mesh changes.
        self._cached_occ_per_robot: Dict[int, Tuple[int, OccupancyGridResult]] = {}

        # Cached A* global path per robot — reused during reorientation steps
        # when the robot is rotating in place and the path is unchanged.
        self._cached_astar_path_per_robot: Dict[int, Optional[List[Tuple[float, float]]]] = {}

        # Per-robot local controller recovery counter — tracks consecutive
        # steps where the local controller was infeasible and the Nav2-style
        # Spin recovery behaviour was active.  After _DWA_MAX_RECOVERY_STEPS
        # consecutive recovery steps the A* cache is invalidated to trigger
        # a global replan, mirroring Nav2's behavior server escalation.
        self._dwa_recovery_count_per_robot: Dict[int, int] = {}
        self._DWA_MAX_RECOVERY_STEPS: int = 30

    def seed(self, seed: Optional[int] = None) -> List[Optional[int]]:
        """Seed action space, observation space, and internal RNG.

        Args:
            seed: Optional random seed used for deterministic behavior.

        Returns:
            A single-item list containing the effective seed.
        """
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
            self._rng = np.random.default_rng(seed=seed)
        return [seed]

    def reset(self) -> np.ndarray:
        """Reset episode state and prepare a new Unity reset payload.

        Returns:
            Initial flattened observation vector.
        """
        
        # Multi-robot RESET payload: all robots, target poses, and item poses are always provided.
        self.n_joints = int(self.config['num_joints'])

        # Active joints: mark all as active by default
        active_joints = [1.0] * self.n_joints

        # Initial joint positions of the AMR [rad or m depending on joint]
        init_pos = [0.0] * self.n_joints

        # Initial joint velocities of the AMR
        init_vel = [0.0] * self.n_joints

        # Convert planar poses [x, y, yaw] to 6-DOF ROS payloads [x, y, z, rx, ry, rz].
        if self.randomize_spawn_poses:
            target_pose_list, item_pose_list, robot_pose_list = self._sample_spawn_poses(self.amr_count, self.item_count)
        else:
            target_pose_list = self._target_poses_from_config(self.amr_count)
            item_pose_list = self._item_poses_from_config(self.item_count)
            robot_pose_list = self._robot_poses_from_config(self.amr_count)

        ground_surface = 0.5 * self.warehouse_config['ground']['ground_size'][2]

        target_poses_state = [
            self._planar_pose_to_payload(target_pose, height=ground_surface)
            for target_pose in target_pose_list
        ]
        item_poses_state: List[List[float]] = []
        for idx in range(self.item_count):
            item_entry = self.item_configs[idx]
            item_size = item_entry['item_size']
            item_height = 0.5 * float(item_size[2])
            item_pose = item_pose_list[idx]
            item_poses_state.append(
                self._planar_pose_to_payload(item_pose, height=ground_surface + item_height)
            )
        robot_poses_state = [
            self._planar_pose_to_payload(robot_pose, height=ground_surface)
            for robot_pose in robot_pose_list
        ]

        self.reset_state = {
            "robots": [
                {
                    "active_joints": list(active_joints),
                    "joint_positions": list(init_pos),
                    "joint_velocities": list(init_vel),
                    "robot_pose": list(robot_pose),
                }
                for robot_pose in robot_poses_state
            ],
            "targets": [list(target_pose) for target_pose in target_poses_state],
            "items": [list(item_pose) for item_pose in item_poses_state],
        }

        self.time_step = 0
        self._done = False
        self.success = False
        self.collision = False
        self.collided_env = 0
        self._last_observation = np.zeros(self.observation_dim, dtype=np.float32)
        self.unity_observation = {}
        self._step_cache = {}
        self._unity_robot_payloads = None
        self._per_robot_monitor_cache: List[Dict[str, Any]] = []
        self._dwa_reorient_done_per_robot: List[bool] = []
        self._monitor_global_path = []
        self._monitor_dwa_traj = []
        self._monitor_p_traj = []
        self._dwa_reorient_done = False
        self._cached_astar_path_per_robot.clear()
        self._cached_occ_per_robot.clear()
        self._dwa_recovery_count_per_robot.clear()
        self._dwb_planner.reset()
        if self.use_laser_scan and self.laser_count > 0:
            zeros = np.zeros(self.laser_count, dtype=np.float32)
            self.unity_observation['laser_scan'] = zeros
            self.laser_scan = zeros
        self._unity_image_bytes_list = []
        return self._last_observation.copy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Return the latest cached transition in Gym step format.

        Args:
            action: Action placeholder (planning outputs are computed in update()).

        Returns:
            Tuple of observation, reward, done flag, and info dictionary.
        """
        return self._last_observation.copy(), self._last_reward, self._done, {"success": self.success, "collision": self.collision}

    def update(self, unity_observation_dict: Dict[str, Any], time_step_update: bool = True) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Consume Unity payloads, rebuild per-robot observations, and compute rewards.

        Args:
            unity_observation_dict: Raw observation payload dictionary received from Unity.
            time_step_update: Whether to increment the episode step counter.

        Returns:
            Tuple of flattened observation, scalar reward, done flag, and info dictionary.
        """
        # Clear per-step caches
        self._step_cache.clear()
        prev_robot_cache = getattr(self, '_per_robot_monitor_cache', []) or []
        self._per_robot_monitor_cache = []
        self._unity_image_labels = []
        if not isinstance(self._dwa_reorient_done_per_robot, list):
            self._dwa_reorient_done_per_robot = []

        # Cache multi-robot payloads
        self._unity_robot_payloads = unity_observation_dict['Robots']
        expected_robot_count = len(self._unity_robot_payloads) if isinstance(self._unity_robot_payloads, list) else 0
        if not expected_robot_count:
            raise ValueError("WarehouseUnityEnv expects multi-robot payloads with 'Robots' populated.")
        if expected_robot_count != self.amr_count:
            raise ValueError(
                f"Robot payload count mismatch: expected {self.amr_count}, got {expected_robot_count}."
            )
        if expected_robot_count and len(self._dwa_reorient_done_per_robot) != expected_robot_count:
            existing_flags = list(self._dwa_reorient_done_per_robot) if isinstance(self._dwa_reorient_done_per_robot, list) else []
            self._dwa_reorient_done_per_robot = (existing_flags[:expected_robot_count] + [False] * expected_robot_count)[:expected_robot_count]

        # Decode overhead camera frames (list) and robot-mounted camera frames
        overhead_image_data = unity_observation_dict['OverheadImages']
        robot_image_payloads: List[Any] = []
        robot_image_labels: List[str] = []
        robots_payload = unity_observation_dict['Robots']
        if isinstance(robots_payload, list) and robots_payload:
            for idx, robot_payload in enumerate(robots_payload):
                robot_image = robot_payload['RobotImage']
                robot_image_data = robot_image['Data']
                robot_image_payloads.append(robot_image_data)
                label = f"Robot_{int(robot_payload['RobotIndex']) + 1}"
                robot_image_labels.append(label)

        if overhead_image_data is not None:
            overhead_payload = [entry['Data'] if isinstance(entry, dict) else entry for entry in overhead_image_data]
        else:
            overhead_payload = None
        overhead_frames = self._unity_retrieve_observation_images(overhead_payload) if overhead_payload is not None else []
        robot_frames = self._unity_retrieve_observation_images(robot_image_payloads) if robot_image_payloads else []

        overhead_labels = [f"Overhead_{i + 1}" for i in range(len(overhead_frames))]
        self._unity_image_labels = overhead_labels + robot_image_labels[:len(robot_frames)]

        # Combine with overhead images first, then robot camera so monitor sees all views
        self._unity_image_bytes_list = overhead_frames + robot_frames

        robots_payload = self._unity_robot_payloads

        obs_chunks: List[np.ndarray] = []
        rewards: List[float] = []
        collisions: List[bool] = []
        successes: List[bool] = []
        if isinstance(robots_payload, list) and len(robots_payload) > 0:
            self._current_item_count = self.item_count
            item_names = [f"Item_{i + 1}" for i in range(self.item_count)]
            last_action = getattr(self, '_monitor_last_action', None)

            def _build_robot_image_payload(robot_idx: int) -> Optional[Dict[str, Any]]:
                """Build monitor image payload for a specific robot view.

                Args:
                    robot_idx: Robot index in the current payload list.

                Returns:
                    Dictionary with frames, labels, and selected frame index, or None.
                """
                if not overhead_frames and not robot_frames:
                    return None
                frames: List[bytes] = list(overhead_frames)
                labels: List[str] = list(overhead_labels)
                if 0 <= robot_idx < len(robot_frames):
                    frames.append(robot_frames[robot_idx])
                    robot_label = robot_image_labels[robot_idx] if robot_idx < len(robot_image_labels) else f"Robot_{robot_idx + 1}"
                    labels.append(robot_label)
                    index = len(overhead_frames)
                else:
                    index = 0
                payload = {
                    'frames': frames,
                    'labels': labels,
                    'index': index,
                }
                return payload

            for idx, robot_payload in enumerate(robots_payload):
                self._unity_retrieve_observation_numeric(robot_payload, unity_observation_dict)

                # Optional obstacle poses (environment-level numeric payload)
                env_numeric = unity_observation_dict['Numeric']
                self.unity_observation['obstacles'] = env_numeric['Obstacles']

                robot_x = float(self.unity_observation['robot_position'][0])
                robot_y = float(self.unity_observation['robot_position'][1])
                robot_yaw = float(self.unity_observation['robot_yaw'])
                robot_linear_velocity = float(self.unity_observation['robot_linear_velocity'])
                robot_angular_velocity = float(self.unity_observation['robot_angular_velocity'])
                target_x = float(self.unity_observation['target_position'][0])
                target_y = float(self.unity_observation['target_position'][1])
                target_yaw = float(self.unity_observation['target_yaw'])

                delta_x = target_x - robot_x
                delta_y = target_y - robot_y
                delta_yaw = self._wrap_angle(target_yaw - robot_yaw)

                observation_base = np.array([
                    robot_x, robot_y, robot_yaw, robot_linear_velocity, robot_angular_velocity,
                    delta_x, delta_y, delta_yaw
                ], dtype=np.float32)
                if self.normalize_observation:
                    max_lin_i = float(self.max_linear_velocities[idx]) if idx < len(self.max_linear_velocities) else self.max_linear_velocity
                    max_ang_i = float(self.max_angular_velocities[idx]) if idx < len(self.max_angular_velocities) else self.max_angular_velocity
                    max_lin_i = max(max_lin_i, 1e-6)
                    max_ang_i = max(max_ang_i, 1e-6)
                    observation_base[0] /= self.position_normalization
                    observation_base[1] /= self.position_normalization
                    observation_base[2] /= self.yaw_normalization
                    observation_base[3] /= max_lin_i
                    observation_base[4] /= max_ang_i
                    observation_base[5] /= self.position_normalization
                    observation_base[6] /= self.position_normalization
                    observation_base[7] /= self.yaw_normalization

                # Laser scan (optional)
                if self.use_laser_scan:
                    laser_enabled_i = bool(self.laser_enabled_per_robot[idx]) if idx < len(self.laser_enabled_per_robot) else False
                    expected_count_i = int(self.laser_counts_per_robot[idx]) if idx < len(self.laser_counts_per_robot) else 0
                    fallback_range_i = float(self.laser_max_ranges_per_robot[idx]) if idx < len(self.laser_max_ranges_per_robot) else 0.0
                    angle_min_i = float(self.laser_angle_min_per_robot[idx]) if idx < len(self.laser_angle_min_per_robot) else 0.0
                    angle_max_i = float(self.laser_angle_max_per_robot[idx]) if idx < len(self.laser_angle_max_per_robot) else 0.0
                    sensor_offset_i = self.laser_sensor_offsets_per_robot[idx] if idx < len(self.laser_sensor_offsets_per_robot) else (0.0, 0.0)

                    laser_vec_raw = np.full(max(expected_count_i, 0), fallback_range_i, dtype=np.float32)
                    if laser_enabled_i and robot_payload['LaserScan'] is not None:
                        laser_in = robot_payload['LaserScan']
                        laser_ranges = laser_in['Ranges']
                        angle_min_i = float(laser_in['AngleMin'])
                        angle_max_i = float(laser_in['AngleMax'])
                        fallback_range_i = float(laser_in['RangeMax'])
                        sensor_offset_i = tuple(laser_in['SensorOffset'])
                        laser_vec_raw = np.asarray(laser_ranges, dtype=np.float32).reshape(-1)
                        if expected_count_i > 0 and laser_vec_raw.shape[0] != expected_count_i:
                            if laser_vec_raw.shape[0] > expected_count_i:
                                laser_vec_raw = laser_vec_raw[:expected_count_i]
                            else:
                                laser_vec_raw = np.pad(laser_vec_raw, (0, expected_count_i - laser_vec_raw.shape[0]), constant_values=fallback_range_i)

                    if self.laser_count > 0:
                        laser_vec = np.full(self.laser_count, fallback_range_i, dtype=np.float32)
                        copy_len = min(self.laser_count, laser_vec_raw.shape[0])
                        if copy_len > 0:
                            laser_vec[:copy_len] = laser_vec_raw[:copy_len]
                    else:
                        laser_vec = np.zeros(0, dtype=np.float32)

                    self.laser_angle_min = angle_min_i
                    self.laser_angle_max = angle_max_i
                    self.laser_max_range = fallback_range_i
                    self.laser_sensor_offset = sensor_offset_i

                    obs_chunk = np.concatenate([observation_base, laser_vec]).astype(np.float32)
                    self.unity_observation['laser_scan'] = laser_vec
                    self.laser_scan = laser_vec
                else:
                    obs_chunk = observation_base
                    laser_vec = np.array([], dtype=np.float32)

                obs_chunks.append(obs_chunk)

                collision_flag = float(self.unity_observation['collision_flag']) >= 0.5
                collisions.append(collision_flag)

                # Success detection per robot
                if self.normalize_observation:
                    dx_m = float(obs_chunk[5] * self.position_normalization)
                    dy_m = float(obs_chunk[6] * self.position_normalization)
                    dyaw_r = float(obs_chunk[7] * self.yaw_normalization)
                else:
                    dx_m = float(obs_chunk[5])
                    dy_m = float(obs_chunk[6])
                    dyaw_r = float(obs_chunk[7])
                dist = np.hypot(dx_m, dy_m)
                success_flag = bool(dist <= self.success_distance_threshold and abs(dyaw_r) <= self.success_yaw_threshold)
                successes.append(success_flag)

                reward_value = self._compute_reward(obs_chunk, success_flag, collision_flag)
                rewards.append(reward_value)

                action_vec = None
                if isinstance(last_action, np.ndarray):
                    if last_action.ndim == 2 and idx < last_action.shape[0]:
                        action_vec = last_action[idx]
                    elif last_action.ndim == 1 and last_action.size >= 2:
                        action_vec = last_action[:2]
                elif isinstance(last_action, (list, tuple)):
                    if last_action and isinstance(last_action[0], (list, tuple, np.ndarray)):
                        if idx < len(last_action):
                            action_vec = np.asarray(last_action[idx], dtype=np.float32)
                    elif len(last_action) >= 2:
                        action_vec = np.asarray(last_action[:2], dtype=np.float32)

                # Build item pose arrays for the monitor cache / data trace
                items_entries = self.unity_observation.get('items', [])
                item_positions = [np.asarray(e['position'], dtype=np.float32).tolist() for e in items_entries]
                item_orientations = [np.asarray(e['orientation_quat'], dtype=np.float32).tolist() for e in items_entries]
                item_yaws = [float(e['yaw']) for e in items_entries]

                # Build obstacle pose arrays for the monitor cache / data trace
                raw_obstacles = self.unity_observation.get('obstacles', [])
                obstacle_entries: List[Dict[str, Any]] = []
                for obs_raw in raw_obstacles:
                    if not isinstance(obs_raw, dict):
                        continue
                    pose = obs_raw.get('Pose', {})
                    obstacle_entries.append({
                        'position': list(pose.get('Position', [0.0, 0.0, 0.0])),
                        'orientation_quat': list(pose.get('Rotation', [1.0, 0.0, 0.0, 0.0])),
                        'is_dynamic': bool(obs_raw.get('IsDynamic', False)),
                    })

                per_robot_entry: Dict[str, Any] = {
                    'robot_name': f"Robot_{idx + 1}",
                    'robot_count': len(robots_payload),
                    'robot_position': np.array([robot_x, robot_y], dtype=np.float32),
                    'robot_yaw': float(robot_yaw),
                    'robot_velocity': [robot_linear_velocity, robot_angular_velocity],
                    'target_delta': [delta_x, delta_y, delta_yaw],
                    'target_position': [target_x, target_y],
                    'target_name': f"Target_{idx + 1}",
                    'target_yaw': float(target_yaw),
                    'item_names': list(item_names),
                    'item_positions': item_positions,
                    'item_orientations': item_orientations,
                    'item_yaws': item_yaws,
                    'obstacle_positions': [e['position'] for e in obstacle_entries],
                    'obstacle_orientations': [e['orientation_quat'] for e in obstacle_entries],
                    'obstacle_is_dynamic': [e['is_dynamic'] for e in obstacle_entries],
                    'reward': float(reward_value),
                    'success': bool(success_flag),
                    'collision': bool(collision_flag),
                    'agent_state': obs_chunk.copy(),
                    'agent_action': np.asarray(action_vec, dtype=np.float32) if action_vec is not None else None,
                    'agent_reward': np.array([reward_value], dtype=np.float32),
                    'agent_image': _build_robot_image_payload(idx),
                }
                if self.use_laser_scan:
                    per_robot_entry.update({
                        'laser_scan': laser_vec.copy(),
                        'laser_angle_min': float(self.laser_angle_min),
                        'laser_angle_max': float(self.laser_angle_max),
                        'laser_max_range': float(self.laser_max_range),
                        'laser_sensor_offset': np.array(self.laser_sensor_offset, dtype=float).tolist(),
                        'laser_points': self._laser_points_from_scan(
                            laser_vec,
                            robot_x,
                            robot_y,
                            robot_yaw,
                            angle_min=float(self.laser_angle_min),
                            angle_max=float(self.laser_angle_max),
                            sensor_offset=self.laser_sensor_offset,
                        ),
                    })

                # Per-robot NavMesh/occupancy/costmap
                navmesh_payload = self._unity_retrieve_observation_navmesh(robot_payload.get('NavMesh'))
                if navmesh_payload:
                    per_robot_entry['navmesh'] = navmesh_payload
                    occ_result = self._build_occupancy_from_navmesh(
                        navmesh_payload,
                        resolution=self.navmesh_occ_resolution,
                        padding_cells=self.navmesh_occ_padding,
                        rotation_deg=self.navmesh_occ_rotation,
                        robot_index=idx,
                    )
                    if occ_result is not None:
                        per_robot_entry['occupancy'] = {
                            'grid': occ_result.grid,
                            'resolution': float(occ_result.resolution),
                            'origin': list(occ_result.origin),
                        }
                        if occ_result.costmap is not None:
                            per_robot_entry['costmap'] = {
                                'costmap': occ_result.costmap,
                                'resolution': float(occ_result.resolution),
                                'origin': list(occ_result.origin),
                            }

                # Preserve last-step planner paths and maps until the planner refreshes them
                if idx < len(prev_robot_cache):
                    prev_entry = prev_robot_cache[idx]
                    if isinstance(prev_entry, dict):
                        if 'planner_paths' in prev_entry:
                            per_robot_entry['planner_paths'] = prev_entry['planner_paths']
                        if 'planner_map' in prev_entry:
                            per_robot_entry['planner_map'] = prev_entry['planner_map']
                        if per_robot_entry['agent_action'] is None and 'agent_action' in prev_entry:
                            per_robot_entry['agent_action'] = prev_entry['agent_action']
                self._per_robot_monitor_cache.append(per_robot_entry)

            self.success = all(successes) if successes else False
            self.collision = any(collisions)
            self._last_rewards = np.array(rewards, dtype=np.float32)
            flat_obs = np.concatenate(obs_chunks) if obs_chunks else np.zeros(self.observation_dim, dtype=np.float32)
            # Pad/truncate to match observation_dim
            if flat_obs.size < self.observation_dim:
                flat_obs = np.pad(flat_obs, (0, self.observation_dim - flat_obs.size))
            else:
                flat_obs = flat_obs[:self.observation_dim]
            self._last_observation = flat_obs
        if time_step_update:
            self.time_step += 1
            self._done = self.time_step >= self.max_time_steps or self.success or self.collision

        self._last_reward = float(np.sum(self._last_rewards))
        info = {
            "success": bool(self.success),
            "collision": bool(self.collision),
            "robot_count": self.amr_count,
            "per_robot_rewards": self._last_rewards.copy().tolist()
        }
        return self._last_observation.copy(), self._last_reward, self._done, info

    def render(self, mode: str = 'human') -> None:
        """No-op render hook for Gym interface compatibility.

        Args:
            mode: Requested render mode.
        """
        return None

    def close(self) -> None:
        """No-op cleanup hook for Gym interface compatibility."""
        return None

    def get_terminal_reward(self) -> bool:
        """Return whether the current episode reached success.

        Returns:
            True when the task goal is achieved, otherwise false.
        """
        return bool(self.success)

    def _compute_reward(self, obs_vec: np.ndarray, success: bool, collision: bool) -> float:
        """Compute shaped scalar reward from one robot observation chunk.

        Args:
            obs_vec: Per-robot observation vector.
            success: Success flag for this robot.
            collision: Collision flag for this robot.

        Returns:
            Shaped reward value.
        """
        step_penalty = float(self.config['step_penalty'])
        success_reward = float(self.config['success_reward'])
        collision_penalty = float(self.config['collision_penalty'])
        distance_weight = float(self.config['distance_weight'])
        yaw_weight = float(self.config['yaw_weight'])

        if self.normalize_observation:
            dx_m = float(obs_vec[5] * self.position_normalization)
            dy_m = float(obs_vec[6] * self.position_normalization)
            dyaw_r = float(obs_vec[7] * self.yaw_normalization)
        else:
            dx_m = float(obs_vec[5])
            dy_m = float(obs_vec[6])
            dyaw_r = float(obs_vec[7])

        dist = np.hypot(dx_m, dy_m)
        shaped = -step_penalty - distance_weight * dist - yaw_weight * abs(dyaw_r)
        if success:
            shaped += success_reward
        if collision:
            shaped -= collision_penalty
        return float(shaped)

    def _sample_spawn_poses(self, robot_count: int, item_count: int) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
        """Sample non-overlapping planar poses for targets, items, and robots.

        Args:
            robot_count: Number of robot poses required.
            item_count: Number of item poses required.

        Returns:
            Tuple of target pose list, item pose list, and robot pose list.
        """
        robot_count = max(int(robot_count), 1)
        item_count = max(int(item_count), 0)
        ground_cfg = self.warehouse_config['ground']
        ground_size = ground_cfg['ground_size']
        base_margin = float(self.config['spawn_min_separation']) * 0.5
        robot_radius = float(self.dwa_robot_radius)
        margin = max(base_margin, robot_radius)
        half_x = float(ground_size[0]) * 0.5 - margin
        half_y = float(ground_size[1]) * 0.5 - margin
        min_sep = float(self.config['spawn_min_separation'])
        max_attempts = 200

        poses: List[List[float]] = []

        def _valid(candidate: Tuple[float, float]) -> bool:
            """Check whether candidate XY pose respects minimum separation.

            Args:
                candidate: Candidate position as (x, y).

            Returns:
                True when candidate is valid; otherwise false.
            """
            for px, py, _ in poses:
                if np.hypot(candidate[0] - px, candidate[1] - py) < min_sep:
                    return False
            return True

        attempts = 0
        total_needed = robot_count * 2 + item_count
        while len(poses) < total_needed and attempts < max_attempts:
            x = float(self._rng.uniform(-half_x, half_x))
            y = float(self._rng.uniform(-half_y, half_y))
            yaw = float(self._rng.uniform(-math.pi, math.pi))
            if _valid((x, y)):
                poses.append([x, y, yaw])
            attempts += 1

        if len(poses) < total_needed:
            raise RuntimeError(
                f"Failed to sample non-overlapping spawn poses (needed={total_needed}, sampled={len(poses)})."
            )

        # First block is targets, second block is items, remaining are robots
        target_pose_list = poses[:robot_count]
        item_pose_list = poses[robot_count:robot_count + item_count]
        robot_pose_list = poses[robot_count + item_count:robot_count + item_count + robot_count]
        return target_pose_list, item_pose_list, robot_pose_list

    def _robot_poses_from_config(self, robot_count: int) -> List[List[float]]:
        """Read robot spawn poses from strict multi-robot configuration.

        Args:
            robot_count: Number of robot poses to read.

        Returns:
            List of [x, y, yaw] robot poses.
        """
        robot_count = max(int(robot_count), 1)
        poses_cfg = self.config['robot_poses']
        if len(poses_cfg) < robot_count:
            raise ValueError(f"robot_poses must include at least {robot_count} entries.")
        return [[float(entry[0]), float(entry[1]), float(entry[2])] for entry in poses_cfg[:robot_count]]

    def _target_poses_from_config(self, robot_count: int) -> List[List[float]]:
        """Read target poses from strict multi-robot configuration.

        Args:
            robot_count: Number of target poses to read.

        Returns:
            List of [x, y, yaw] target poses.
        """
        robot_count = max(int(robot_count), 1)
        poses_cfg = self.config['target_poses']
        if len(poses_cfg) < robot_count:
            raise ValueError(f"target_poses must include at least {robot_count} entries.")
        return [[float(entry[0]), float(entry[1]), float(entry[2])] for entry in poses_cfg[:robot_count]]

    def _item_poses_from_config(self, item_count: int) -> List[List[float]]:
        """Read item poses from strict item configuration.

        Args:
            item_count: Number of item poses to read.

        Returns:
            List of [x, y, yaw] item poses.
        """
        item_count = max(int(item_count), 0)
        if item_count == 0:
            return []
        poses_cfg = self.config['item_poses']
        if len(poses_cfg) < item_count:
            raise ValueError(f"item_poses must include at least {item_count} entries.")
        return [[float(entry[0]), float(entry[1]), float(entry[2])] for entry in poses_cfg[:item_count]]

    def random_action(self) -> np.ndarray:
        """Sample a random action from the configured action space.

        Returns:
            Action array shaped as (robot_count, 2).
        """
        return self.action_space.sample()

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map normalized actions in [-1, 1] to physical velocity limits.

        Args:
            action: Normalized action array.

        Returns:
            Action array in physical units [m/s, rad/s].
        """
        if action is None:
            return np.zeros((self.amr_count, 2), dtype=np.float32)
        act = np.asarray(action, dtype=float)
        if act.ndim == 1:
            act = act.reshape(-1, 2)
        cmds = []
        for idx, row in enumerate(act):
            max_lin_i = float(self.max_linear_velocities[idx]) if idx < len(self.max_linear_velocities) else self.max_linear_velocity
            max_ang_i = float(self.max_angular_velocities[idx]) if idx < len(self.max_angular_velocities) else self.max_angular_velocity
            v = float(np.clip(row[0], -1.0, 1.0) * max_lin_i)
            w = float(np.clip(row[1], -1.0, 1.0) * max_ang_i)
            cmds.append([v, w])
        return np.array(cmds, dtype=np.float32)

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Map physical velocity commands to normalized [-1, 1] actions.

        Args:
            action: Physical action array in [m/s, rad/s].

        Returns:
            Normalized action array.
        """
        if action is None:
            return np.zeros((self.amr_count, 2), dtype=np.float32)
        act = np.asarray(action, dtype=float)
        if act.ndim == 1:
            act = act.reshape(-1, 2)
        cmds = []
        for idx, row in enumerate(act):
            max_lin_i = float(self.max_linear_velocities[idx]) if idx < len(self.max_linear_velocities) else self.max_linear_velocity
            max_ang_i = float(self.max_angular_velocities[idx]) if idx < len(self.max_angular_velocities) else self.max_angular_velocity
            max_lin_i = max(max_lin_i, 1e-6)
            max_ang_i = max(max_ang_i, 1e-6)
            v = float(np.clip(row[0] / max_lin_i, -1.0, 1.0))
            w = float(np.clip(row[1] / max_ang_i, -1.0, 1.0))
            cmds.append([v, w])
        return np.array(cmds, dtype=np.float32)

    def action_by_p_control(self, k_v: float = 1.0, k_w: float = 1.0) -> np.ndarray:
        """Compute per-robot commands using A* + DWA with P-control fallback.

        Args:
            k_v: Linear proportional gain used by fallback controller.
            k_w: Angular proportional gain used by fallback controller.

        Returns:
            Per-robot action array clipped to action-space limits.
        """
        robots_payload = getattr(self, '_per_robot_monitor_cache', None)
        # Always operate per-robot.
        if isinstance(robots_payload, list) and len(robots_payload) >= 1:
            actions: List[np.ndarray] = []
            if len(self._dwa_reorient_done_per_robot) < len(robots_payload):
                self._dwa_reorient_done_per_robot = (self._dwa_reorient_done_per_robot + [False] * len(robots_payload))[:len(robots_payload)]
            elif len(self._dwa_reorient_done_per_robot) > len(robots_payload):
                self._dwa_reorient_done_per_robot = self._dwa_reorient_done_per_robot[:len(robots_payload)]
            for idx in range(len(robots_payload)):
                entry = robots_payload[idx]
                r_pos = entry['robot_position']
                try:
                    rx = float(r_pos[0])
                    ry = float(r_pos[1])
                except Exception:
                    rx, ry = 0.0, 0.0
                r_yaw = float(entry['robot_yaw'])
                r_lin = float(entry['robot_velocity'][0])
                r_ang = float(entry['robot_velocity'][1])

                target_pos_entry = None
                if isinstance(entry, dict):
                    tp = entry['target_position']
                    if tp is not None and len(tp) >= 2:
                        target_pos_entry = (float(tp[0]), float(tp[1]))
                if target_pos_entry is None:
                    t_raw = self.unity_observation['target_position']
                    if t_raw is not None and len(t_raw) >= 2:
                        target_pos_entry = (float(t_raw[0]), float(t_raw[1]))

                # Skip A* during reorientation: the robot is rotating in
                # place (v=0) so the path from the previous step is still
                # valid.  Only re-plan when reorientation is done or when
                # there is no cached path yet.
                reorient_flag = self._dwa_reorient_done_per_robot[idx] if idx < len(self._dwa_reorient_done_per_robot) else False
                cached_path = self._cached_astar_path_per_robot.get(idx)
                if reorient_flag or cached_path is None:
                    global_path = self.plan_global_path_astar(
                        resolution=self.navmesh_occ_resolution,
                        padding_cells=self.navmesh_occ_padding,
                        allow_diagonal=True,
                        unknown_is_obstacle=False,
                        rotation_deg=self.navmesh_occ_rotation,
                        robot_pose=(rx, ry),
                        target_pose=target_pos_entry,
                        navmesh_payload=entry.get('navmesh') if isinstance(entry, dict) else None,
                        robot_index=idx,
                    )
                    self._cached_astar_path_per_robot[idx] = global_path
                else:
                    global_path = cached_path

                planner_paths_local: Dict[str, Any] = {}
                planner_action = None
                dwa_traj_local: List[Tuple[float, float, float]] = []
                p_traj_local: List[Tuple[float, float, float]] = []
                if global_path is not None and len(global_path) >= 2:
                    max_lin_i = float(self.max_linear_velocities[idx]) if idx < len(self.max_linear_velocities) else self.max_linear_velocity
                    max_ang_i = float(self.max_angular_velocities[idx]) if idx < len(self.max_angular_velocities) else self.max_angular_velocity
                    twist_local = (
                        float(np.clip(r_lin, -max_lin_i, max_lin_i)),
                        float(np.clip(-r_ang, -max_ang_i, max_ang_i)),
                    )
                    reorient_flag = self._dwa_reorient_done_per_robot[idx] if idx < len(self._dwa_reorient_done_per_robot) else False
                    target_yaw_local = r_yaw
                    if isinstance(entry, dict) and entry['target_yaw'] is not None:
                        try:
                            target_yaw_local = float(entry['target_yaw'])
                        except Exception:
                            target_yaw_local = r_yaw
                    planner_action, dwa_traj_local, reorient_next = self.compute_controller_action(
                        global_path_world=global_path,
                        current_pose=(rx, ry, r_yaw),
                        current_twist=twist_local,
                        target_yaw_local=target_yaw_local,
                        reorient_done=reorient_flag,
                        robot_index=idx,
                    )
                    if idx < len(self._dwa_reorient_done_per_robot):
                        self._dwa_reorient_done_per_robot[idx] = reorient_next

                if planner_action is None:
                    # Safe stop: no obstacle-aware path available - hold
                    # position rather than blindly driving through obstacles
                    # with a naive P-controller.
                    planner_action = np.array([0.0, 0.0], dtype=np.float32)
                    dwa_traj_local = []

                planner_paths_local['global_path'] = global_path or []
                planner_paths_local['dwa_traj'] = dwa_traj_local
                planner_paths_local['p_traj'] = list(p_traj_local) if p_traj_local else []

                # Update per-robot cache entry so task monitor sees robot-specific planner overlays
                if isinstance(entry, dict):
                    entry['planner_paths'] = planner_paths_local
                    entry['agent_action'] = planner_action
                actions.append(np.asarray(planner_action, dtype=np.float32))

            if actions:
                actions_arr = np.vstack(actions)
                self._monitor_last_action = actions_arr
                return np.clip(actions_arr, -self.action_space.high, self.action_space.high)
        # No legacy fallback; if no actions computed, return zeros per robot
        zero_actions = np.zeros((self.amr_count, 2), dtype=np.float32)
        self._monitor_last_action = zero_actions
        self._monitor_p_traj = []
        self._monitor_dwa_traj = []
        self._monitor_global_path = []
        return zero_actions

    def plan_global_path_astar(
        self,
        resolution: float = 0.1,
        padding_cells: int = 2,
        allow_diagonal: bool = True,
        unknown_is_obstacle: bool = True,
        rotation_deg: float = 0.0,
        cost_weight: float = 3.0,
        robot_pose: Optional[Tuple[float, float]] = None,
        target_pose: Optional[Tuple[float, float]] = None,
        navmesh_payload: Optional[Dict[str, Any]] = None,
        robot_index: int = 0,
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan a global path on a per-robot navmesh-derived occupancy grid using A*.

        Each robot receives its own NavMesh from Unity where the active
        robot's carved hole is already patched as walkable.  The grid and
        costmap are derived from this per-robot mesh and used directly for
        A* planning.

        Args:
            resolution: Occupancy grid resolution in meters.
            padding_cells: Grid padding in cells.
            allow_diagonal: Whether diagonal A* transitions are allowed.
            unknown_is_obstacle: Whether unknown cells are treated as blocked.
            rotation_deg: Rasterization frame rotation in degrees.
            cost_weight: Costmap weight for weighted A*.
            robot_pose: Current robot position (x, y).
            target_pose: Target position (x, y).
            navmesh_payload: Per-robot navmesh payload dictionary.
            robot_index: Robot index for occupancy grid cache lookup.

        Returns:
            List of world-frame waypoints or None when planning fails.
        """
        if robot_pose is None or target_pose is None or navmesh_payload is None:
            return None

        occ = self._build_occupancy_from_navmesh(
            navmesh_payload,
            resolution=resolution,
            padding_cells=padding_cells,
            rotation_deg=rotation_deg,
            robot_index=robot_index,
        )

        if occ is None:
            return None
        robot_x = float(robot_pose[0])
        robot_y = float(robot_pose[1])
        target_x = float(target_pose[0])
        target_y = float(target_pose[1])

        def _to_cell(x: float, y: float) -> Tuple[int, int]:
            """Convert world coordinates to occupancy grid row/column indices.

            Args:
                x: World x coordinate.
                y: World y coordinate.

            Returns:
                Tuple of (row, column) indices.
            """
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
            cost_grid=occ.costmap,
            cost_weight=cost_weight,
            min_obstacle_distance_cells=self.astar_obstacle_inflation_cells,
            project_start_goal_to_valid=True,
            max_projection_distance_cells=self.astar_max_projection_distance_cells,
            allow_partial=True,
        )
        if result is None or not result.path:
            return None

        def _to_world(cell: Tuple[int, int]) -> Tuple[float, float]:
            """Convert occupancy grid row/column indices back to world coordinates.

            Args:
                cell: Tuple containing (row, column).

            Returns:
                Tuple containing world (x, y) at cell center.
            """
            r, c = cell
            x = occ.origin[0] + (c + 0.5) * occ.resolution
            y = occ.origin[1] + (r + 0.5) * occ.resolution
            return (x, y)

        return [_to_world(cell) for cell in result.path]

    def _build_occupancy_from_navmesh(
        self,
        navmesh_payload: Optional[Dict[str, Any]],
        resolution: float = 0.1,
        padding_cells: int = 1,
        rotation_deg: float = 0.0,
        robot_index: int = 0,
    ) -> Optional[OccupancyGridResult]:
        """Convert a per-robot navmesh payload into an occupancy grid.

        Results are cached per robot index and only recomputed when the navmesh
        vertex data changes (detected via a fast hash of the vertex bytes).

        Args:
            navmesh_payload: Navmesh dictionary containing vertices and indices.
            resolution: Grid resolution in meters.
            padding_cells: Grid padding in cells.
            rotation_deg: Rasterization rotation in degrees.
            robot_index: Robot index used as cache key.

        Returns:
            OccupancyGridResult when successful; otherwise None.
        """
        if navmesh_payload is None:
            return None
        vertices = navmesh_payload['vertices'] if isinstance(navmesh_payload, dict) else None
        indices = navmesh_payload['indices'] if isinstance(navmesh_payload, dict) else None
        if vertices is None or indices is None:
            return None

        # Cache lookup: hash vertex bytes so we only rebuild when the mesh changes.
        vert_arr = np.asarray(vertices, dtype=float)
        vert_hash = hash(vert_arr.data.tobytes())

        cached = self._cached_occ_per_robot.get(robot_index)
        if cached is not None and cached[0] == vert_hash:
            return cached[1]

        face_arr = np.reshape(np.asarray(indices, dtype=int), (-1, 3))

        try:
            result = navmesh_to_occupancy_grid(
                vertices=vert_arr,
                faces=face_arr,
                resolution=resolution,
                padding_cells=padding_cells,
                rotation_deg=rotation_deg,
            )
        except Exception:
            return None

        if result is not None:
            self._cached_occ_per_robot[robot_index] = (vert_hash, result)
        return result

    def compute_controller_action(
        self,
        global_path_world: Sequence[Tuple[float, float]],
        obstacles_world: Optional[np.ndarray] = None,
        current_pose: Optional[Tuple[float, float, float]] = None,
        current_twist: Optional[Tuple[float, float]] = None,
        target_yaw_local: Optional[float] = None,
        reorient_done: bool = False,
        robot_index: int = 0,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[float, float, float]], bool]:
        """Dispatch to either DWA (ROS1) or DWB (Nav2) local controller.

        The active controller is selected by ``self.local_controller_type``.
        Both paths share the same return signature so the step loop is
        agnostic to the controller choice.

        Args:
            global_path_world: World-frame global path waypoints.
            obstacles_world: Optional world-frame obstacle points.
            current_pose: Current robot pose (x, y, yaw).
            current_twist: Current robot twist (v, omega).
            target_yaw_local: Target yaw in local planar frame.
            reorient_done: Whether initial reorientation has completed.
            robot_index: Robot index for per-robot state tracking.

        Returns:
            Tuple of action array, rollout trajectory, and reorientation state.
        """
        if self.local_controller_type == 'DWB':
            return self._compute_dwb_action(
                global_path_world=global_path_world,
                obstacles_world=obstacles_world,
                current_pose=current_pose,
                current_twist=current_twist,
                target_yaw_local=target_yaw_local,
                reorient_done=reorient_done,
                robot_index=robot_index,
            )
        return self._compute_dwa_action(
            global_path_world=global_path_world,
            obstacles_world=obstacles_world,
            current_pose=current_pose,
            current_twist=current_twist,
            target_yaw_local=target_yaw_local,
            reorient_done=reorient_done,
            robot_index=robot_index,
        )

    # ------------------------------------------------------------------
    #  DWA (ROS1-style) implementation
    # ------------------------------------------------------------------

    def _compute_dwa_action(
        self,
        global_path_world: Sequence[Tuple[float, float]],
        obstacles_world: Optional[np.ndarray] = None,
        current_pose: Optional[Tuple[float, float, float]] = None,
        current_twist: Optional[Tuple[float, float]] = None,
        target_yaw_local: Optional[float] = None,
        reorient_done: bool = False,
        robot_index: int = 0,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[float, float, float]], bool]:
        """Compute ROS1-style DWA velocity command for a robot.

        Performs a one-time reorientation to the initial path segment
        (Stage 1), then hands off to pure DWA tracking (Stage 2).
        When the robot is essentially at the goal position, it aligns
        to the target yaw.  Returns ``None`` on total infeasibility so
        the caller can fall back to P-control.

        Args:
            global_path_world: World-frame global path waypoints.
            obstacles_world: Optional world-frame obstacle points.
            cfg: Optional planner configuration override.
            current_pose: Current robot pose (x, y, yaw).
            current_twist: Current robot twist (v, omega).
            target_yaw_local: Target yaw in local planar frame.
            reorient_done: Whether initial reorientation has completed.
            robot_index: Robot index (unused, kept for interface compatibility).

        Returns:
            Tuple of (action_array, rollout_trajectory, reorient_state).
            ``action_array`` is ``None`` when the planner is infeasible.
        """
        self._monitor_global_path = [tuple(map(float, p)) for p in global_path_world] if global_path_world else []
        traj_out: List[Tuple[float, float, float]] = []
        reorient_state = bool(reorient_done)
        cfg = self._dwa_cfg
        if current_pose is None or current_twist is None or target_yaw_local is None:
            raise ValueError("_compute_dwa_action requires current_pose, current_twist, and target_yaw_local.")

        path_arr = np.asarray(global_path_world, dtype=float)
        if path_arr.size:
            # Stage 1: one-time reorientation to the initial path segment,
            # then permanently hand off to DWA.
            if not reorient_state and path_arr.shape[0] >= 2:
                init_heading = math.atan2(path_arr[1, 1] - path_arr[0, 1], path_arr[1, 0] - path_arr[0, 0])
                heading_err_init = self._wrap_angle(init_heading - current_pose[2])
                turn_exit = 0.3
                if abs(heading_err_init) > turn_exit:
                    if abs(heading_err_init) > 0.6:
                        w_init = float(np.sign(heading_err_init) * cfg.max_angular_vel)
                    else:
                        w_init = float(np.clip(heading_err_init * 3.0, cfg.min_angular_vel, cfg.max_angular_vel))
                    traj_out = [current_pose] + self._preview_trajectory(current_pose, 0.0, w_init, horizon=cfg.lookahead)
                    return np.array([0.0, w_init], dtype=np.float32), traj_out, reorient_state
                reorient_state = True

            goal_wp = path_arr[-1]
            dist_to_goal = float(np.hypot(goal_wp[0] - current_pose[0], goal_wp[1] - current_pose[1]))

            # If we are essentially at the goal position, prioritize
            # aligning to the target yaw.
            if dist_to_goal < cfg.min_dist_goal:
                yaw_err_to_target = self._wrap_angle(target_yaw_local - current_pose[2])
                if abs(yaw_err_to_target) > 0.05:
                    w_align = float(np.clip(yaw_err_to_target * 1.5, cfg.min_angular_vel, cfg.max_angular_vel))
                    traj_out = [current_pose] + self._preview_trajectory(current_pose, 0.0, w_align, horizon=cfg.lookahead)
                    return np.array([0.0, w_align], dtype=np.float32), traj_out, reorient_state
                # Heading already aligned — stop.
                return np.array([0.0, 0.0], dtype=np.float32), [], reorient_state

        obs_array = obstacles_world if obstacles_world is not None else np.zeros((0, 2), dtype=float)
        result: Optional[DWAResult] = self._dwa_planner.run(
            current_twist=current_twist,
            current_pose=current_pose,
            global_path=global_path_world,
            obstacles=obs_array,
            force_follow_plan=True,
        )
        if result is None:
            return None, traj_out, reorient_state

        # Stage 2: pure DWA tracking once initial reorientation is done.
        traj_out = [current_pose] + list(result.trajectory)
        return np.array([result.linear_vel, result.angular_vel], dtype=np.float32), traj_out, reorient_state

    # ------------------------------------------------------------------
    #  DWB (Nav2-style critic-based) implementation
    # ------------------------------------------------------------------

    def _compute_dwb_action(
        self,
        global_path_world: Sequence[Tuple[float, float]],
        obstacles_world: Optional[np.ndarray] = None,
        current_pose: Optional[Tuple[float, float, float]] = None,
        current_twist: Optional[Tuple[float, float]] = None,
        target_yaw_local: Optional[float] = None,
        reorient_done: bool = False,
        robot_index: int = 0,
    ) -> Tuple[Optional[np.ndarray], List[Tuple[float, float, float]], bool]:
        """Compute DWB (Nav2) control for a robot.

        Unlike the DWA path, the DWB planner's **PathAlign** critic provides
        heading gradient that naturally aligns the robot with the path
        direction — so the explicit reorientation stage is not needed.
        The DWB planner also has built-in RotateToGoal and oscillation
        detection.

        When all DWB trajectories are infeasible the same Nav2 Spin recovery
        is applied as in the DWA path, with A* cache invalidation after
        ``_DWA_MAX_RECOVERY_STEPS`` consecutive failures.

        Args:
            global_path_world: World-frame global path waypoints.
            obstacles_world: Optional world-frame obstacle points.
            current_pose: Current robot pose (x, y, yaw).
            current_twist: Current robot twist (v, omega).
            target_yaw_local: Target yaw in local planar frame.
            reorient_done: Whether initial reorientation has completed.
            robot_index: Robot index for per-robot state tracking.

        Returns:
            Tuple of action array, rollout trajectory, and reorientation state.
        """
        self._monitor_global_path = [tuple(map(float, p)) for p in global_path_world] if global_path_world else []
        traj_out: List[Tuple[float, float, float]] = []
        cfg = self._dwb_cfg

        if current_pose is None or current_twist is None or target_yaw_local is None:
            raise ValueError("_compute_dwb_action requires current_pose, current_twist, and target_yaw_local.")

        path_arr = np.asarray(global_path_world, dtype=float)
        if not path_arr.size:
            return None, traj_out, True

        goal_wp = path_arr[-1]
        dist_to_goal = float(np.hypot(goal_wp[0] - current_pose[0], goal_wp[1] - current_pose[1]))

        # ---- Final goal alignment to target_yaw (transport orientation) ----
        # DWB's RotateToGoal rotates toward the goal direction;
        # here we additionally align to the task-level target yaw once
        # the robot is at the goal position.
        if dist_to_goal < cfg.min_dist_goal:
            yaw_err_to_target = self._wrap_angle(target_yaw_local - current_pose[2])
            if abs(yaw_err_to_target) > cfg.yaw_goal_tolerance:
                w_align = float(np.clip(
                    yaw_err_to_target * 1.5,
                    cfg.min_angular_vel,
                    cfg.max_angular_vel,
                ))
                traj_out = [current_pose] + self._preview_trajectory(
                    current_pose, 0.0, w_align, horizon=cfg.lookahead,
                )
                return np.array([0.0, w_align], dtype=np.float32), traj_out, True
            # Already aligned — stop
            return np.array([0.0, 0.0], dtype=np.float32), [], True

        # ---- Run DWB critic-based planner ----
        obs_array = obstacles_world if obstacles_world is not None else np.zeros((0, 2), dtype=float)
        result: Optional[DWBResult] = self._dwb_planner.run(
            current_twist=current_twist,
            current_pose=current_pose,
            global_path=global_path_world,
            obstacles=obs_array,
            force_follow_plan=True,
        )

        if result is None:
            # ----- Nav2-style Spin recovery behaviour -----
            recovery_count = self._dwa_recovery_count_per_robot.get(robot_index, 0) + 1
            self._dwa_recovery_count_per_robot[robot_index] = recovery_count

            if recovery_count > self._DWA_MAX_RECOVERY_STEPS:
                self._cached_astar_path_per_robot.pop(robot_index, None)
                self._dwa_recovery_count_per_robot[robot_index] = 0

            # Rotate toward the next waypoint ahead of the closest point
            if path_arr.shape[0] >= 2:
                dists_to_robot = np.hypot(
                    path_arr[:, 0] - current_pose[0],
                    path_arr[:, 1] - current_pose[1],
                )
                closest_idx = int(np.argmin(dists_to_robot))
                look_idx = min(closest_idx + 1, path_arr.shape[0] - 1)
                desired_heading = math.atan2(
                    path_arr[look_idx, 1] - current_pose[1],
                    path_arr[look_idx, 0] - current_pose[0],
                )
                heading_err = self._wrap_angle(desired_heading - current_pose[2])
                if abs(heading_err) > 0.6:
                    w_rec = float(np.sign(heading_err) * cfg.max_angular_vel)
                else:
                    w_rec = float(np.clip(heading_err * 3.0, cfg.min_angular_vel, cfg.max_angular_vel))
                traj_out = [current_pose] + self._preview_trajectory(
                    current_pose, 0.0, w_rec, horizon=cfg.lookahead,
                )
                return np.array([0.0, w_rec], dtype=np.float32), traj_out, True

            return None, traj_out, True

        # DWB succeeded — clear recovery counter
        self._dwa_recovery_count_per_robot[robot_index] = 0

        traj_out = [current_pose] + list(result.trajectory)
        return np.array([result.linear_vel, result.angular_vel], dtype=np.float32), traj_out, True

    def _preview_trajectory(
        self,
        pose: Tuple[float, float, float],
        linear: float,
        angular: float,
        horizon: float,
        steps: int = 12,
    ) -> List[Tuple[float, float, float]]:
        """Generate a short kinematic rollout used for monitor visualization.

        Uses vectorized NumPy operations instead of a Python for-loop.

        Args:
            pose: Starting pose as (x, y, yaw).
            linear: Linear velocity command.
            angular: Angular velocity command.
            horizon: Simulation horizon in seconds.
            steps: Number of rollout points.

        Returns:
            List of sampled (x, y, yaw) states.
        """
        dt = horizon / max(int(steps), 1)
        x, y, yaw = pose
        step_arr = np.arange(1, steps + 1)

        if abs(angular) < 1e-6:
            offsets = linear * dt * step_arr
            xs = x + offsets * math.cos(yaw)
            ys = y + offsets * math.sin(yaw)
            yaws = np.full(steps, yaw)
        else:
            # If turning in place (v≈0), fabricate a small arc for visualisation.
            radius = 0.2 if abs(linear) < 1e-6 else (linear / angular if angular != 0 else 0.0)
            yaw_steps = yaw + angular * dt * step_arr
            xs = x - radius * math.sin(yaw) + radius * np.sin(yaw_steps)
            ys = y + radius * math.cos(yaw) - radius * np.cos(yaw_steps)
            yaws = yaw_steps

        return list(zip(xs.tolist(), ys.tolist(), yaws.tolist() if isinstance(yaws, np.ndarray) else [yaw] * steps))

    def get_monitor_payload(self) -> MonitorPayload:
        """Build structured monitor telemetry for all robots.

        Returns:
            MonitorPayload with robot state, maps, planner outputs, and agent fields.
        """
        robots_data = getattr(self, '_per_robot_monitor_cache', None) or []
        if not robots_data:
            return MonitorPayload(
                robot_pose=Pose2D(0.0, 0.0, 0.0),
                robot_velocity=(0.0, 0.0),
                target_delta=(0.0, 0.0, 0.0),
                reward=0.0,
                success=False,
                collision=False,
                planner=PlannerData(),
                target_pos=(0.0, 0.0),
            )
        # Top-level monitor fields use the first robot payload; all robots are available via `robots`.
        selected_robot = robots_data[0] if isinstance(robots_data[0], dict) else {}

        # Poses
        robot_pos_xy = selected_robot.get('robot_position', [0.0, 0.0])
        robot_x = float(robot_pos_xy[0]) if len(robot_pos_xy) >= 1 else 0.0
        robot_y = float(robot_pos_xy[1]) if len(robot_pos_xy) >= 2 else 0.0
        robot_yaw = float(selected_robot.get('robot_yaw', 0.0))
        robot_pose = Pose2D(robot_x, robot_y, robot_yaw)

        target_pos_list = selected_robot.get('target_position', [0.0, 0.0])
        target_pos = (
            float(target_pos_list[0]) if len(target_pos_list) >= 1 else 0.0,
            float(target_pos_list[1]) if len(target_pos_list) >= 2 else 0.0,
        )

        td = selected_robot.get('target_delta', [0.0, 0.0, 0.0])
        target_delta = (
            float(td[0]) if len(td) >= 1 else 0.0,
            float(td[1]) if len(td) >= 2 else 0.0,
            float(td[2]) if len(td) >= 3 else 0.0,
        )

        rv = selected_robot.get('robot_velocity', [0.0, 0.0])
        linear_velocity = float(rv[0]) if len(rv) >= 1 else 0.0
        angular_velocity = float(rv[1]) if len(rv) >= 2 else 0.0

        laser_scan_dc, laser_points_dc = self._laser_dataclass(selected_robot)

        occ_dc, cost_dc = self._occupancy_dataclass(selected_robot)
        navmesh_dc = self._navmesh_dataclass(selected_robot)

        pp = selected_robot.get('planner_paths') if isinstance(selected_robot.get('planner_paths'), dict) else {}
        try:
            planner_paths = PlannerPaths(
                global_path=[(float(p[0]), float(p[1])) for p in pp.get('global_path', [])],
                dwa_traj=[(float(p[0]), float(p[1]), float(p[2])) for p in pp.get('dwa_traj', [])],
                p_traj=[(float(p[0]), float(p[1]), float(p[2])) for p in pp.get('p_traj', [])],
            )
        except Exception:
            planner_paths = PlannerPaths()
        planner_dc = PlannerData(
            paths=planner_paths,
            occupancy=occ_dc,
            costmap=cost_dc,
            laser_points=laser_points_dc,
            robot_pose=robot_pose,
            target_pos=target_pos,
        )

        reward_value = float(selected_robot.get('reward', 0.0))
        success_flag = bool(selected_robot.get('success', False))
        collision_flag = bool(selected_robot.get('collision', False))

        agent_state = selected_robot.get('agent_state')
        agent_action = selected_robot.get('agent_action')
        agent_reward = selected_robot.get('agent_reward')
        agent_image = selected_robot.get('agent_image')

        robots_enriched: List[Dict[str, Any]] = []
        if robots_data:
            def _planner_payload_from_paths(
                pp_dict: Optional[Dict[str, Any]],
                robot_pos: Tuple[float, float],
                robot_yaw_val: float,
                target_pos_robot: Optional[Tuple[float, float]],
                laser_points: Optional[Any],
                occ_entry: Optional[OccupancyGridData],
                cost_entry: Optional[CostmapData],
                navmesh_entry: Optional[NavmeshData],
            ) -> Dict[str, Any]:
                """Build planner-map payload for one robot monitor card.

                Args:
                    pp_dict: Planner path dictionary with global and local trajectories.
                    robot_pos: Robot XY position.
                    robot_yaw_val: Robot yaw angle.
                    target_pos_robot: Target XY position when available.
                    laser_points: Optional projected laser points.
                    occ_entry: Optional occupancy dataclass.
                    cost_entry: Optional costmap dataclass.
                    navmesh_entry: Optional navmesh dataclass.

                Returns:
                    Dictionary ready for monitor planner-map rendering.
                """
                payload: Dict[str, Any] = {
                    'robot_position': [float(robot_pos[0]), float(robot_pos[1])],
                    'robot_yaw': float(robot_yaw_val),
                    'target_position': list(target_pos_robot) if target_pos_robot is not None else None,
                }
                # Prefer per-robot planner paths, fall back to last monitor-level paths if missing
                gp = pp_dict.get('global_path') if pp_dict is not None else None
                dwa = pp_dict.get('dwa_traj') if pp_dict is not None else None
                pt = pp_dict.get('p_traj') if pp_dict is not None else None
                payload['global_path'] = gp if gp is not None else []
                payload['dwa_traj'] = dwa if dwa is not None else []
                payload['p_traj'] = pt if pt is not None else []
                if laser_points is not None:
                    payload['laser_points'] = laser_points
                if occ_entry is not None:
                    payload['occupancy'] = {
                        'grid': np.array(occ_entry.grid, copy=True),
                        'resolution': occ_entry.resolution,
                        'origin': list(occ_entry.origin_xy),
                    }
                if cost_entry is not None:
                    payload['costmap'] = {
                        'costmap': np.array(cost_entry.grid, copy=True),
                        'resolution': cost_entry.resolution,
                        'origin': list(cost_entry.origin_xy),
                    }
                if navmesh_entry is not None:
                    payload['navmesh'] = {
                        'vertices': np.array(navmesh_entry.vertices, copy=True),
                        'indices': np.array(navmesh_entry.triangles, copy=True),
                    }
                return payload

            for r_idx, robot_entry in enumerate(robots_data):
                if not isinstance(robot_entry, dict):
                    robots_enriched.append(robot_entry)
                    continue
                enriched = dict(robot_entry)
                if 'robot_name' not in enriched:
                    enriched['robot_name'] = f"Robot_{r_idx + 1}"
                # Per-robot occupancy/cost/navmesh from the robot entry
                occ_entry, cost_entry = self._occupancy_dataclass(enriched)
                navmesh_entry = self._navmesh_dataclass(enriched)
                if occ_entry is not None:
                    enriched['occupancy'] = {
                        'grid': np.array(occ_entry.grid, copy=True),
                        'resolution': occ_entry.resolution,
                        'origin': list(occ_entry.origin_xy),
                    }
                if cost_entry is not None:
                    enriched['costmap'] = {
                        'costmap': np.array(cost_entry.grid, copy=True),
                        'resolution': cost_entry.resolution,
                        'origin': list(cost_entry.origin_xy),
                    }
                if navmesh_entry is not None:
                    enriched['navmesh'] = {
                        'vertices': np.array(navmesh_entry.vertices, copy=True),
                        'indices': np.array(navmesh_entry.triangles, copy=True),
                    }

                robot_pose_xy = enriched.get('robot_position', [0.0, 0.0])
                robot_yaw_val = float(enriched.get('robot_yaw', 0.0))
                try:
                    rp_x = float(robot_pose_xy[0]) if len(robot_pose_xy) >= 1 else 0.0
                    rp_y = float(robot_pose_xy[1]) if len(robot_pose_xy) >= 2 else 0.0
                except Exception:
                    rp_x, rp_y = 0.0, 0.0

                rp_paths = enriched.get('planner_paths') if isinstance(enriched.get('planner_paths'), dict) else None
                # Per-robot laser points from that robot's cache
                l_scan, l_pts = self._laser_dataclass(enriched)
                laser_points_override = l_pts.as_list() if l_pts is not None else None

                target_pos_robot = None
                target_pos_list = enriched.get('target_position')
                if isinstance(target_pos_list, (list, tuple)) and len(target_pos_list) >= 2:
                    target_pos_robot = (float(target_pos_list[0]), float(target_pos_list[1]))

                enriched['planner_map'] = _planner_payload_from_paths(
                    rp_paths,
                    (rp_x, rp_y),
                    robot_yaw_val,
                    target_pos_robot,
                    laser_points_override,
                    occ_entry,
                    cost_entry,
                    navmesh_entry,
                )

                robots_enriched.append(enriched)
        else:
            robots_enriched = robots_data

        return MonitorPayload(
            robot_pose=robot_pose,
            robot_velocity=(linear_velocity, angular_velocity),
            target_delta=target_delta,
            reward=reward_value,
            success=success_flag,
            collision=collision_flag,
            laser_scan=laser_scan_dc,
            laser_points=laser_points_dc,
            occupancy=occ_dc,
            costmap=cost_dc,
            navmesh=navmesh_dc,
            planner=planner_dc,
            target_pos=target_pos,
            agent_state=agent_state,
            agent_action=agent_action,
            agent_reward=agent_reward,
            agent_image=agent_image,
            robots=robots_enriched if robots_enriched else None,
        )

    def _occupancy_dataclass(self, robot_entry: Optional[Dict[str, Any]]) -> Tuple[Optional[OccupancyGridData], Optional[CostmapData]]:
        """Build occupancy and costmap dataclasses from one robot cache entry.

        Args:
            robot_entry: Per-robot monitor cache dictionary.

        Returns:
            Tuple of occupancy dataclass and costmap dataclass.
        """
        if not isinstance(robot_entry, dict):
            return None, None

        occ_entry = robot_entry.get('occupancy')
        cost_entry = robot_entry.get('costmap')
        occ_payload = occ_entry if isinstance(occ_entry, dict) else None
        cost_payload = cost_entry if isinstance(cost_entry, dict) else None

        occ = None
        if occ_payload is not None:
            try:
                grid = np.asarray(occ_payload['grid'], dtype=float)
                if grid.ndim == 2 and grid.size > 0:
                    origin = occ_payload['origin']
                    occ = OccupancyGridData(
                        grid=grid,
                        resolution=float(occ_payload['resolution']),
                        origin_xy=(float(origin[0]), float(origin[1])),
                    )
            except Exception:
                occ = None

        cost = None
        if cost_payload is not None:
            try:
                cost_grid = np.asarray(cost_payload['costmap'], dtype=float)
                if cost_grid.ndim == 2 and cost_grid.size > 0:
                    origin = cost_payload['origin']
                    cost = CostmapData(
                        grid=cost_grid,
                        resolution=float(cost_payload['resolution']),
                        origin_xy=(float(origin[0]), float(origin[1])),
                    )
            except Exception:
                cost = None
        return occ, cost

    def _navmesh_dataclass(self, robot_entry: Optional[Dict[str, Any]]) -> Optional[NavmeshData]:
        """Build navmesh dataclass from one robot cache entry.

        Args:
            robot_entry: Per-robot monitor cache dictionary.

        Returns:
            NavmeshData when valid navmesh data exists; otherwise None.
        """
        if not isinstance(robot_entry, dict):
            return None
        navmesh_entry = robot_entry.get('navmesh')
        navmesh_payload = navmesh_entry if isinstance(navmesh_entry, dict) else None
        if navmesh_payload is None:
            return None
        try:
            vertices = np.asarray(navmesh_payload['vertices'], dtype=float)
            indices = np.asarray(navmesh_payload['indices'], dtype=int)
            if vertices.size == 0 or indices.size == 0:
                return None
            return NavmeshData(vertices=vertices, triangles=indices)
        except Exception:
            return None

    def _laser_dataclass(self, robot_entry: Optional[Dict[str, Any]]) -> Tuple[Optional[LaserScanData], Optional[LaserPoints]]:
        """Build laser scan and projected points from one robot cache entry.

        Args:
            robot_entry: Per-robot monitor cache dictionary.

        Returns:
            Tuple of LaserScanData and LaserPoints; None values when unavailable.
        """
        if not isinstance(robot_entry, dict):
            return None, None

        # Laser payload is optional when AMR laser is disabled; do not fail monitor updates.
        laser_scan_raw = robot_entry.get('laser_scan')
        if laser_scan_raw is None:
            return None, None

        ranges = np.asarray(laser_scan_raw, dtype=np.float32).reshape(-1)
        if ranges.size == 0:
            return None, None

        if 'laser_angle_min' not in robot_entry or 'laser_angle_max' not in robot_entry or 'laser_max_range' not in robot_entry:
            return None, None

        angle_min = float(robot_entry['laser_angle_min'])
        angle_max = float(robot_entry['laser_angle_max'])
        max_range = float(robot_entry['laser_max_range'])

        angle_offset = - (angle_min + angle_max)
        angles = np.linspace(angle_min, angle_max, ranges.size, dtype=float) + angle_offset

        if 'robot_position' not in robot_entry or 'robot_yaw' not in robot_entry:
            return None, None

        robot_pos_xy = robot_entry['robot_position']
        if not isinstance(robot_pos_xy, (list, tuple, np.ndarray)) or len(robot_pos_xy) < 2:
            return None, None
        robot_x = float(robot_pos_xy[0])
        robot_y = float(robot_pos_xy[1])
        robot_yaw = float(robot_entry['robot_yaw'])

        sensor_offset = robot_entry.get('laser_sensor_offset', (0.0, 0.0))
        if not isinstance(sensor_offset, (list, tuple, np.ndarray)) or len(sensor_offset) < 2:
            ox, oy = 0.0, 0.0
        else:
            ox = float(sensor_offset[0])
            oy = float(sensor_offset[1])

        dir_x = np.cos(robot_yaw + angles)
        dir_y = np.sin(robot_yaw + angles)

        offset_world_x = ox * np.cos(robot_yaw) - oy * np.sin(robot_yaw)
        offset_world_y = ox * np.sin(robot_yaw) + oy * np.cos(robot_yaw)

        sensor_x = robot_x + offset_world_x
        sensor_y = robot_y + offset_world_y

        xs = ranges * dir_x + sensor_x
        ys = ranges * dir_y + sensor_y
        points = np.stack([xs, ys], axis=1) if ranges.size else np.zeros((0, 2), dtype=float)

        scan = LaserScanData(
            ranges=ranges,
            angle_min=angle_min,
            angle_max=angle_max,
            max_range=max_range,
            sensor_offset_xy=(ox, oy),
            frame_id="laser",
        )
        laser_points = LaserPoints(
            points_xy=points,
            origin_xy=(sensor_x, sensor_y),
            angle_offset_applied=angle_offset,
        )
        return scan, laser_points

    def _laser_points_from_scan(
        self,
        ranges: np.ndarray,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        angle_min: float,
        angle_max: float,
        sensor_offset: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """Project laser ranges into world-frame XY points.

        Args:
            ranges: Range array for one scan.
            robot_x: Robot x position in world coordinates.
            robot_y: Robot y position in world coordinates.
            robot_yaw: Robot heading in radians.
            angle_min: Laser minimum angle in radians.
            angle_max: Laser maximum angle in radians.
            sensor_offset: Laser sensor offset in robot frame as (x, y).

        Returns:
            List of projected (x, y) points.
        """
        if ranges is None:
            return []
        r = np.asarray(ranges, dtype=float).reshape(-1)
        if r.size == 0:
            return []

        angle_offset = - (angle_min + angle_max)
        angles = np.linspace(angle_min, angle_max, r.size, dtype=float) + angle_offset

        try:
            ox, oy = float(sensor_offset[0]), float(sensor_offset[1])
        except Exception:
            ox, oy = 0.0, 0.0

        dir_x = np.cos(robot_yaw + angles)
        dir_y = np.sin(robot_yaw + angles)

        offset_world_x = ox * np.cos(robot_yaw) - oy * np.sin(robot_yaw)
        offset_world_y = ox * np.sin(robot_yaw) + oy * np.cos(robot_yaw)

        sensor_x = robot_x + offset_world_x
        sensor_y = robot_y + offset_world_y

        xs = r * dir_x + sensor_x
        ys = r * dir_y + sensor_y
        pts = np.stack([xs, ys], axis=1) if r.size else np.zeros((0, 2), dtype=float)
        return [(float(px), float(py)) for px, py in pts]

    def _unity_retrieve_observation_numeric(self, robot_payload: Dict[str, Any], env_payload: Dict[str, Any]) -> None:
        """Parse Unity numeric payloads and update cached observation fields.

        Args:
            robot_payload: Robot-level payload dictionary from Unity.
            env_payload: Environment-level payload dictionary from Unity.
        """
        robot_numeric = robot_payload['Numeric']
        env_numeric = env_payload['Numeric']

        base_pose = robot_numeric['BasePose']
        base_twist = robot_numeric['BaseTwist']
        target_pose = robot_numeric['TargetPose']

        items = env_numeric['Items']
        item_entries: List[Dict[str, Any]] = []
        for item in items:
            item_pose = item['Pose']
            item_pos = np.asarray(item_pose['Position'], dtype=np.float32)
            item_q = np.asarray(item_pose['Rotation'], dtype=np.float32)
            item_yaw = self._yaw_from_wxyz(item_q[0], item_q[1], item_q[2], item_q[3])
            item_entries.append({
                'position': item_pos,
                'orientation_quat': item_q,
                'yaw': item_yaw,
            })

        robot_pos = np.asarray(base_pose['Position'], dtype=np.float32)
        robot_q = np.asarray(base_pose['Rotation'], dtype=np.float32)
        linear = base_twist['Linear']
        angular = base_twist['Angular']

        target_pos = np.asarray(target_pose['Position'], dtype=np.float32)
        target_q = np.asarray(target_pose['Rotation'], dtype=np.float32)

        robot_yaw = self._yaw_from_wxyz(robot_q[0], robot_q[1], robot_q[2], robot_q[3])
        target_yaw = self._yaw_from_wxyz(target_q[0], target_q[1], target_q[2], target_q[3])

        self.unity_observation['robot_position'] = robot_pos
        self.unity_observation['robot_orientation_quat'] = robot_q
        self.unity_observation['robot_linear_velocity'] = float(linear[0]) if len(linear) > 0 else 0.0
        self.unity_observation['robot_angular_velocity'] = float(angular[2]) if len(angular) > 2 else 0.0
        self.unity_observation['target_position'] = target_pos
        self.unity_observation['target_orientation_quat'] = target_q
        self.unity_observation['items'] = item_entries
        self.unity_observation['items_position'] = [entry['position'] for entry in item_entries]
        self.unity_observation['items_orientation_quat'] = [entry['orientation_quat'] for entry in item_entries]
        self.unity_observation['collision_flag'] = 1.0 if bool(robot_numeric['Collided']) else 0.0
        self.unity_observation['robot_yaw'] = robot_yaw
        self.unity_observation['target_yaw'] = target_yaw
        self.unity_observation['items_yaw'] = [float(entry['yaw']) for entry in item_entries]

    def _unity_retrieve_observation_navmesh(self, payload: Any) -> Dict[str, Any]:
        """Parse Unity navmesh payload into vertices and triangle indices.

        Args:
            payload: Raw navmesh payload object from Unity.

        Returns:
            Dictionary with 'vertices' and 'indices', or empty dictionary on failure.
        """
        if payload is None or not isinstance(payload, dict):
            return {}

        if 'Vertices' not in payload or 'Indices' not in payload:
            return {}
        raw_vertices = payload['Vertices']
        raw_indices = payload['Indices']

        try:
            vertices = np.asarray(raw_vertices, dtype=np.float32)
            indices = np.asarray(raw_indices, dtype=np.int32).flatten()
        except Exception:
            return {}

        if vertices.size == 0 or indices.size == 0 or indices.size % 3 != 0:
            return {}

        # Unity now sends 2D vertices (x, y pairs). Check % 2 first to avoid
        # misinterpreting 2N floats as 3-component vectors when N % 3 == 0.
        if vertices.size % 2 == 0:
            vertices = vertices.reshape(-1, 2)
        elif vertices.size % 3 == 0:
            vertices = vertices.reshape(-1, 3)
        else:
            return {}

        return {
            'vertices': vertices,
            'indices': indices,
        }

    # Image format magic-byte prefixes used to distinguish raw image data
    # from base64-encoded strings arriving via the JSON transport.
    _IMAGE_MAGIC = (
        b'\x89PNG',       # PNG
        b'\xff\xd8\xff',  # JPEG / JFIF / EXIF
        b'GIF8',          # GIF87a / GIF89a
        b'BM',            # BMP
        b'RIFF',          # WebP (RIFF container)
    )

    def _unity_retrieve_observation_images(self, raw_image_payload: Any) -> List[bytes]:
        """Decode Unity camera payload values into raw image byte frames.

        Handles both base64-encoded strings (JSON transport) and raw binary
        bytes (MessagePack / binary transport) transparently.

        Args:
            raw_image_payload: Raw image payload value or collection.

        Returns:
            List of decoded image frames as bytes.
        """

        frames: List[bytes] = []

        def _append_candidate(candidate: Any) -> None:
            """Try to decode one candidate image payload entry.

            Args:
                candidate: Candidate image data in bytes/string form.
            """
            if candidate in (None, "", b""):
                return

            # Raw bytes that already start with a known image header
            # do NOT need base64 decoding (binary / MsgPack transport).
            if isinstance(candidate, (bytes, bytearray, memoryview)):
                raw = bytes(candidate)
                if not raw:
                    return
                if any(raw.startswith(magic) for magic in self._IMAGE_MAGIC):
                    frames.append(raw)
                    return
                # Not a recognised image header — try base64 decode.
                try:
                    decoded = base64.b64decode(raw, validate=True)
                    if decoded:
                        frames.append(decoded)
                        return
                except Exception:
                    pass
                # Last resort: treat the raw bytes as image data directly.
                frames.append(raw)
                return

            # String path (JSON transport): always base64-encoded.
            try:
                base64_bytes = str(candidate).encode('ascii')
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

        return frames

    @staticmethod
    def _yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
        """Extract yaw about the ROS Z-axis from quaternion components.

        Args:
            w: Quaternion scalar component.
            x: Quaternion x component.
            y: Quaternion y component.
            z: Quaternion z component.

        Returns:
            Yaw angle in radians (ROS convention).
        """
        # Standard yaw around Z-axis from quaternion in ROS (right-handed, Z-up).
        # Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _planar_pose_to_payload(pose_xy_yaw: Sequence[float], height: float) -> List[float]:
        """Convert a ROS planar pose [x, y, yaw] to a 6-DOF payload in ROS convention.

        All values remain in DART/ROS convention (right-handed, Z-up, radians).
        Unity handles coordinate conversion internally via CoordinateConverter.

        Args:
            pose_xy_yaw: Input planar pose [x, y, yaw] in ROS frame (yaw in radians).
            height: Height above the ground plane (z in ROS Z-up convention).

        Returns:
            6-element list [x, y, z, rx, ry, rz] in ROS convention (radians).
        """
        x = float(pose_xy_yaw[0])
        y = float(pose_xy_yaw[1])
        yaw = float(pose_xy_yaw[2])
        return [x, y, float(height), 0.0, 0.0, yaw]

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Normalize an angle to the interval [-pi, pi).

        Args:
            a: Input angle in radians.

        Returns:
            Wrapped angle in radians.
        """
        return (a + np.pi) % (2 * np.pi) - np.pi

    def get_wrapper_attr(self, name: str) -> Any:
        """Expose attributes through a VecEnvWrapper-compatible API.

        Args:
            name: Name of the attribute to retrieve.

        Returns:
            Attribute value for the given name.

        Raises:
            AttributeError: Raised when the attribute does not exist.
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def get_monitor_image(self) -> Optional[Dict[str, Any]]:
        """Return cached monitor image payload including frame labels.

        Returns:
            Dictionary with image frames and labels, or None when unavailable.
        """
        if not self._unity_image_bytes_list:
            return None
        payload: Dict[str, Any] = {
            'frames': list(self._unity_image_bytes_list)
        }
        if self._unity_image_labels:
            payload['labels'] = list(self._unity_image_labels)
        return payload
