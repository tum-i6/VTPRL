"""
A minimal Warehouse Unity Gym environment compatible with SimulatorVecEnv.
Numeric-only observations; optional laser scan with fixed length.
"""
from typing import Dict, Any
import numpy as np
import gym
from gym import spaces


class WarehouseUnityEnv(gym.Env):
    """
    WarehouseUnityEnv: A Gym environment for warehouse robot navigation using Unity as the simulator.
    Closely mirrors the code style and conventions of IiwaDartUnityEnv for consistency.
    """
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, max_time_steps: int, env_id: int, config: Dict[str, Any]):
        """
        Initialize the WarehouseUnityEnv.
        :param max_time_steps: Maximum number of steps per episode.
        :param env_id: Environment ID (for vectorized use).
        :param config: Configuration dictionary.
        """
        super().__init__()
        self.id = env_id
        self.max_time_steps = int(max_time_steps)
        self.config = config

        # Action space: [v, omega]
        self.max_linear_velocity = float(config.get('max_v', 1.0))
        self.max_angular_velocity = float(config.get('max_omega', 1.0))
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_velocity, -self.max_angular_velocity], dtype=np.float32),
            high=np.array([self.max_linear_velocity, self.max_angular_velocity], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: [x, y, yaw, v, omega, dx, dy, dyaw] + optional laser[N]
        self.use_laser_scan = bool(config.get('use_laser_scan', False))
        self.laser_count = int(config.get('laser_count', 0)) if self.use_laser_scan else 0
        self.base_observation_dim = 8
        self.observation_dim = self.base_observation_dim + (self.laser_count if self.use_laser_scan else 0)
        obs_high = np.array([np.inf] * self.observation_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

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
        self.normalize_observation = bool(config.get('normalize_obs', False))
        self.position_normalization = float(config.get('pos_norm', 10.0))
        self.yaw_normalization = float(config.get('yaw_norm', np.pi))
        self.success_distance_threshold = float(config.get('success_distance_threshold', 0.1))
        self.success_yaw_threshold = float(config.get('success_yaw_threshold', np.deg2rad(10.0)))

        # Unity observation cache
        self.unity_observation = {}

    def seed(self, seed: int = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed=seed)
        return [seed]

    def reset(self):
        """
        Reset the environment state for a new episode.
        """
        
        # For RESET command (Unity expects a flat array like IiwaDartUnityEnv):
        # reset_state = active_joints + joint_positions + joint_velocities + target_positions_mapped + object_positions_mapped + robot_positions_mapped
        # Where lengths are: n_joints (active), n_joints (positions), n_joints (velocities), 6 (target pose xyz/rotation), 6 (object pose xyz/rotation), 6 (robot pose xyz/rotation)
        self.n_joints = int(self.config.get('num_joints', 3))

        # Active joints: mark all as active by default
        active_joints = [1.0] * self.n_joints

        # Initial joint positions and velocities (pad/truncate to n_joints)
        init_pos = list(self.config.get('initial_positions', [0.0] * self.n_joints))
        if len(init_pos) < self.n_joints:
            init_pos += [0.0] * (self.n_joints - len(init_pos))
        elif len(init_pos) > self.n_joints:
            init_pos = init_pos[:self.n_joints]

        init_vel = list(self.config.get('initial_velocities', [0.0] * self.n_joints))
        if len(init_vel) < self.n_joints:
            init_vel += [0.0] * (self.n_joints - len(init_vel))
        elif len(init_vel) > self.n_joints:
            init_vel = init_vel[:self.n_joints]

        # Target and object poses already in Unity frame (x,y,z,rx,ry,rz). Keep length 6 per plan.
        target_pose_unity = list(self.config.get('target_pose_unity', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        target_pose_unity[0] += np.random.uniform(-1.0, 1.0)
        target_pose_unity[2] += np.random.uniform(-1.0, 1.0)

        object_pose_unity = list(self.config.get('object_pose_unity', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        object_pose_unity[0] += np.random.uniform(-1.0, 1.0)
        object_pose_unity[2] += np.random.uniform(-1.0, 1.0)

        robot_pose_unity = list(self.config.get('robot_pose_unity', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        robot_pose_unity[0] += np.random.uniform(-1.0, 1.0)
        robot_pose_unity[2] += np.random.uniform(-1.0, 1.0)

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
        return self._last_observation.copy()

    def step(self, action: np.ndarray):
        """
        Return the last computed state, reward, done, and info (for Gym VecEnv compatibility).
        """
        reward = self._compute_reward()
        info = {"success": self.success, "collision": self.collision}
        return self._last_observation.copy(), reward, self._done, info

    def update(self, unity_observation_dict: Dict[str, Any], time_step_update: bool = True):
        """
        Update the environment state from Unity's observation.
        :param unity_observation_dict: Dictionary with keys 'Observation' (required) and 'LaserScan' (optional).
        :param time_step_update: Whether to increment the time step.
        :return: (state, reward, done, info)
        """
        # Parse numeric observation from Unity
        observation_vector = np.array(unity_observation_dict['Observation'], dtype=np.float32)
        self._unity_retrieve_observation_numeric(observation_vector)

        # Build agent observation [x, y, yaw, v, omega, dx, dy, dyaw]
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

        observation_base = np.array([robot_x, robot_y, robot_yaw, robot_linear_velocity, robot_angular_velocity,
                                     delta_x, delta_y, delta_yaw], dtype=np.float32)
        if self.normalize_observation:
            observation_base[0] /= self.position_normalization
            observation_base[1] /= self.position_normalization
            observation_base[2] /= self.yaw_normalization
            observation_base[3] /= max(self.action_space.high[0], 1e-6)
            observation_base[4] /= max(self.action_space.high[1], 1e-6)
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

        return self._last_observation.copy(), self._compute_reward(), self._done, {"success": self.success, "collision": self.collision}

    def render(self, mode='human'):
        return None

    def close(self):
        return None

    # Wrapper compatibility helper
    def get_terminal_reward(self) -> bool:
        return bool(self.success)

    # Simple helpers used by evaluation code
    def random_action(self) -> np.ndarray:
        return self.action_space.sample()

    def action_by_p_control(self, k_v: float = 1.0, k_w: float = 1.0) -> np.ndarray:
        # Use goal deltas from observation to compute a simple proportional control
        # dx = float(self._last_observation[5])
        # dy = float(self._last_observation[6])
        # dyaw = float(self._last_observation[7])
        # if self.normalize_observation:
        #     dx *= self.pos_norm
        #     dy *= self.pos_norm
        #     dyaw *= self.yaw_norm
        # v_cmd = k_v * float(np.hypot(dx, dy))
        # w_cmd = k_w * dyaw
        # v_cmd = float(np.clip(v_cmd, -self.action_space.high[0], self.action_space.high[0]))
        # w_cmd = float(np.clip(w_cmd, -self.action_space.high[1], self.action_space.high[1]))

        v_cmd = 0.1
        w_cmd = 0.1
        return np.array([v_cmd, w_cmd], dtype=np.float32)

    def _compute_reward(self) -> float:
        step_penalty = float(self.config.get('step_penalty', 0.01))
        success_reward = float(self.config.get('success_reward', 1.0))
        collision_penalty = float(self.config.get('collision_penalty', 1.0))
        distance_weight = float(self.config.get('distance_weight', 0.0))
        yaw_weight = float(self.config.get('yaw_weight', 0.0))

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

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
        """Extract yaw (rotation about Y) from a w,x,y,z quaternion (Unity convention)."""
        # Standard yaw around Y-axis
        # Reference: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _unity_retrieve_observation_numeric(self, observation_vector_unity: np.ndarray):
        """
        Parse Unity's Observation float vector and populate self.unity_observation dict.
        Uses the last 24 values as the fixed warehouse schema, similar to IiwaDartUnityEnv conventions.
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
        self.unity_observation['robot_yaw'] = self._yaw_from_wxyz(*self.unity_observation['robot_orientation_quat'])
        self.unity_observation['target_yaw'] = self._yaw_from_wxyz(*self.unity_observation['target_orientation_quat'])

    def get_wrapper_attr(self, name):
        """Mimics Stable-Baselines3 VecEnvWrapper method."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
