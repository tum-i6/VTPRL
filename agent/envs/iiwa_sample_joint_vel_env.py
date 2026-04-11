"""
A sample Env class inheriting from basic gym.Env for the Kuka LBR iiwa manipulator with 7 links and a Gripper.

Important: There is no inverse kinematics calculation supported here (see iiwa_sample_env for IK), only joint velocity control is supported with it.

Unity is used as the main simulator for physics/rendering computations. The Unity interface receives joint velocities as commands
and returns joint positions and velocities. The class presents a way to alternate between numeric observations and image observations,
how to parse images returned from the Unity simulator and how to reset the environments in the simulator
"""

import cv2
import base64

import numpy as np
from numpy import newaxis

from gym import spaces, core
from gym.utils import seeding
from typing import Any, Dict, List
from utils.config_utils import (
    expand_item_instances,
    resolve_item_poses_unity,
    resolve_robot_poses_unity,
    resolve_target_poses_unity,
    transform_local_pose_to_world,
)
from utils.telemetry import MonitorPayload, Pose2D

class IiwaJointVelEnv(core.Env):
    # the max velocity allowed for a joint in rad, used for normalization of the state values related to angle velocity
    MAX_VEL = 1.1  # value used to normalize the speed returned from the
    # values used to normalize the  angle positions of Kuka
    MAX_ANGLE_1 = 3
    MAX_ANGLE_2 = 2.1
    # assumed max distance between target and end-effector, used for normalizing the agent state in range (-1,1)
    MAX_DISTANCE = 2

    # min distance to declare that the target is reached by the end-effector, adapt the values based on your task
    MIN_POS_DISTANCE = 0.05  # [m]

    def __init__(self, id, max_ts, config):
        self.max_ts = max_ts
        self.ts = 0

        self.use_images = config["use_images"]
        self.image_size = config["image_size"]

        # relevant for numeric observations: whether only angles (a) or angles and velocities (av) of the joints
        self.state_type = config["state"]
        self.simulator_observation = None
        self.id = id
        self.reset_counter = 0
        self.reset_state = None
        self.first_reset_done = False

        self.collided_env = 0
        self.seed()

        # number of joints to be controlled from the 7 available
        self.num_joints = config["num_joints"]
        self.manipulator_config = config.get("manipulator_config", {}) if isinstance(config, dict) else {}
        self.manipulator_gym_config = config.get("manipulator_gym_config", {}) if isinstance(config, dict) else {}
        pose_cfg = self.manipulator_gym_config if self.manipulator_gym_config else self.manipulator_config
        self.robot_poses_unity = resolve_robot_poses_unity(pose_cfg, default_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.robot_count = len(self.robot_poses_unity)
        self.item_entries = expand_item_instances(self.manipulator_config)
        self._per_robot_success = [False] * self.robot_count

        # dx, dy, dz placeholders in the state for the distance between the end-effector and the target
        state_array_limits = [1., 1., 1.]
        # angle value placeholders
        state_array_limits.extend([1] * self.num_joints)
        if config["state"] == 'av':
            # add angle velocity placeholders
            state_array_limits.extend([1] * self.num_joints)

        high = np.array(state_array_limits)
        low = -high

        # Define a gym observation space suitable for images #
        if self.use_images:
            # if using images make self.image_sizexself.image_size pixels grayscale observation space
            self.observation_space = spaces.Box(low=0, high=255,
                                                shape=(self.image_size, self.image_size, 1),
                                                dtype=np.uint8)
        else:
            low_multi = np.tile(low, self.robot_count)
            high_multi = np.tile(high, self.robot_count)
            self.observation_space = spaces.Box(low=low_multi, high=high_multi, dtype=np.float32)

        # the action space is in the range [-1, 1] for the controllable joints and a 0 for the gripper opening as we
        # do not need to control the gripper for the reaching task
        temp = [1] * self.num_joints
        temp.append(0)
        high = np.array(temp, dtype=np.float32)
        low = -high
        action_high = np.tile(high, (self.robot_count, 1))
        action_low = -action_high
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.reward_range = (-5.0, 0.0)

    def update(self, observation, time_step_update=True):
        """
            a replacement of the standart step() method used in OpenAI gym. Unlike typical gym environment where the step()
            function is called at each timestep, in this case the simulator_vec_env step() function is called during
            training, which in turn calls the update() function here for a single env to pass its simulator observation.
            Reasons are related to the logic of how to communicate with the simulator and how to reset individual
            environments running in the simulator independently.
        """
        if(time_step_update == True):
            self.ts += 1

        robots = observation.get('Robots', []) if isinstance(observation, dict) else []
        if not robots:
            raise ValueError("Unity observation payload missing 'Robots'.")
        if len(robots) != self.robot_count:
            raise ValueError(f"Robot payload count mismatch: expected {self.robot_count}, got {len(robots)}.")

        robots_by_index = {}
        for list_idx, payload in enumerate(robots):
            if not isinstance(payload, dict):
                continue
            try:
                ridx = int(payload.get('RobotIndex', list_idx))
            except Exception:
                ridx = list_idx
            if ridx not in robots_by_index:
                robots_by_index[ridx] = payload

        env_numeric = observation.get('Numeric', {}) if isinstance(observation, dict) else {}

        per_robot_states = []
        per_robot_rewards = []
        per_robot_success = []
        collided_any = False
        self._per_robot_obs_cache = []
        self._per_robot_states_cache = []
        self._per_robot_rewards_cache = []
        for idx in range(self.robot_count):
            robot_payload = robots_by_index.get(idx)
            if robot_payload is None and idx < len(robots):
                candidate = robots[idx]
                if isinstance(candidate, dict):
                    robot_payload = candidate
            if robot_payload is None:
                raise ValueError(f"Missing robot payload for robot index {idx}.")

            robot_numeric = robot_payload.get('Numeric', {}) if isinstance(robot_payload, dict) else {}
            obs_vec = self._build_observation_vector(robot_numeric, env_numeric, idx)
            self._convert_observation(obs_vec)
            per_robot_states.append(np.asarray(self.state, dtype=np.float32))
            reward_i = self._get_reward()
            per_robot_rewards.append(float(reward_i))
            per_robot_success.append(bool(self._get_distance() < self.MIN_POS_DISTANCE))
            collided_any = collided_any or bool(self.collided_env)

            # Cache per-robot snapshots for get_monitor_payload()
            self._per_robot_obs_cache.append(obs_vec.copy())
            self._per_robot_states_cache.append(np.asarray(self.state, dtype=np.float32))
            self._per_robot_rewards_cache.append(float(reward_i))

        self.state = np.concatenate(per_robot_states, axis=0)
        self.collided_env = int(collided_any)
        self._per_robot_success = per_robot_success

        info = {
            "success": bool(any(per_robot_success)),
            "robot_count": self.robot_count,
            "per_robot_rewards": per_robot_rewards,
        } # Episode was successful. For now it is set at simulator_vec_env.py before the reset. Adapt if needed
        terminal = self._get_terminal()
        reward = float(np.sum(per_robot_rewards))

        if self.use_images:
            overhead_images = observation.get('OverheadImages', None)
            image_list = []
            if overhead_images is not None:
                for entry in overhead_images:
                    if isinstance(entry, dict):
                        image_list.append(entry.get('Data'))
                    else:
                        image_list.append(entry)

            for robot_payload in robots:
                if not isinstance(robot_payload, dict):
                    continue
                robot_image = robot_payload.get('RobotImage', {})
                if isinstance(robot_image, dict):
                    robot_image_data = robot_image.get('Data')
                    if robot_image_data is not None:
                        image_list.append(robot_image_data)
            return self._retrieve_image({'ImageData': image_list}), reward, terminal, info
        else:
            return self.state, reward, terminal, info

    def _get_reward(self):
        """
            example definition of a reward function for a specific task
        """
        absolute_distance = self._get_distance()

        # in this case the reward is the negative distance between the target and the end-effector and a punishment
        # of 1 for collision. This reward is suitable for training reaching tasks
        return - 1 * (absolute_distance + self.collided_env)

    def get_terminal_reward(self):
        """
           checks if the target is reached

           _get_distance(): returns norm of the Euclidean error from the end-effector to the target position

           Important: by default a 0.0 value of a terminal reward will be given to the agent. To adapt it please refer to the config.py,
                      in the reward_dict. This terminal reward is given to the agent during the step() function in the simulator_vec_env.py

           :return: a boolean value representing if the target is reached within the defined threshold
        """
        return bool(any(self._per_robot_success))

    def _convert_observation(self, new_observation):
        """
            method used for creating task-specific agent state from the generic simulator observation returned.

            The simulator observation has the following array of 34 values:
                [a1, a2, a3, a4, a5, a6, a7,
                 v1, v2, v3, v4, v5, v6, v7,
                 ee_x, ee_y, ee_z ee_rx, ee_ry, ee_rz,
                 t_x, t_y, t_z, t_rx, t_ry, t_rz,
                 o_x, o_y, o_z, o_rx, o_ry, o_rz,
                 g_p, c
                ]

            where
                - a1..a7 are the angles of each joint of the robot in radians,
                - v1..v7 are the velocities of each joint in rad/sec,
                - x, y, and z for ee, t and o are the coordinates and and rx, ry, and rz for ee, t and o are the
                  quaternion x, y and z components of the rotation for the end-effector, the target and the object(box) respectively
                - g_p is the position (opening) of the gripper and,
                - c is a collision flag (0 if no collision and 1 if a collision of any part of the robot with the floor happened)
        """

        # below is an example agent state suitable for solving the position reaching task
        self.simulator_observation = new_observation

        self.collided_env = self.simulator_observation[-1]
        self.gripper_position = self.simulator_observation[-2]

        self.target_x, self.target_y, self.target_z, _, _, _ = self._get_target_pose()
        self.object_x, self.object_y, self.object_z, _, _, _ = self._get_object_pose()
        self.ee_x, self.ee_y, self.ee_z, _, _, _ = self._get_end_effector_pose()
        self.joint_angles = self.simulator_observation[0 : self.num_joints]
        self.joint_speeds = self.simulator_observation[7 : 7 + self.num_joints]

        dx = (self.target_x - self.ee_x) / self.MAX_DISTANCE
        dy = (self.target_y - self.ee_y) / self.MAX_DISTANCE
        dz = (self.target_z - self.ee_z) / self.MAX_DISTANCE

        self.state = [dx, dy, dz]
        for i in range(self.num_joints):
            # add the angles for the enabled joints
            if i % 2 == 0:
                self.state.append(self.simulator_observation[i] / self.MAX_ANGLE_1)
            else:
                self.state.append(self.simulator_observation[i] / self.MAX_ANGLE_2)

        if self.state_type == 'av':
            for i in range(self.num_joints):
                # the second 7 values are joint velocities, if velocities are part of the state include them
                self.state.append(self.simulator_observation[7 + i] / self.MAX_VEL)

    def _build_observation_vector(self, robot_numeric, env_numeric, robot_index):
        joints = robot_numeric.get('Joints', []) if isinstance(robot_numeric, dict) else []
        joint_positions = [float(j.get('Position', 0.0)) for j in joints]
        joint_velocities = [float(j.get('Velocity', 0.0)) for j in joints]

        if len(joint_positions) < 7:
            joint_positions.extend([0.0] * (7 - len(joint_positions)))
        if len(joint_velocities) < 7:
            joint_velocities.extend([0.0] * (7 - len(joint_velocities)))

        ee_pose = robot_numeric.get('EndEffectorPose', {}) if isinstance(robot_numeric, dict) else {}
        target_pose = robot_numeric.get('TargetPose', {}) if isinstance(robot_numeric, dict) else {}

        items = env_numeric.get('Items', []) if isinstance(env_numeric, dict) else []
        selected_item = {}
        if isinstance(items, list) and items:
            item_idx = min(robot_index, len(items) - 1)
            selected_item = items[item_idx] if isinstance(items[item_idx], dict) else {}
        item_pose = selected_item.get('Pose', {}) if isinstance(selected_item, dict) else {}

        def _vec3(src):
            arr = src if isinstance(src, (list, tuple)) else []
            return [
                float(arr[0]) if len(arr) > 0 else 0.0,
                float(arr[1]) if len(arr) > 1 else 0.0,
                float(arr[2]) if len(arr) > 2 else 0.0,
            ]

        def _quat_xyz(src):
            arr = src if isinstance(src, (list, tuple)) else []
            return [
                float(arr[1]) if len(arr) > 1 else 0.0,
                float(arr[2]) if len(arr) > 2 else 0.0,
                float(arr[3]) if len(arr) > 3 else 0.0,
            ]

        ee_pos = _vec3(ee_pose.get('Position', []))
        ee_rot = _quat_xyz(ee_pose.get('Rotation', []))
        target_pos = _vec3(target_pose.get('Position', []))
        target_rot = _quat_xyz(target_pose.get('Rotation', []))
        item_pos = _vec3(item_pose.get('Position', []))
        item_rot = _quat_xyz(item_pose.get('Rotation', []))

        gripper_pos = float(robot_numeric.get('GripperPosition', 0.0))
        collision_flag = 1.0 if bool(robot_numeric.get('Collided', False)) else 0.0

        return np.array(
            joint_positions[:7] + joint_velocities[:7] +
            ee_pos + ee_rot +
            target_pos + target_rot +
            item_pos + item_rot +
            [gripper_pos, collision_flag],
            dtype=np.float32,
        )

    def _get_end_effector_pose(self):
        return \
            self.simulator_observation[14], self.simulator_observation[15], self.simulator_observation[16], \
            self.simulator_observation[17], self.simulator_observation[18], self.simulator_observation[19]

    def _get_object_pose(self):
        return \
            self.simulator_observation[26], self.simulator_observation[27], self.simulator_observation[28], \
            self.simulator_observation[29], self.simulator_observation[30], self.simulator_observation[31]

    def _get_target_pose(self):
        return \
            self.simulator_observation[20], self.simulator_observation[21], self.simulator_observation[22], \
            self.simulator_observation[23], self.simulator_observation[24], self.simulator_observation[25]

    def _get_distance(self):

        return \
            np.linalg.norm(np.array([self.target_x - self.ee_x, self.target_y - self.ee_y, self.target_z - self.ee_z]))

    def step(self):
        """
            not used directly on level on single environment,
            see update() method instead and step() of simulator_vec_env.py
        """
        pass

    def reset(self):
        """
            reset method that can set the environment in the simulator in a specific initial state and enable/disable joints
            The method defines the self.reset_state that is used from simulator_vec_env to send the reset values to the Unity simulator.

            The simulator reset state has the following array of 32 values:
                [j1, j2, 23, j4, j5, j6, j7,
                 a1, a2, a3, a4, a5, a6, a7,
                 v1, v2, v3, v4, v5, v6, v7,
                 t_x, t_y, t_z, t_rx, t_ry, t_rz,
                 o_x, o_y, o_z, o_rx, o_ry, o_rz,
                 g_p]

            where
                - j1..j7 are flags indicating whether a joint should be enabled (1) or disabled (0),
                - a1..a7 are the initial angles of each joint of the robot in radians (currently only 0 initial values supported
                  due to unity limitations)
                - v1..v7 are the initial velocities of each joint in rad/sec (currently only 0 initial values supported due to
                  unity limitations)
                - x, y, and z for t and o are the initial coordinates of the target and the object in meters
                  (ROS convention: x=forward, y=left, z=up)
                - rx, ry, and rz for t and o are the euler angles in radians for the rotation of the object and the target
                - g_p is the position (opening) of the gripper (0 is open, value up to 90 supported)
        """
        self.reset_counter += 1

        self.ts = 0
        self.collided_env = 0
        self._per_robot_success = [False] * self.robot_count
        self._per_robot_obs_cache = []
        self._per_robot_states_cache = []
        self._per_robot_rewards_cache = []

        default_target_pose = [-0.38, -0.9, 0.0, 0.0, -np.pi/2, 0.0]
        default_item_pose = [-1.0, 1.0, 1.0, 0.0, -np.pi/2, 0.0]

        pose_cfg = self.manipulator_gym_config if self.manipulator_gym_config else self.manipulator_config
        target_poses = resolve_target_poses_unity(pose_cfg, self.robot_count, default_target_pose)
        item_poses = resolve_item_poses_unity(pose_cfg, default_item_pose)
        if not item_poses:
            item_poses = [list(default_item_pose)]

        robot_poses = resolve_robot_poses_unity(pose_cfg, default_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if len(robot_poses) != self.robot_count:
            if len(robot_poses) == 1:
                robot_poses = [list(robot_poses[0]) for _ in range(self.robot_count)]
            else:
                raise ValueError(f"Expected {self.robot_count} robot poses, got {len(robot_poses)}.")

        item_count = len(self.item_entries)
        if item_count > 0:
            if len(item_poses) == 1 and item_count > 1:
                item_poses = [list(item_poses[0]) for _ in range(item_count)]
            elif len(item_poses) != item_count:
                raise ValueError(f"Expected {item_count} item poses, got {len(item_poses)}.")

        active_joints = [1] * self.num_joints + [0] * (7 - self.num_joints)
        joint_positions = [0] * 7
        joint_velocities = [0] * 7

        robots = []
        for idx, pose in enumerate(robot_poses):
            robots.append({
                "robot_index": idx,
                "active_joints": list(active_joints),
                "joint_positions": list(joint_positions),
                "joint_velocities": list(joint_velocities),
                "gripper_position": 0.0,
                "robot_pose": list(pose),
            })

        targets = [list(pose) for pose in target_poses]
        items = [list(pose) for pose in item_poses]

        for idx, target_pose in enumerate(targets):
            if idx < len(robot_poses):
                robot_pose = robot_poses[idx]
                targets[idx] = transform_local_pose_to_world(robot_pose, target_pose)

        self.reset_state = {
            "robots": robots,
            "targets": targets,
            "items": items,
        }

    def get_monitor_payload(self) -> MonitorPayload:
        """Build structured monitor telemetry for all robots (manipulator).

        Uses per-robot observation snapshots cached during the last
        ``update()`` call so multi-robot setups report correct per-robot
        data instead of always reflecting the last robot processed.

        Returns:
            MonitorPayload with manipulator-specific telemetry.
        """
        obs_cache = getattr(self, '_per_robot_obs_cache', [])
        rewards_cache = getattr(self, '_per_robot_rewards_cache', [])
        states_cache = getattr(self, '_per_robot_states_cache', [])

        if not obs_cache:
            return MonitorPayload(
                robot_pose=Pose2D(0.0, 0.0, 0.0),
                robot_velocity=(0.0, 0.0),
                target_delta=(0.0, 0.0, 0.0),
                reward=0.0,
                success=False,
                collision=False,
            )

        robots_data: List[Dict[str, Any]] = []
        for ridx in range(self.robot_count):
            obs = obs_cache[ridx] if ridx < len(obs_cache) else None
            if obs is None or len(obs) == 0:
                continue

            ee_x = float(obs[14]) if len(obs) > 14 else 0.0
            ee_y = float(obs[15]) if len(obs) > 15 else 0.0
            ee_z = float(obs[16]) if len(obs) > 16 else 0.0
            t_x = float(obs[20]) if len(obs) > 20 else 0.0
            t_y = float(obs[21]) if len(obs) > 21 else 0.0
            t_z = float(obs[22]) if len(obs) > 22 else 0.0

            joint_angles = [float(obs[j]) for j in range(min(7, len(obs)))]
            joint_velocities = [float(obs[7 + j]) for j in range(min(7, max(0, len(obs) - 7)))]
            reward_i = rewards_cache[ridx] if ridx < len(rewards_cache) else 0.0

            rdict: Dict[str, Any] = {
                'robot_position': [ee_x, ee_y],
                'robot_yaw': 0.0,
                'robot_velocity': [0.0, 0.0],
                'target_position': [t_x, t_y],
                'target_delta': [t_x - ee_x, t_y - ee_y, t_z - ee_z],
                'reward': reward_i,
                'success': bool(self._per_robot_success[ridx]) if ridx < len(self._per_robot_success) else False,
                'collision': bool(self.collided_env),
                'joint_position': [np.rad2deg(a) for a in joint_angles],
                'joint_velocity': [np.rad2deg(v) for v in joint_velocities],
                'end_effector_pose': [float(obs[i]) for i in range(14, min(20, len(obs)))],
                'target_pose': [float(obs[i]) for i in range(20, min(26, len(obs)))],
                'object_pose': [float(obs[i]) for i in range(26, min(32, len(obs)))],
                'gripper_position': float(obs[32]) if len(obs) > 32 else 0.0,
            }

            # Agent action set by simulator_vec_env before update()
            last_action = getattr(self, '_monitor_last_action', None)
            if last_action is not None:
                act_arr = np.asarray(last_action, dtype=np.float32)
                if act_arr.ndim == 2 and ridx < act_arr.shape[0]:
                    rdict['agent_action'] = act_arr[ridx].tolist()
                elif act_arr.ndim == 1:
                    rdict['agent_action'] = act_arr.tolist()

            # Per-robot agent_reward
            rdict['agent_reward'] = np.array([reward_i], dtype=np.float32)

            # Attach per-robot agent_state from cached states
            if ridx < len(states_cache):
                rdict['agent_state'] = states_cache[ridx]

            robots_data.append(rdict)

        r0 = robots_data[0] if robots_data else {}
        r0_pos = r0.get('robot_position', [0.0, 0.0])
        r0_delta = r0.get('target_delta', [0.0, 0.0, 0.0])
        reward_total = float(getattr(self, '_last_reward', sum(rewards_cache)))

        return MonitorPayload(
            robot_pose=Pose2D(float(r0_pos[0]), float(r0_pos[1]), 0.0),
            robot_velocity=(0.0, 0.0),
            target_delta=(float(r0_delta[0]), float(r0_delta[1]), float(r0_delta[2])),
            reward=reward_total,
            success=bool(any(self._per_robot_success)),
            collision=bool(self.collided_env),
            target_pos=(float(r0.get('target_position', [0.0, 0.0])[0]),
                        float(r0.get('target_position', [0.0, 0.0])[1])),
            agent_state=states_cache[0] if states_cache else None,
            agent_reward=np.array([reward_total], dtype=np.float32),
            robots=robots_data,
        )

    def _get_terminal(self):
        """
            a function to define terminal condition, it can be customized to define terminal conditions based on
            episode duration, whether the task was solved, whether collision or some illegal state happened
        """
        if self.ts > self.max_ts:
            return True
        return False

    def _retrieve_image(self, observation):
        """
            if the image observations are enabled at the simulator, it passes image data in the observation in addition
            to the numeric data. This method shows how the image data can be accessed, saved locally, processed, etc.
        """
        # how to access the image data when it is provided by the simulator
        base64_bytes = observation['ImageData'][0].encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)
        # saving the image locally
        # with open('images/received-image-{}.{}'.format(1, 'jpg'), 'wb') as image_file:
        #    image_file.write(image_bytes)
        image = np.frombuffer(image_bytes, np.uint8)
        # image_bw = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        #cv2.imwrite('images/received-image-{}.{}'.format('cv', 'jpg'), image)
        # cv2.imwrite('images/received-image-bw-{}.{}'.format('cv', 'jpg'), image_bw)
        image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
        # cv2.imwrite('images/received-image-{}.{}'.format('blurred', 'jpg'), dst)
        #dim = (self.image_size, self.image_size)
        # resize image
        #image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        #return image
        return image[:, :, newaxis]
        # return np.zeros(shape=(self.image_size, self.image_size, 1),
        #        dtype=np.uint8)
