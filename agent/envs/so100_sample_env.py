"""
A sample Env class inheriting from the DART-Unity Env for the Standard Open SO-100 arm with 5 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment
Unity is used as the main simulator for physics/rendering computations.
The Unity interface receives joint velocities as commands and returns joint positions and velocities

DART is used to calculate inverse kinematics of the SO-100 chain.
DART changes the agent action space from the joint space to the cartesian space
(position-only or pose/SE(3)) of the end-effector.

action_by_pd_control method can be called to implement a Proportional-Derivative control law instead of an RL policy.

Note: Coordinates in the Unity simulator are different from the ones in DART which used here:
      The mapping is [X, Y, Z] of Unity is [-y, z, x] of DART
"""

import numpy as np
import os

from gym import spaces
from typing import Any, Dict, List
from envs_dart.so100_dart_unity import SO100DartUnityEnv
from utils.config_utils import (
    expand_item_instances,
    resolve_item_poses_unity,
    resolve_robot_poses_unity,
    resolve_target_poses_unity,
    transform_local_pose_to_world,
)
from utils.telemetry import MonitorPayload, Pose2D

# Import when images are used as state representation
# import cv2
# import base64

class SO100SampleEnv(SO100DartUnityEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns,
                 state_type, enable_render=False,
                 with_objects=False, target_mode="random", target_path="/misc/generated_random_targets/cart_pose_7dof.csv", goal_type="target",
                 joints_safety_limit=0.0, max_joint_vel=20.0, max_ee_cart_vel=10.0, max_ee_cart_acc =3.0, max_ee_rot_vel=4.0, max_ee_rot_acc=1.2,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, 0, 0],
                 robotic_tool=None, end_effector_model=None, manipulator_config=None, manipulator_gym_config=None, env_id=0):

        # range of vertical, horizontal pixels for the DART viewer
        viewport = (0, 0, 500, 500)

        self.goal_type = goal_type # Target or box 

        ##############################################################################
        # Set Limits -> Important: Must be set before calling the super().__init__() #
        ##############################################################################

        # Variables below exist in the parent class, hence the names should not be changed                            #
        # Min distance to declare that the target is reached by the end-effector, adapt the values based on your task #
        self.MIN_POS_DISTANCE = 0.05  # [m]
        self.MIN_ROT_DISTANCE = 0.1   # [rad]

        self.JOINT_POS_SAFE_LIMIT = np.deg2rad(joints_safety_limit) # Manipulator joints safety limit

        # admissible range for joint positions, velocities, accelerations, # 
        # and torques of the SO-100 kinematic chain                        #
        self.MAX_JOINT_POS = np.deg2rad([ 114.5916, 200.5352,         0,  68.75494,  179.9993]) - self.JOINT_POS_SAFE_LIMIT  # [rad]: based on the specs
        self.MIN_JOINT_POS = np.deg2rad([-114.5916,        0, -179.9993, -143.2395, -179.9993]) + self.JOINT_POS_SAFE_LIMIT

        # Joint space #
        self.MAX_JOINT_VEL = np.deg2rad(np.full(5, max_joint_vel))                                        # [rad/s]: just approximation due to no existing data
        self.MAX_JOINT_ACC = 3.0 * self.MAX_JOINT_VEL                                                     # [rad/s^2]: just approximation due to no existing data
        self.MAX_JOINT_TORQUE = np.full(5, 35.0)                                                          # [Nm]: just approximation due to no existing data

        # admissible range for Cartesian pose translational and rotational velocities, #
        # and accelerations of the end-effector                                        #
        self.MAX_EE_CART_VEL = np.full(3, max_ee_cart_vel)                                                # np.full(3, 10.0) # [m/s] --- not optimized values for sim2real transfer
        self.MAX_EE_CART_ACC = np.full(3, max_ee_cart_acc)                                                # np.full(3, 3.0) # [m/s^2] --- not optimized values
        self.MAX_EE_ROT_VEL = np.full(3, max_ee_rot_vel)                                                  # np.full(3, 4.0) # [rad/s] --- not optimized values
        self.MAX_EE_ROT_ACC = np.full(3, max_ee_rot_acc)                                                  # np.full(3, 1.2) # [rad/s^2] --- not optimized values

        ##################################################################################
        # End set limits -> Important: Must be set before calling the super().__init__() #
        ##################################################################################

        # Backward/forward compatibility: map end_effector_model to legacy robotic_tool if not provided
        if robotic_tool is None:
            robotic_tool = 'default_gripper'

        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns,
                         state_type=state_type, robotic_tool=robotic_tool, enable_render=enable_render,
                         with_objects=with_objects, target_mode=target_mode, target_path=target_path, viewport=viewport,
                         random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions, env_id=env_id)

        # Collision happened for this env when set to 1. In case the manipulator is spanwed in a different position than the #
        # default home, for the remaining of the episode zero velocities are sent to UNITY                                   #
        self.collided_env = 0
        self.reset_flag = False

        ################################
        # Set initial joints positions #
        ################################
        self.random_initial_joint_positions = random_initial_joint_positions                                                 # True, False
        self.initial_positions = np.asarray(initial_positions, dtype=float) if initial_positions is not None else None

        # Initial positions flag for the manipulator after reseting. 1 means different than the default home position #
        # In that case, the environments should terminate at the same time step due to UNITY synchronization          #
        if((self.initial_positions is None or np.count_nonzero(initial_positions) == 0) and self.random_initial_joint_positions == False):
            self.flag_zero_initial_positions = 0
        else:
            self.flag_zero_initial_positions = 1

        # Clip gripper action to this limit #
        self.gripper_clip = 255

        self.transform_ee_initial = None
        self.save_image = False  # change it to True to save images received from Unity into jpg files

        if self.save_image:
            self.save_image_folder = self.get_save_image_folder()

        self.manipulator_config = manipulator_config if isinstance(manipulator_config, dict) else {}
        self.manipulator_gym_config = manipulator_gym_config if isinstance(manipulator_gym_config, dict) else {}
        self._per_robot_obs_cache: List[Dict[str, Any]] = []
        self._per_robot_states_cache: List[np.ndarray] = []
        self._per_robot_rewards_cache: List[float] = []
        self._robot_poses_unity = self._resolve_robot_poses_unity()
        self.robot_count = len(self._robot_poses_unity)
        self._item_instances = expand_item_instances(self.manipulator_config)
        self.item_count = len(self._item_instances)
        self._gripper_targets = np.zeros(self.robot_count, dtype=np.float32)
        self._prev_pos_distance_per_robot = [None] * self.robot_count
        self._prev_rot_distance_per_robot = [None] * self.robot_count
        self._per_robot_success = [False] * self.robot_count

        # Some attributes that are initialized in the parent class:
        # self.reset_counter --> keeps track of the number of resets performed
        # self.reset_state   --> reset vector for initializing the Unity simulator at each episode
        # self.tool_target   --> initial position for the gripper state

        # helper methods that can be called from the parent class:
        # self._dart_pose_to_payload(dart_pose) -- reorders DART pose [rx,ry,rz,x,y,z] to payload [x,y,z,rx,ry,rz]
        # self.get_rot_error_from_quaternions(target_quat, current_quat) -- rotation error in angle-axis from quaternion inputs
        # self.get_rot_ee_quat()  -- DART/ROS coords -- get the orientation of the ee in quaternion
        # self.get_ee_pos()       -- DART/ROS coords -- get the position of the ee
        # self.get_ee_orient()    -- DART/ROS coords -- get the orientation of the ee in angle-axis
        # self.get_ee_pose()      -- DART/ROS coords -- get the pose of the ee
        # NOTE: Coordinate conversion is handled entirely on the Unity side via CoordinateConverter.

        # the lines below should stay as it is
        self.MAX_EE_VEL = self.MAX_EE_CART_VEL
        self.MAX_EE_ACC = self.MAX_EE_CART_ACC
        if orientation_control:
            self.MAX_EE_VEL = np.concatenate((self.MAX_EE_ROT_VEL, self.MAX_EE_VEL))
            self.MAX_EE_ACC = np.concatenate((self.MAX_EE_ROT_ACC, self.MAX_EE_ACC))

        # the lines below wrt action_space_dimension should stay as it is
        self.action_space_dimension = self.n_links  # there would be 5 actions in case of joint-level control
        if use_ik:
            # There are three cartesian coordinates x,y,z for inverse kinematic control
            self.action_space_dimension = 3
            if orientation_control:
                # and the three rotations around each of the axis
                self.action_space_dimension += 3

        if self.with_gripper:
            self.action_space_dimension += 1  # gripper velocity

        # Variables below exist in the parent class, hence the names should not be changed
        tool_length = 0.2  # [m] allows for some tolerances in maximum observation

        # x,y,z of TCP: maximum reach of arm plus tool length in meters
        ee_pos_high = np.array([0.95 + tool_length, 0.95 + tool_length, 1.31 + tool_length])
        ee_pos_low = -np.array([0.95 + tool_length, 0.95 + tool_length, 0.39 + tool_length])

        high = np.empty(0)
        low = np.empty(0)
        self.observation_indices = {'obs_len': 0}

        if orientation_control:
            # rx,ry,rz of TCP: maximum orientation in radians without considering dexterous workspace
            ee_rot_high = np.full(3, np.pi)
            # observation space is distance to target orientation (rx,ry,rz), [rad]
            high = np.append(high, ee_rot_high)
            low = np.append(low, -ee_rot_high)
            self.observation_indices['ee_rot'] = self.observation_indices['obs_len']
            self.observation_indices['obs_len'] += 3

        # and distance to target position (dx,dy,dz), [m]
        high = np.append(high, ee_pos_high - ee_pos_low)
        low = np.append(low, -(ee_pos_high - ee_pos_low))
        self.observation_indices['ee_pos'] = self.observation_indices['obs_len']
        self.observation_indices['obs_len'] += 3

        # and joint positions [rad] and possibly velocities [rad/s]
        if 'a' in self.state_type:
            high = np.append(high, self.MAX_JOINT_POS)
            low = np.append(low, self.MIN_JOINT_POS)
            self.observation_indices['joint_pos'] = self.observation_indices['obs_len']
            self.observation_indices['obs_len'] += self.n_links
        if 'v' in self.state_type:
            high = np.append(high, self.MAX_JOINT_VEL)
            low = np.append(low, -self.MAX_JOINT_VEL)
            self.observation_indices['joint_vel'] = self.observation_indices['obs_len']
            self.observation_indices['obs_len'] += self.n_links

        # the lines below should stay as it is.                                                                         #
        # Important:        Adapt only if you use images as state representation, or your task is more complicated      #
        # Good practice:    If you need to adapt several methods, inherit from SO100SampleEnv and define your own class #
        self.action_space = spaces.Box(
            low=-np.ones((self.robot_count, self.action_space_dimension), dtype=np.float32),
            high=np.ones((self.robot_count, self.action_space_dimension), dtype=np.float32),
            dtype=np.float32,
        )

        low_multi = np.tile(low.astype(np.float32), self.robot_count)
        high_multi = np.tile(high.astype(np.float32), self.robot_count)
        self.observation_space = spaces.Box(low=low_multi, high=high_multi, dtype=np.float32)

        self.reward_range = (-0.05, 0.05)

    def _resolve_robot_poses_unity(self):
        pose_cfg = self.manipulator_gym_config if self.manipulator_gym_config else self.manipulator_config
        return resolve_robot_poses_unity(pose_cfg, default_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def _build_robot_entries(self, base_entry):
        self._robot_poses_unity = self._resolve_robot_poses_unity()
        self.robot_count = len(self._robot_poses_unity)
        self._gripper_targets = np.zeros(self.robot_count, dtype=np.float32)
        self._prev_pos_distance_per_robot = [None] * self.robot_count
        self._prev_rot_distance_per_robot = [None] * self.robot_count
        self._per_robot_success = [False] * self.robot_count
        return [dict(base_entry, robot_pose=list(pose)) for pose in self._robot_poses_unity]

    def _update_env_flags(self):
        ###########################################################################################
        # collision happened or joints limits overpassed                                          #
        # Important: in case the manipulator is spanwed in a different position than the default  #
        #            home, for the remaining of the episode zero velocities are sent to UNITY     #
        #            see _send_actions_and_update() method in simulator_vec_env.py                #
        ###########################################################################################
        if self.unity_observation['collision_flag'] == 1.0 or self.joints_limits_violation():
            self.collided_env = 1
            # Reset when we have a collision only when we spawn the robot to the default #
            # home position, else wait the episode to finish                             #
            # Important: you may want to reset anyway depending on your task - adapt     #
            if self.flag_zero_initial_positions == 0:
                self.reset_flag = True

    def get_save_image_folder(self):
        path = os.path.dirname(os.path.realpath(__file__))
        env_params_list = [self.max_ts, self.dart_sim.orientation_control, self.dart_sim.use_ik, self.dart_sim.ik_by_sns,
                           self.state_type, self.target_mode, self.goal_type, self.random_initial_joint_positions,
                           self.robotic_tool]
        env_params = '_'.join(map(str, env_params_list))
        idx = 0
        while True:
            save_place = path + '/misc/unity_image_logs/' + self.env_key + '_' + env_params + '_%s' % idx
            if not os.path.exists(save_place):
                save_place += '/'
                os.makedirs(save_place)
                break
            idx += 1
        return save_place

    def create_target(self):
        """
            defines the target to reach per episode, this should be adapted by the task

            should always return rx,ry,rz,x,y,z in order, -> dart coordinates system
            i.e., first target orientation rx,ry,rz in radians, and then target position x,y,z in meters
                in case of orientation_control=False --> rx,ry,rz are irrelevant and can be set to zero

            _random_target_gen_joint_level(): generate a random sample target in the reachable workspace of the SO-100

            :return: Cartesian pose of the target to reach in the task space (dart coordinates)
        """

        # Default behaviour # 
        if(self.target_mode == "None"): 
            rx, ry, rz = 0.0, np.pi, 0.0
            target = None
            while True:
                x, y, z = np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0), 0.2

                if 0.4*0.4 < x*x + y*y < 0.8*0.8:
                    target = rx, ry, rz, x, y, z
                    break

        elif self.target_mode == "import":
            target = self._recorded_next_target()

        elif self.target_mode == "random":
            target = self._random_target()

        elif self.target_mode == "random_joint_level":
            target = self._random_target_gen_joint_level() # Sample always a rechable target

        elif self.target_mode == "fixed":
            target = self._fixed_target()

        elif self.target_mode == "fixed_joint_level":
            target = self._fixed_target_gen_joint_level()

        else:
            target = [0, 0, 0, 0, -5, -200] # Dummy will be defined from the user later - advise define it in the reset() function 

        return target
    
    def generate_object(self):
        """
            defines the initial box position per episode
            should always return rx,ry,rz,x,y,z in order,
            i.e., first box orientation rx,ry,rz in radians, and then box position x,y,z in meters

            :return: Cartesian pose of the initial box position in the task space
        """

        # depending on your task, positioning the object might be necessary, start from the following sample code
        # sample code to position the object
        # object_height = 0.1
        # z = object_height / 2.0 + 0.005 # use it for tasks including object such as grasping or pushing
        z = -1.0  # use it for tasks without object such as reaching
        x, y = self.np_random.uniform(-1.0, 1.0), self.np_random.uniform(-1.0, 1.0)
        rx, ry, rz = 0.0, 0.0, 0.0

        return rx, ry, rz, x, y, z

    def get_state(self):
        """
           defines the environment state, this should be adapted by the task

           get_pos_error(): returns Euclidean error from the end-effector to the target position
           get_rot_error(): returns Quaternion error from the end-effector to the target orientation

           :return: state for the policy training
        """
        state = np.empty(0)
        if self.dart_sim.orientation_control:
            state = np.append(state, self.dart_sim.get_rot_error())

        state = np.append(state, self.dart_sim.get_pos_error())

        if 'a' in self.state_type: # Append the joints position of the manipulator
            state = np.append(state, self.dart_sim.chain_get_positions())
        if 'v' in self.state_type:
            state = np.append(state, self.dart_sim.chain_get_velocities())

        # the lines below should stay as it is
        self.observation_state = np.array(state)

        return self.observation_state

    def get_reward(self, action, robot_index=0):
        """
           defines the environment reward, this should be adapted by the task

           :param action: is the current action decided by the RL agent

           :return: reward for the policy training
        """
        # stands for reducing position error
        # reward = -self.dart_sim.get_pos_distance()

        # stands for reducing orientation error
        # if self.dart_sim.orientation_control:
        #     reward -= 0.5 * self.dart_sim.get_rot_distance()

        # stands for avoiding abrupt changes in actions
        # reward -= 0.1 * np.linalg.norm(action - self.prev_action)

        # stands for shaping the reward to increase when target is reached to balance at the target
        # if self.get_terminal_reward():
        #     reward += 1.0 * (np.linalg.norm(np.ones(self.action_space_dimension)) - np.linalg.norm(action))

        # incremental reward implementation -- default
        reward = 0.0

        curr_pos_distance = self.dart_sim.get_pos_distance()
        prev_pos = self._prev_pos_distance_per_robot[robot_index]
        if prev_pos is not None:
            reward += prev_pos - curr_pos_distance
        self._prev_pos_distance_per_robot[robot_index] = curr_pos_distance

        if self.dart_sim.orientation_control:
            curr_rot_distance = self.dart_sim.get_rot_distance()
            prev_rot = self._prev_rot_distance_per_robot[robot_index]
            if prev_rot is not None:
                reward += 0.3 * (prev_rot - curr_rot_distance)
            self._prev_rot_distance_per_robot[robot_index] = curr_rot_distance

        # the lines below should stay as it is
        self.reward_state = reward

        return self.reward_state

    def get_terminal_reward(self):
        """
           checks if the target is reached

           get_pos_distance(): returns norm of the Euclidean error from the end-effector to the target position
           get_rot_distance(): returns norm of the Quaternion error from the end-effector to the target orientation

           Important: by default a 0.0 value of a terminal reward will be given to the agent. To adapt it please refer to the config.py,
                      in the reward_dict. This terminal reward is given to the agent during reset in the step() function in the simulator_vec_env.py

           :return: a boolean value representing if the target is reached within the defined threshold
        """
        target_reached = False
        if self.dart_sim.get_pos_distance() < self.MIN_POS_DISTANCE:
            if not self.dart_sim.orientation_control:
                target_reached = True
            if self.dart_sim.get_rot_distance() < self.MIN_ROT_DISTANCE:
                target_reached = True

        return target_reached

    def get_terminal(self):
        """
           checks the terminal conditions for the episode - for reset

           :return: a boolean value indicating if the episode should be terminated
        """

        if self.time_step > self.max_ts:
            self.reset_flag = True

        return self.reset_flag

    def update_action(self, action):
        """
           converts env action to the required unity action by possibly using inverse kinematics from DART

           self.dart_sim.command_from_action() --> changes the action from velocities in the task space (position-only 'x,y,z' or complete pose 'rx,ry,rz,x,y,z')
                                                   to velocities in the joint space of the kinematic chain (j1,...,j5)

           :param action: The action vector decided by the RL agent, acceptable range: [-1,+1]

                          It should be a numpy array with the following shape: [arm_action] or [arm_action, tool_action]
                          in case 'robotic_tool' is a gripper, tool_action is always a dim-1 scalar value representative of the normalized gripper velocity

                          arm_action has different dim based on each case of control level:
                              in case of use_ik=False    -> is dim-5 representative of normalized joint velocities
                              in case of use_ik=True     -> there would be two cases:
                                  orientation_control=False  -> of dim-3: Normalized EE Cartesian velocity in x,y,z DART coord
                                  orientation_control=True   -> of dim-6: Normalized EE Rotational velocity in x,y,z DART coord followed by Normalized EE Cartesian velocity in x,y,z DART coord

           :return: the command to send to the Unity simulator including joint velocities and possibly the gripper position
        """

        # the lines below should stay as it is
        act = np.asarray(action, dtype=np.float32)
        if act.ndim == 1:
            act = np.reshape(act, (1, -1))
        if act.shape[0] != self.robot_count:
            raise ValueError(f"Expected action rows={self.robot_count}, got {act.shape[0]}.")

        act = np.clip(act, self.action_space.low, self.action_space.high)
        self.action_state = act.copy()

        unity_actions = []
        latest_payload = getattr(self, '_latest_observation_payload', None)
        robots_payload = latest_payload.get('Robots', []) if isinstance(latest_payload, dict) else []
        robots_by_index = {}
        if isinstance(robots_payload, list):
            for list_idx, payload in enumerate(robots_payload):
                if not isinstance(payload, dict):
                    continue
                try:
                    ridx = int(payload.get('RobotIndex', list_idx))
                except Exception:
                    ridx = list_idx
                if ridx not in robots_by_index:
                    robots_by_index[ridx] = payload
        for idx in range(self.robot_count):
            row = np.asarray(act[idx], dtype=np.float32)

            # This updates the gripper target by accumulating the tool velocity from the action vector   #
            # Note: adapt if needed: e.g. accumulating the tool velocity may not work well for your task #
            if self.with_gripper:
                tool_action = float(row[-1])
                self._gripper_targets[idx] = float(np.clip(self._gripper_targets[idx] + tool_action, 0.0, self.gripper_clip))

                # This removes the gripper velocity from the action vector for inverse kinematics calculation
                row = row[:-1]

            if self.dart_sim.use_ik:
                selected_payload = robots_by_index.get(idx)
                if selected_payload is None and isinstance(robots_payload, list) and idx < len(robots_payload):
                    candidate = robots_payload[idx]
                    if isinstance(candidate, dict):
                        selected_payload = candidate
                if selected_payload is not None:
                    self._unity_retrieve_observation_numeric(selected_payload, latest_payload, robot_index=idx)
                    self._update_dart_chain()
                task_vel = self.MAX_EE_VEL * row
                joint_vel = self.dart_sim.command_from_action(task_vel, normalize_action=False)
            else:
                joint_vel = self.MAX_JOINT_VEL * row

            unity_row = np.asarray(joint_vel, dtype=np.float32)
            if self.with_gripper:
                unity_row = np.append(unity_row, [float(self._gripper_targets[idx])])
            unity_actions.append(unity_row.tolist())

        return np.asarray(unity_actions, dtype=np.float32)

    def update(self, observation, time_step_update=True):
        """
            converts the unity observation to the required env state defined in get_state()
            with also computing the reward value from the get_reward(...) and done flag,
            it increments the time_step, and outputs done=True when the environment should be reset

            important: it also updates the dart kinematic chain of the robot using the new unity simulator observation.
                       always call this function, once you have send a new command to unity to synchronize the agent environment

            :param observation: is the observation received from the Unity simulator within its [X,Y,Z] coordinate system
                                'joint_values':       indices [0:5],
                                'joint_velocities':   indices [5:10],
                                'ee_position':        indices [10:13],
                                'ee_orientation':     indices [13:17],
                                'target_position':    indices [17:20],
                                'target_orientation': indices [20:24],
                                'object_position':    indices [24:27],
                                'object_orientation': indices [27:31],
                                'gripper_position':   indices [31:32],
                                'collision_flag':     indices [32:33],

            :param time_step_update: whether to increase the time_step of the agent

            :return: The state, reward, episode termination flag (done), and an info dictionary
        """
        self._latest_observation_payload = observation if isinstance(observation, dict) else None
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

        # ── per-robot image separation ────────────────────────────────
        overhead_raw = observation.get('OverheadImages', None)
        overhead_candidates: list = []
        if overhead_raw is not None:
            for entry in overhead_raw:
                overhead_candidates.append(entry.get('Data') if isinstance(entry, dict) else entry)

        per_robot_image_raw: list = []
        for payload in robots:
            if not isinstance(payload, dict):
                per_robot_image_raw.append(None)
                continue
            robot_image = payload.get('RobotImage', {})
            if isinstance(robot_image, dict) and robot_image.get('Data') is not None:
                per_robot_image_raw.append(robot_image['Data'])
            else:
                per_robot_image_raw.append(None)

        overhead_frames = self._unity_retrieve_observation_images(overhead_candidates) if overhead_candidates else []
        self._overhead_image_frames = list(overhead_frames)

        self._per_robot_image_frames = []
        for raw in per_robot_image_raw:
            if raw is not None:
                frames = self._unity_retrieve_observation_images([raw])
                self._per_robot_image_frames.append(frames[0] if frames else None)
            else:
                self._per_robot_image_frames.append(None)

        self._unity_image_bytes_list = list(overhead_frames)
        for f in self._per_robot_image_frames:
            if f is not None:
                self._unity_image_bytes_list.append(f)

        # the methods below handles synchronizing states of the DART kinematic chain with the observation from Unity
        # hence it should be always called
        per_robot_states = []
        per_robot_rewards = []
        per_robot_success = []
        self._per_robot_obs_cache = []
        self._per_robot_states_cache = []
        self._per_robot_rewards_cache = []
        self._per_robot_torque_cache = []

        for idx in range(self.robot_count):
            payload = robots_by_index.get(idx)
            if payload is None and idx < len(robots):
                candidate = robots[idx]
                if isinstance(candidate, dict):
                    payload = candidate
            if payload is None:
                raise ValueError(f"Missing robot payload for robot index {idx}.")

            self._unity_retrieve_observation_numeric(payload, observation, robot_index=idx)
            self._update_dart_chain()
            self._update_env_flags()

            # Cache per-robot observation snapshot for get_monitor_payload()
            self._per_robot_obs_cache.append(dict(self.unity_observation))

            # Cache per-robot joint torque from DART inverse dynamics
            try:
                chain = self.dart_sim.chain
                M = chain.getMassMatrix()
                jacc = chain.getAccelerations()
                tau = np.matmul(M, jacc) + chain.getCoriolisAndGravityForces()
                self._per_robot_torque_cache.append(tau[:self.n_links].copy())
            except Exception:
                self._per_robot_torque_cache.append(None)

            state_i = self.get_state()
            per_robot_states.append(np.asarray(state_i, dtype=np.float32))
            self._per_robot_states_cache.append(np.asarray(state_i, dtype=np.float32))
            action_i = self.action_state[idx] if isinstance(self.action_state, np.ndarray) and self.action_state.ndim == 2 else self.action_state
            reward_i = float(self.get_reward(action_i, robot_index=idx))
            per_robot_rewards.append(reward_i)
            self._per_robot_rewards_cache.append(reward_i)
            per_robot_success.append(bool(self.get_terminal_reward()))

        # Class attributes below exist in the parent class, hence the names should not be changed
        if(time_step_update == True):
            self.time_step += 1

        self._state = np.concatenate(per_robot_states, axis=0) if per_robot_states else np.zeros(self.observation_space.shape, dtype=np.float32)
        self._reward = float(np.sum(per_robot_rewards))
        self._done = bool(self.get_terminal())
        self._per_robot_success = per_robot_success
        self._info = {
            "success": bool(any(per_robot_success)),
            "robot_count": self.robot_count,
            "per_robot_rewards": per_robot_rewards,
        }

        self.prev_action = np.array(self.action_state, copy=True)

        return self._state, self._reward, self._done, self._info

    def reset(self, temp=False):
        """
            resets the DART-gym environment and creates the reset_state vector to be sent to Unity

            :param temp: not relevant

            :return: The initialized state
        """
        # takes care of resetting the DART chain and should stay as it is
        self._state = super().reset(random_initial_joint_positions=self.random_initial_joint_positions, initial_positions=self.initial_positions)

        self.transform_ee_initial = None
        self.collided_env = 0
        self._prev_pos_distance_per_robot = [None] * self.robot_count
        self._prev_rot_distance_per_robot = [None] * self.robot_count
        self._per_robot_success = [False] * self.robot_count
        self._gripper_targets = np.zeros(self.robot_count, dtype=np.float32)

        # movement control of each joint can be disabled by setting zero for that joint index
        active_joints = [1] * 5

        # the lines below should stay as it is, Unity simulator expects these joint values in radians
        joint_positions = self.dart_sim.chain.getPositions().tolist()
        joint_velocities = self.dart_sim.chain.getVelocities().tolist()

        # Draw one target per robot. Allow config override through manipulator_gym_config target_poses/target_pose.
        target_positions_mapped_default = []
        for _ in range(self.robot_count):
            random_target = self.create_target()
            
            # sets the initial reaching target for the current episode,
            # should be always called in the beginning of each episode,
            # you might need to call it even during the episode run to change the reaching target for the IK-P controller
            self.set_target(random_target)
            target_positions = self.dart_sim.target.getPositions().tolist()
            target_positions_mapped_default.append(self._dart_pose_to_payload(target_positions))

        pose_cfg = self.manipulator_gym_config if self.manipulator_gym_config else self.manipulator_config
        target_positions_mapped = resolve_target_poses_unity(
            pose_cfg,
            robot_count=self.robot_count,
            default_pose=target_positions_mapped_default if target_positions_mapped_default else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )

        # Spawn the object in UNITY: by default a green box is spawned.
        # Pose is reordered from DART to payload format in ROS convention.
        object_positions = self.generate_object()
        object_positions_mapped = self._dart_pose_to_payload(object_positions)
        item_positions_mapped = resolve_item_poses_unity(pose_cfg, default_pose=object_positions_mapped)
        if not item_positions_mapped:
            item_positions_mapped = [list(object_positions_mapped)]
  
        robot_entry = {
            "active_joints": list(active_joints),
            "joint_positions": list(joint_positions),
            "joint_velocities": list(joint_velocities),
        }

        if self.with_gripper:
            # initial position for the gripper state, accumulates the tool_action velocity received in update_action
            self.tool_target = 0.0  # should be in range [0.0,255.0] for default_gripper
            robot_entry["gripper_position"] = float(self._gripper_targets[0])

        robot_entries = self._build_robot_entries(robot_entry)

        target_positions_shifted = []
        for idx, target_pose in enumerate(target_positions_mapped):
            pose = list(target_pose)
            if idx < len(robot_entries):
                robot_pose = robot_entries[idx].get("robot_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                pose = transform_local_pose_to_world(robot_pose, pose)
            target_positions_shifted.append(pose)

        self.reset_state = {
            "robots": robot_entries,
            "targets": target_positions_shifted,
            "items": [list(pose) for pose in item_positions_mapped],
        }

        self.reset_counter += 1
        self.reset_flag = False

        # Clear caches so get_monitor_payload() returns a safe default until update()
        self._per_robot_obs_cache = []
        self._per_robot_states_cache = []
        self._per_robot_rewards_cache = []
        self._per_robot_torque_cache = []
        self._overhead_image_frames = []
        self._per_robot_image_frames = []

        return self._state

    # ── Data-trace / task-monitor telemetry ──────────────────────────

    def get_monitor_payload(self) -> MonitorPayload:
        """Build structured telemetry for all robots in this environment.

        Uses the per-robot observation snapshots cached during the last
        ``update()`` call, so the payload always reflects the most recent
        Unity observation for every robot.

        Returns:
            MonitorPayload consumed by the data-trace recorder and
            task monitor.
        """
        obs_cache = self._per_robot_obs_cache
        rewards_cache = self._per_robot_rewards_cache
        states_cache = self._per_robot_states_cache

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
            obs = obs_cache[ridx] if ridx < len(obs_cache) else {}

            ee_pos = np.asarray(obs.get('ee_position', [0.0, 0.0, 0.0]), dtype=np.float32)
            ee_rot = np.asarray(obs.get('ee_orientation', [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            tgt_pos = np.asarray(obs.get('target_position', [0.0, 0.0, 0.0]), dtype=np.float32)
            tgt_rot = np.asarray(obs.get('target_orientation', [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            obj_pos = np.asarray(obs.get('object_position', [0.0, 0.0, 0.0]), dtype=np.float32)
            obj_rot = np.asarray(obs.get('object_orientation', [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)
            jvals = np.asarray(obs.get('joint_values', np.zeros(self.n_links)), dtype=np.float32)
            jvels = np.asarray(obs.get('joint_velocities', np.zeros(self.n_links)), dtype=np.float32)
            grip = float(obs.get('gripper_position', 0.0))
            reward_i = rewards_cache[ridx] if ridx < len(rewards_cache) else 0.0

            rdict: Dict[str, Any] = {
                'robot_position': [float(ee_pos[0]), float(ee_pos[1])],
                'robot_yaw': 0.0,
                'robot_velocity': [0.0, 0.0],
                'target_position': [float(tgt_pos[0]), float(tgt_pos[1])],
                'target_delta': [float(tgt_pos[i] - ee_pos[i]) for i in range(3)],
                'reward': reward_i,
                'success': bool(self._per_robot_success[ridx]) if ridx < len(self._per_robot_success) else False,
                'collision': bool(self.collided_env),
                'joint_position': np.rad2deg(jvals).tolist(),
                'joint_velocity': np.rad2deg(jvels).tolist(),
                'end_effector_pose': np.concatenate([ee_pos, ee_rot[:3]]).tolist(),
                'target_pose': np.concatenate([tgt_pos, tgt_rot[:3]]).tolist(),
                'object_pose': np.concatenate([obj_pos, obj_rot[:3]]).tolist(),
                'gripper_position': grip,
            }

            # Per-robot joint torque cached during update()
            torque_cache = getattr(self, '_per_robot_torque_cache', [])
            if ridx < len(torque_cache) and torque_cache[ridx] is not None:
                rdict['joint_torque'] = torque_cache[ridx].tolist()

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

            # Per-robot camera image: overhead frames + this robot's camera
            overhead = getattr(self, '_overhead_image_frames', [])
            per_robot_imgs = getattr(self, '_per_robot_image_frames', [])
            frames: list = list(overhead)
            labels: list = [f'Overhead_{i}' for i in range(len(overhead))]
            if ridx < len(per_robot_imgs) and per_robot_imgs[ridx] is not None:
                frames.append(per_robot_imgs[ridx])
                labels.append(f'Robot_{ridx + 1}')
            if frames:
                rdict['agent_image'] = {'frames': frames, 'labels': labels}

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

    ####################
    # Commands related #
    ####################
    def action_by_p_control(self, coeff_kp_lin, coeff_kp_rot):
        """
            computes the task-space velocity commands proportional to the reaching target error

            :param coeff_kp_lin: proportional coefficient for the translational error
            :param coeff_kp_rot: proportional coefficient for the rotational error

            :return: The action in task space
        """

        robots_payload = []
        latest_payload = getattr(self, '_latest_observation_payload', None)
        if isinstance(latest_payload, dict):
            maybe_robots = latest_payload.get('Robots', [])
            if isinstance(maybe_robots, list):
                robots_payload = maybe_robots

        robots_by_index = {}
        if isinstance(robots_payload, list):
            for list_idx, payload in enumerate(robots_payload):
                if not isinstance(payload, dict):
                    continue
                try:
                    ridx = int(payload.get('RobotIndex', list_idx))
                except Exception:
                    ridx = list_idx
                if ridx not in robots_by_index:
                    robots_by_index[ridx] = payload

        if len(robots_payload) >= self.robot_count:
            per_robot_actions = []
            for idx in range(self.robot_count):
                payload = robots_by_index.get(idx)
                if payload is None and idx < len(robots_payload):
                    candidate = robots_payload[idx]
                    if isinstance(candidate, dict):
                        payload = candidate
                if payload is None:
                    continue
                self._unity_retrieve_observation_numeric(payload, latest_payload, robot_index=idx)
                self._update_dart_chain()

                action_lin = coeff_kp_lin * self.dart_sim.get_pos_error()
                action_i = action_lin

                if self.dart_sim.orientation_control:
                    action_rot = coeff_kp_rot * self.dart_sim.get_rot_error()
                    action_i = np.concatenate(([action_rot, action_i]))

                if self.with_gripper:
                    tool_vel = 0.0
                    action_i = np.append(action_i, [tool_vel])

                per_robot_actions.append(np.asarray(action_i, dtype=np.float32))

            return np.asarray(per_robot_actions, dtype=np.float32)

        action_lin = coeff_kp_lin * self.dart_sim.get_pos_error()
        action = action_lin

        if self.dart_sim.orientation_control:
            action_rot = coeff_kp_rot * self.dart_sim.get_rot_error()
            action = np.concatenate(([action_rot, action]))

        if self.with_gripper:
            tool_vel = 0.0                         # zero velocity means no gripper movement - should be adapted for the task
            action = np.append(action, [tool_vel])

        return action
    
    def get_wrapper_attr(self, name):
        """Mimics Stable-Baselines3 VecEnvWrapper method."""
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

