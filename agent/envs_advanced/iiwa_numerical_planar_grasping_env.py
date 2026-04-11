"""
A numerical planar grasping Env class inheriting from the IiwaSampleEnv for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment

Unity is used as the main simulator for physics/rendering computations.
The Unity interface receives joint velocities as commands and returns joint positions and velocities

DART is used to calculate inverse kinematics of the iiwa chain.
DART changes the agent action space from the joint space to the cartesian space (position-only or pose/SE(3)) of the end-effector.

Notes: - All data exchanged with Unity is now in DART/ROS convention (x=forward, y=left, z=up).
         Coordinate conversions are handled entirely on the Unity (C#) side.
       - In numerical planar grasping envs, a P-controller (agent controller) keeps fixed the uncontrolled DoF during the episode. E.g. height of the ee
       - At the end of the episode, manual actions help the agent to grasp the box
         In that case, the P-controller (manual_actions controller) does not correct the controlled DoF by the RL agent. It keeps them fixed. E.g. rotation of the ee
         see simulator_vec_env.py for more details
       - gym functions use DART/ROS coordinates - e.g. get_state(), get_reward()
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import math
import dartpy as dart
from gym import spaces

from envs.iiwa_sample_env import IiwaSampleEnv
from utils.config_utils import transform_local_pose_to_world

class IiwaNumericalPlanarGraspingEnv(IiwaSampleEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns, state_type, enable_render=False,
                 with_objects=False, target_mode="None", goal_type="box", randomBoxesGenerator=None,
                 joints_safety_limit=10, max_joint_vel=20, max_ee_cart_vel=0.035, max_ee_cart_acc =10, max_ee_rot_vel=0.15, max_ee_rot_acc=10,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2], noise_enable_rl_obs=False, noise_rl_obs_ratio=0.05,
                 reward_dict=None, agent_kp=0.5, agent_kpr=1.5,
                 robotic_tool=None, end_effector_model=None, manipulator_config=None, manipulator_gym_config=None, env_id=0):

        # Backward/forward compatibility: map end_effector_model to legacy robotic_tool if not provided
        if robotic_tool is None:
            ee_map = {
                None: 'None',
                'None': 'None',
                'ROBOTIQ_2F85': '2_gripper',
                'ROBOTIQ_3F': '3_gripper',
                'CALIBRATION_PIN': 'calibration_pin',
                'DEFAULT_GRIPPER': 'default_gripper'
            }
            robotic_tool = ee_map.get(end_effector_model, 'default_gripper')

        # Some checks #
        if(use_ik == False or orientation_control == False or 'v' in state_type):
            raise Exception("Enable orientation control and inverse kinematics for planar grasping. Use also only joints for the state - abort")

        if(ik_by_sns == True):
            raise Exception("RL agent with sns is not advised - abort")

        if(goal_type != "box"):
            raise Exception("Grasping accepts a box target for now - abort")

        if(robotic_tool.find("gripper") == -1):
            raise Exception("Enable gripper for grasping - abort")

        if(not np.isclose(initial_positions[3], -np.pi/2) or not np.isclose(initial_positions[5], np.pi/2) or (not np.isclose(initial_positions[6], np.pi/2) and not np.isclose(initial_positions[6], 0))):
            print("Warning: initial_positions are different - make sure you have adapted the planar envs and manual actions correctly")

        # the init of the parent class should be always called, this will in the end call reset() once
        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns, state_type=state_type, enable_render=enable_render,
                        with_objects=with_objects, target_mode=target_mode, goal_type=goal_type, joints_safety_limit=joints_safety_limit, max_joint_vel=max_joint_vel, max_ee_cart_vel=max_ee_cart_vel,
                        max_ee_cart_acc=max_ee_cart_acc, max_ee_rot_vel=max_ee_rot_vel, max_ee_rot_acc=max_ee_rot_acc, random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions,
                        robotic_tool=robotic_tool, manipulator_config=manipulator_config, manipulator_gym_config=manipulator_gym_config, env_id=env_id)

        # Box dataset to spawn boxes #
        self.randomBoxesGenerator = randomBoxesGenerator

        # Keep object initial pose. Easier calculations for rotations - box does not move during the planar episode #
        self.init_object_pose = None
        self.init_object_pose_per_robot = [None] * self.robot_count

        ###################################################################
        # Reward related                                                  #
        ###################################################################
        self.reward_collision = reward_dict["reward_collision"]              # Value
        self.reward_height_goal = reward_dict["reward_height_goal"]          # Height in meters

        self.reward_pose_weight = reward_dict["reward_pose_weight"]          # Weight for the pose error displacement
        self.reward_pose_norm_const = reward_dict["reward_pose_norm_const"]  # Normalization constant for the pose error displacement

        self.reward_x_weight = reward_dict["reward_x_weight"]
        self.reward_x_norm_const = reward_dict["reward_x_norm_const"]

        self.reward_y_weight = reward_dict["reward_y_weight"]
        self.reward_y_norm_const = reward_dict["reward_y_norm_const"]

        self.collision_flag = False                                          # Give collision reward one once per episode

        # Save previous distance ee/box for calculting the displacements - see reward definition #
        self.prev_dist_ee_box_x = np.inf
        self.prev_dist_ee_box_y = np.inf
        self.prev_dist_ee_box_rz = np.inf
        self.prev_dist_ee_box_x_per_robot = [np.inf] * self.robot_count
        self.prev_dist_ee_box_y_per_robot = [np.inf] * self.robot_count
        self.prev_dist_ee_box_rz_per_robot = [np.inf] * self.robot_count
        self._per_robot_success = [False] * self.robot_count
        ###################################################################

        ###################################################################
        # P-controller related                                            #
        ###################################################################
        self.agent_kp = agent_kp                                          # P-controller gain for positional errors during the episode
        self.agent_kpr = agent_kpr
        self.target_rot_quat_dart = None                                  # Save the target orientation in quaternions - keep moving planar during the episode
        self.target_z_dart = None                                         # Target height - keep moving planar during the episode
        ###################################################################

        ###################################################################
        # Noise related                                                   #
        ###################################################################
        self.noise_enable_rl_obs = noise_enable_rl_obs                    # True/False
        self.noise_rl_obs_ratio = noise_rl_obs_ratio                      # e.g. 0.05
        ###################################################################

        #############################################################################################
        # Joints are normalized in [-1, 1] at the get_state() function - for planar envs            #
        # The below settings are necessary to be set so that in iiwa_dart.py when                   #
        # calling the self.observation_space.sample() function, we denormalize the joints positions #
        #############################################################################################
        self.normalized_rl_obs = True
        self.observation_indices = {'obs_len': 0}

        #########################################
        # Action and observation spaces related #
        #########################################
        if np.isclose(np.asarray([self.reward_pose_weight]), np.asarray([0])): # No rotation control - 2DoF
            ##################
            # Viewer-related #
            ##################

            # Exist in base-class - overide #
            self.observation_indices['joint_pos'] = 2 # In which position of the agent state the joint positions start
            self.observation_indices['obs_len'] = 9

            ###############
            # Gym-related #
            ###############

            # x (forward), y (lateral) in ROS - no rotation #
            self.action_space_dimension = 2
            self.observation_space_dimension = 9

            # 4 DoF that are being controlled by the P-controller - keep moving planar during the episode #
            self.config_p_controller = {"kpr_x": 1,
                                        "kpr_y": 1,
                                        "kpr_z": 1,
                                        "kp_x": 0,  # Controlled by the RL-agent 
                                        "kp_y": 0,
                                        "kp_z": 1
                                       }
        else: # 3DoF control, rotation is active
            ##################
            # Viewer-related #
            ##################
            self.observation_indices['joint_pos'] = 3
            self.observation_indices['obs_len'] = 10

            ###############
            # Gym-related #
            ###############

            # x (forward), y (lateral), rz (yaw) in ROS #
            self.action_space_dimension = 3
            self.observation_space_dimension = 10

            # 3 DoF that are being controlled by the P-controller - keep moving planar during the episode #
            self.config_p_controller = {"kpr_x": 1,
                                        "kpr_y": 1,
                                        "kpr_z": 0,
                                        "kp_x": 0,
                                        "kp_y": 0,
                                        "kp_z": 1
                                       }

        #####################
        # Define gym spaces #
        #####################
        self.action_space = spaces.Box(
            low=-np.ones((self.robot_count, self.action_space_dimension), dtype=np.float32),
            high=np.ones((self.robot_count, self.action_space_dimension), dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-np.ones(self.observation_space_dimension * self.robot_count, dtype=np.float32),
            high=np.ones(self.observation_space_dimension * self.robot_count, dtype=np.float32),
            dtype=np.float32,
        )

    def _update_env_flags(self):
        # Collision or joints limits overpassed                                       #
        # env will not be reseted - wait until the end of the episode for planar envs #
        if self.unity_observation['collision_flag'] == 1.0 or self.joints_limits_violation():
            self.collided_env = 1

    def get_state(self):
        """
            defines the environment state:
                Format: [x_error, y_error, rz_error, j1, .., j7] normalized in [-1, 1]
                        - error from the target pose: ee to the target box (x, y, rz axis)
                        - joints positions
                        - ROS/DART coords system

                Note: if rz rotation is not controlled by the RL agent, then the state will not include the rz_error part

           :return: observation state for the policy training.
        """
        state = np.empty(0)
        if(self.init_object_pose == None):
            return state

        # Relative position normalized errors of ee to the box (ROS: x=forward, y=lateral) #
        dx_ee_b, dy_ee_b = self.get_error_ee_box_x_y_normalized()

        # Add to state #
        state = np.append(state, np.array([dx_ee_b, dy_ee_b]))

        # Rotation control is active #
        if(self.action_space_dimension == 3):
            # Normalized yaw (rz) rotation error #
            drz_ee_b = self.get_error_ee_box_rz_normalized()

            # Add to state #
            state = np.append(state, np.array([drz_ee_b]))

        # Get joints positions #
        joint_positions = self.dart_sim.chain_get_positions()
        joint_positions = self.normalize_joints(joint_positions)

        # Add to state #
        state = np.append(state, joint_positions)

        #####################################################
        # Apply random noise to the RL-observation          #
        # Needed for deploying the model to the real system #
        #####################################################
        if (self.noise_enable_rl_obs == True):
            state = (state + self.np_random.uniform(-self.noise_rl_obs_ratio, self.noise_rl_obs_ratio,
                                                     (1, len(state)))).clip(-1, 1).tolist()[0]

        # the lines below should stay as it is - exist in parent class
        self.observation_state = np.array(state)

        return self.observation_state

    def get_reward(self, action):
        """
           defines the environment reward
           novel reward that uses displacements -- see self.get_reward_term_displacement()

           :param action: is the current action decided by the RL agent - not used

           :return: reward for the policy training
        """

        reward = 0.0
        if(self.collided_env != 1): # Valid env
            reward += self.get_reward_term_displacement()

        # Collision penalty is given only once #
        elif(self.collided_env == 1 and self.collision_flag == False):
            reward += self.reward_collision
            self.collision_flag = True

        # the lines below should stay as it is
        self.reward_state = reward

        return self.reward_state

    def get_terminal_reward(self):
        """
           checks if the box is in the air after the end of the episode and manual actions (down/close/up) - see also simulator_vec_env.py

           :return: a boolean value representing if the box is in the air (True means success)
        """
        return bool(any(self._per_robot_success))

    def get_terminal(self):
        """
           checks the terminal conditions for the episode
                Important: planar envs should terminate at the same time step - when 'num_envs' != 1

           :return: a boolean value indicating if the episode should be terminated - maximum timesteps reached
        """
        self.reset_flag = False
        if self.time_step > self.max_ts:
            self.reset_flag = True

        return self.reset_flag

    def update_action(self, action):
        """
           converts env action to the required unity action by using inverse kinematics from DART
               - action dim: 3, or 2 (in case of no rotation control)
               - the remaining DoF are controlled by the P-controller.
               - the Gripper is not controlled by the RL agent only during the manual and at the end of the episode - see simulator_vec_env.py

           :param action: The action vector decided by the RL agent. Acceptable range: [-1, +1]

           :return: the command to send to the Unity simulator including joint velocities and gripper position
        """

        act = np.asarray(action, dtype=np.float32)
        if act.ndim == 1:
            if act.size != self.action_space_dimension:
                raise ValueError(f"Expected action dim {self.action_space_dimension}, got {act.size}.")
            if self.robot_count > 1:
                act = np.tile(act.reshape(1, -1), (self.robot_count, 1))
            else:
                act = act.reshape(1, -1)
        elif act.ndim == 2:
            if act.shape[1] != self.action_space_dimension:
                raise ValueError(f"Expected per-robot action dim {self.action_space_dimension}, got {act.shape[1]}.")
            if act.shape[0] != self.robot_count:
                raise ValueError(f"Expected {self.robot_count} robot action rows, got {act.shape[0]}.")
        else:
            raise ValueError(f"Unsupported action shape {act.shape}.")

        self.action_state = act
        env_actions = np.clip(act, self.action_space.low, self.action_space.high)

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

        unity_actions = []
        for ridx in range(self.robot_count):
            payload = robots_by_index.get(ridx)
            if payload is None and ridx < len(robots_payload):
                candidate = robots_payload[ridx]
                if isinstance(candidate, dict):
                    payload = candidate
            if payload is not None and isinstance(latest_payload, dict):
                self._unity_retrieve_observation_numeric(payload, latest_payload, robot_index=ridx)
                self._update_dart_chain()

            if isinstance(self.init_object_pose_per_robot, list) and ridx < len(self.init_object_pose_per_robot):
                self.init_object_pose = self.init_object_pose_per_robot[ridx]

            ##############################################################################
            # Reset the P-controller - save the initial pose of the manipulator          #
            # for moving in a planar manner during the episode e.g. keep the same height #
            ##############################################################################
            if self.time_step == 1:
                self.reset_agent_p_controller()

            env_action = env_actions[ridx]

            # Rotation is controlled by the RL-agent
            task_vel = np.zeros(3 if self.action_space_dimension == 3 else 2)

            ###################################################################
            # The RL agent controls the x, y and rotational z-axis (optional) #
            ###################################################################

            ##################################################################################
            # Calculate the errors for the P-controller. Axis not controlled by the RL-agent #
            # Important: P-controller expects dart coordinates                               #
            ##################################################################################
            ee_height = self.get_ee_pos()[2]  # ROS z = height
            z_diff = self.target_z_dart - ee_height                                                               # Height 
            curr_quat = self.get_rot_ee_quat()                                                                    # Current orientation of the ee in quaternions
            rx_diff, ry_diff, rz_diff = self.get_rot_error_from_quaternions(self.target_rot_quat_dart, curr_quat) # Orientation error from the target pose
            ################################################################################################

            if self.action_space_dimension == 3: # Rotation is active - 3DoF control by the RL agent
                task_vel[0] = self.MAX_EE_VEL[2] * env_action[0]
                task_vel[1] = self.MAX_EE_VEL[3] * env_action[1]
                task_vel[2] = self.MAX_EE_VEL[4] * env_action[2]

                ######################################################################################
                # P-controller + inverse kinematics                                                  #
                #   - The DoF that are controlled by the RL-agent are unaffected by the P-controller #
                #   - see config_p_controller dictionary                                             #                        
                ######################################################################################
                joint_vel = self.action_by_p_controller_custom(
                    rx_diff, ry_diff, task_vel[0], task_vel[1], task_vel[2], z_diff,
                    self.agent_kpr, self.agent_kp, self.config_p_controller
                )
            else: # Rotation is not active - 2DoF control by the RL agent
                task_vel[0] = self.MAX_EE_VEL[3] * env_action[0]
                task_vel[1] = self.MAX_EE_VEL[4] * env_action[1]
                joint_vel = self.action_by_p_controller_custom(
                    rx_diff, ry_diff, rz_diff, task_vel[0], task_vel[1], z_diff,
                    self.agent_kpr, self.agent_kp, self.config_p_controller
                )

            ##########################################################################################
            # Gripper is not controlled via the RL-agent - manual actions - see simulator_vec_env.py # 
            ##########################################################################################
            unity_actions.append(np.append(joint_vel, [float(0.0)]))

        return np.asarray(unity_actions, dtype=np.float32)

    def update(self, observation, time_step_update=True):
        """
            converts the unity observation to the required env state defined in get_state()
            with also computing the reward value from the get_reward(...) and done flag,
            it increments the time_step, and outputs done=True when the environment should be reset

            important: it also updates the dart kinematic chain of the robot using the new unity simulator observation.
                       always call this function, once you have send a new command to unity to synchronize the agent environment

            :param observation: is the observation received from the Unity simulator within its [X,Y,Z] coordinate system
                                'joint_values':       indices [0:7],
                                'joint_velocities':   indices [7:14],
                                'ee_position':        indices [14:17],
                                'ee_orientation':     indices [17:21],
                                'target_position':    indices [21:24],
                                'target_orientation': indices [24:28],
                                'object_position':    indices [28:31],
                                'object_orientation': indices [31:35],
                                'gripper_position':   indices [35:36], ---(it is optional, in case a gripper is enabled)
                                'collision_flag':     indices [36:37], ---([35:36] in case of without gripper)
            :param time_step_update: whether to increase the time_step of the agent - during manual actions call with False

            :return: The state, reward, episode termination flag (done), and an empty info dictionary
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

        per_robot_states = []
        per_robot_rewards = []
        per_robot_success = []

        for ridx in range(self.robot_count):
            payload = robots_by_index.get(ridx)
            if payload is None and ridx < len(robots):
                candidate = robots[ridx]
                if isinstance(candidate, dict):
                    payload = candidate
            if payload is None:
                raise ValueError(f"Missing robot payload for robot index {ridx}.")

            self.current_obs = payload.get('Numeric', {}) if isinstance(payload, dict) else {}

            # the methods below handles synchronizing states of the DART kinematic chain with the observation from Unity
            # hence it should be always called
            self._unity_retrieve_observation_numeric(payload, observation, robot_index=ridx)
            self._update_dart_chain()
            self._update_env_flags()

            if isinstance(self.init_object_pose_per_robot, list) and ridx < len(self.init_object_pose_per_robot):
                self.init_object_pose = self.init_object_pose_per_robot[ridx]

            self.prev_dist_ee_box_x = self.prev_dist_ee_box_x_per_robot[ridx]
            self.prev_dist_ee_box_y = self.prev_dist_ee_box_y_per_robot[ridx]
            self.prev_dist_ee_box_rz = self.prev_dist_ee_box_rz_per_robot[ridx]

            state_i = self.get_state()
            action_i = self.action_state[ridx] if isinstance(self.action_state, np.ndarray) and self.action_state.ndim == 2 else self.action_state
            reward_i = self.get_reward(action_i)

            self.prev_dist_ee_box_x = self.get_relative_distance_ee_box_x()   # forward distance (ROS x)
            self.prev_dist_ee_box_y = self.get_relative_distance_ee_box_y()   # lateral distance (ROS y)
            self.prev_dist_ee_box_rz = self.get_relative_distance_ee_box_rz()  # yaw distance (ROS rz)
            self.prev_dist_ee_box_x_per_robot[ridx] = self.prev_dist_ee_box_x
            self.prev_dist_ee_box_y_per_robot[ridx] = self.prev_dist_ee_box_y
            self.prev_dist_ee_box_rz_per_robot[ridx] = self.prev_dist_ee_box_rz

            success_i = bool(self.get_object_height() >= self.reward_height_goal and self.collided_env != 1)
            per_robot_success.append(success_i)
            per_robot_states.append(np.asarray(state_i, dtype=np.float32))
            per_robot_rewards.append(float(reward_i))

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
        state = super().reset()

        ############################################################
        # Spawn the next box and fix the target (for task monitor) #
        ############################################################
        self.init_object_pose_per_robot = []
        for _ in range(self.robot_count):
            object_X, object_Y, object_Z, object_RX, object_RY, object_RZ = self.randomBoxesGenerator()
            self.init_object_pose_per_robot.append([object_X, object_Y, object_Z, object_RX, object_RY, object_RZ])
        self.init_object_pose = self.init_object_pose_per_robot[0] if self.init_object_pose_per_robot else None

        ################################################################################################
        # Align the target with the box. This is done for visualization purposes for the dart viewer   #
        # In the Unity simulator, however, we spawn the target far away                                #
        # Vision-based models might get confused if there is in the image a red target - if the target #
        # is used in vision-based envs -> adapt                                                        #
        ################################################################################################

        target_positions_mapped = []
        object_positions_mapped = []
        for ridx in range(self.robot_count):
            self.init_object_pose = self.init_object_pose_per_robot[ridx]

            # Object position in ROS convention (x=forward, y=left, z=height)
            target_object_X, target_object_Y, target_object_Z = self.init_object_pose[0], self.init_object_pose[1], self.init_object_pose[2]

            # Note in the dart viewer the ee at the goal is more up as we assume that we have a gripper (urdf) but the gripper is not #
            # yet visualized in the viewer. The self.dart_sim.get_pos_distance() returns 0 correctly at the goal                      #
            tool_length = 0.0
            target_object_RX, target_object_RY, target_object_RZ = self.get_box_rotation_in_target_dart_coords_angle_axis()
            # DART target: [rx, ry, rz, x, y, z] — same axis convention as ROS
            target = [target_object_RX, target_object_RY, target_object_RZ, target_object_X, target_object_Y, target_object_Z + tool_length]

            # sets the initial reaching target for the current episode,
            # should be always called in the beginning of each episode,
            # you might need to call it even during the episode run to change the reaching target for the IK-P controller
            self.set_target(target)

            #######################################################################################
            # Spawn the target rectangle outside of the Unity simulator -> since it is not used   #
            # or can harm the vision-based methods                                                #
            # in dart we map the target to be in the same pose as the pose of the box (see above) #
            #######################################################################################
            # Spawn the target far away from the workspace (ROS convention payload)
            # Unity side will convert to its coordinate system
            target_positions_mapped.append([0, -5, -200, 0, 0, 0])

            object_X, object_Y, object_Z, object_RX, object_RY, object_RZ = self.init_object_pose
            object_positions_mapped.append([object_X, object_Y, object_Z, object_RX, object_RY, object_RZ])

        if self.init_object_pose_per_robot:
            self.init_object_pose = self.init_object_pose_per_robot[0]

        # initial position for the gripper state, accumulates the tool_action velocity received in update_action
        self.tool_target = 0.0  # should be in range [0.0,90.0]

        # movement control of each joint can be disabled by setting zero for that joint index
        active_joints = [1] * 7

        # the lines below should stay as it is, Unity simulator expects these joint values in radians
        joint_positions = self.dart_sim.chain.getPositions().tolist()
        joint_velocities = self.dart_sim.chain.getVelocities().tolist()

        robot_entry = {
            "active_joints": list(active_joints),
            "joint_positions": list(joint_positions),
            "joint_velocities": list(joint_velocities),
            "gripper_position": float(self.tool_target),
        }

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
            "items": [list(pose) for pose in object_positions_mapped],
        }

        self.collision_flag = False

        #######################################################################################
        # Keep track the previous distance of the ee to the box - used in the reward function #
        # Initialize -during the __init__() they are set to np.inf                            #
        #######################################################################################
        if('object_position' in self.unity_observation):
            for ridx in range(self.robot_count):
                if isinstance(self.init_object_pose_per_robot, list) and ridx < len(self.init_object_pose_per_robot):
                    self.init_object_pose = self.init_object_pose_per_robot[ridx]
                self.prev_dist_ee_box_x_per_robot[ridx] = self.get_relative_distance_ee_box_x()   # forward (ROS x)
                self.prev_dist_ee_box_y_per_robot[ridx] = self.get_relative_distance_ee_box_y()   # lateral (ROS y)
                self.prev_dist_ee_box_rz_per_robot[ridx] = self.get_relative_distance_ee_box_rz()  # yaw (ROS rz)
            self.prev_dist_ee_box_x = self.prev_dist_ee_box_x_per_robot[0]
            self.prev_dist_ee_box_y = self.prev_dist_ee_box_y_per_robot[0]
            self.prev_dist_ee_box_rz = self.prev_dist_ee_box_rz_per_robot[0]
        else:
            self.prev_dist_ee_box_x_per_robot = [0.0] * self.robot_count
            self.prev_dist_ee_box_y_per_robot = [0.0] * self.robot_count
            self.prev_dist_ee_box_rz_per_robot = [0.0] * self.robot_count
            self.prev_dist_ee_box_x = self.prev_dist_ee_box_y = self.prev_dist_ee_box_rz = 0.0

        self._per_robot_success = [False] * self.robot_count

        return state

    ################
    # P-controller #
    ################
    def action_by_p_controller_custom(self, rx_diff, ry_diff, rz_diff, x_diff, y_diff, z_diff, kpr, kp, config_p_controller, normalize_action=False):
        """
            computes the joint-space velocity command given position and orientation errors and possibly task-space velocities generated from an RL-agent
            it uses a P-controller for generating task-space velocities for the input errors and IK from dart for the joint-space transformation

            Important: input params should be in DARTS cords

            Note: some axis can be controller by an RL-agent while others by the p-controller
                  hence, inputs should be errors or task-space velocities


            :param rx_diff: rx axis error or task-space vel
            :param ry_diff: ry axis error or task-space vel
            :param rz_diff: rz axis error or task-space vel

            :param x_diff: x axis error or task-space vel
            :param y_diff: y axis error or task-space vel
            :param z_diff: z axis error or task-space vel


            :param kpr:    P-controller gain for orientation errors
            :param kp:     P-controller gain for position errors

            :param config_p_controller: dictionary to define which axis are controlled by the P-controller. It should be 6-dim
                                        - 1 means apply the corresponding gain. Else, input is a task-space velocity. Do nothing.
                                        - e.g. {"kpr_x": 1, "kpr_y": 1,
                                                "kpr_z": 1, "kp_x": 0,
                                                "kp_y": 1,   "kp_z": 0
                                               }

             :param normalize_action: whether IK should denormalize the input velocities

            :return: Action in joint-space
        """

        # Wheter to multiply with the P-controller gains #
        kpr_x = kpr if config_p_controller["kpr_x"] == 1 else 1
        kpr_y = kpr if config_p_controller["kpr_y"] == 1 else 1
        kpr_z = kpr if config_p_controller["kpr_z"] == 1 else 1
        kp_x = kp if config_p_controller["kp_x"] == 1 else 1
        kp_y = kp if config_p_controller["kp_y"] == 1 else 1
        kp_z = kp if config_p_controller["kp_z"] == 1 else 1

        task_space_vel = np.array([kpr_x * rx_diff, kpr_y * ry_diff, kpr_z * rz_diff, kp_x * x_diff, kp_y * y_diff, kp_z * z_diff])

        # IK #
        joint_space_vel = self.dart_sim.command_from_action(task_space_vel, normalize_action=normalize_action)

        return joint_space_vel

    def reset_agent_p_controller(self):
        """
            reset the P-controller. Save the initial pose of ee. e.g. for keeping the same height during the RL episode
                Note: dart coords

            affects: self.target_z_dart
            affects: self.target_rot_quat_dart
        """

        ee_height = self.get_ee_pos()[2]  # ROS z = height

        self.target_z_dart = ee_height
        self.target_rot_quat_dart = self.get_rot_ee_quat()

    ################
    # Reward terms #
    ################
    def get_reward_term_displacement(self):
        """
            returns the reward value for the current observation

            uses a displacements logic:
                - the current ee distance to the box in x, y, and rz axis minus the previous ee distance to box for the same axis (see implementation for more)

            :return: reward displacement term (float)
        """
        reward = 0

        # Forward axis (ROS x) #
        curr_dist_ee_box_x = self.get_relative_distance_ee_box_x()
        dx = (self.prev_dist_ee_box_x - curr_dist_ee_box_x)              # Displacement

        dx /= self.reward_x_norm_const # Normalize
        dx = np.clip(dx, -1, 1)        # Clip for safety
        dx *= self.reward_x_weight     # Weight this term

        # Lateral axis (ROS y) #
        curr_dist_ee_box_y = self.get_relative_distance_ee_box_y()
        dy = (self.prev_dist_ee_box_y - curr_dist_ee_box_y)

        dy /= self.reward_y_norm_const
        dy = np.clip(dy, -1, 1)
        dy *= self.reward_y_weight

        # Yaw axis (ROS rz) #
        curr_dist_ee_box_rz = self.get_relative_distance_ee_box_rz()
        drz = (self.prev_dist_ee_box_rz - curr_dist_ee_box_rz)

        drz /= self.reward_pose_norm_const
        drz = np.clip(drz, -1, 1)
        drz *= self.reward_pose_weight

        reward = dx + dy + drz

        return reward


    #############
    # Accessors #
    #############
    def get_ee_orient_euler(self):
        """
        Get the end-effector orientation as Euler angles in DART/ROS convention.

        :return: rx, ry, rz orientation of the ee in Euler (radians), DART/ROS convention
        """
        rot_mat = self.dart_sim.transform_ee.rotation()
        rx, ry, rz = dart.math.matrixToEulerXYZ(rot_mat)
        return rx, ry, rz

    def get_box_rotation_in_target_dart_coords_angle_axis(self):
        """
        Return in angle-axis DART/ROS coordinates the orientation of the box.

        Reads the box yaw from init_object_pose (ROS convention, index 5 = rz in radians)
        and converts to a 3x3 rotation matrix, then to logMap angle-axis.

        :return: a, b, c (in rad) angle-axis DART/ROS coordinates of the box orientation
        """
        # In ROS convention, yaw is rz (index 5 of [x, y, z, rx, ry, rz])
        object_rz = self.init_object_pose[5]  # radians
        object_rz = -object_rz if object_rz >= -np.pi/4 else -object_rz - np.pi/2
        r = R.from_euler('xyz', [-np.pi, 0, -np.pi + object_rz], degrees=False)
        r = r.as_matrix()
        a, b, c = dart.math.logMap(r)
        return a, b, c

    def get_object_pos(self):
        """
        Get the position of the box from Unity observation (now in ROS convention).

        :return: x, y, z coords of box (ROS: x=forward, y=left, z=up)
        """
        return self.unity_observation['object_position'][0], self.unity_observation['object_position'][1], self.unity_observation['object_position'][2]

    def get_object_orient(self):
        """
        Get the orientation of the box (ROS convention).
        Important: works for planar grasping envs only — assumes the box does not move during the episode.

        :return: rx, ry, rz orientation of box (ROS convention, radians)
        """
        return self.init_object_pose[3], self.init_object_pose[4], self.init_object_pose[5]

    def get_object_height(self):
        """
        Get the height of the box (ROS Z-up convention).

        :return: z coord of box (height above ground)
        """
        return self.unity_observation['object_position'][2]

    def get_object_pose(self):
        """
        Get the pose of the box (ROS convention).
        Important: works for planar grasping envs only — assumes the box does not move during the episode.

        :return: x, y, z, rx, ry, rz coords of box (ROS convention)
        """
        return self.unity_observation['object_position'][0], self.unity_observation['object_position'][1], self.unity_observation['object_position'][2], \
               self.init_object_pose[3], self.init_object_pose[4], self.init_object_pose[5]

    def get_collision_flag(self):
        """
            get the collision flag value

            :return: 0 (no collision) or 1 (collision)
        """
        return self.unity_observation['collision_flag']

    def get_relative_distance_ee_box_y(self):
        """
        Get the absolute lateral distance (ROS y-axis) from box to ee.

        :return: dist (float) lateral distance from ee to box
        """
        _, y_err, _ = self.get_error_ee_box_pos()
        return abs(y_err)

    def get_relative_distance_ee_box_x(self):
        """
        Get the absolute forward distance (ROS x-axis) from box to ee.

        :return: dist (float) forward distance from ee to box
        """
        x_err, _, _ = self.get_error_ee_box_pos()
        return abs(x_err)

    def get_relative_distance_ee_box_rz(self):
        """
        Get the absolute yaw distance (ROS rz) from box to ee.

        :return: dist (float) yaw distance from ee to box (radians)
        """
        rz_err = self.get_error_ee_box_rz()
        return abs(rz_err)

    def get_error_ee_box_pos(self):
        """
        Get the position error from box to ee in ROS convention.

        :return: x_err (forward), y_err (lateral), z_err (height) of box minus ee
        """
        object_x, object_y, object_z = self.get_object_pos()
        ee_x, ee_y, ee_z = self.get_ee_pos()
        return object_x - ee_x, object_y - ee_y, object_z - ee_z

    def get_error_ee_box_x_y(self):
        """
        Get the forward (x) and lateral (y) error from box to ee in ROS convention.

        :return: x_err (forward), y_err (lateral) of box minus ee
        """
        x_err, y_err, _ = self.get_error_ee_box_pos()
        return x_err, y_err

    def get_error_ee_box_x_y_normalized(self):
        """
        Get the normalized forward (x) and lateral (y) error from box to ee.
        Important: hard-coded normalization — adapt if the ee starts from a different initial position,
                   or the boxes are not spawned in front of the robot.

        :return: x_err_norm (forward/0.73), y_err_norm (lateral/1.4)
        """
        x_err, y_err = self.get_error_ee_box_x_y()
        return x_err / 0.73, y_err / 1.4

    def get_error_ee_box_rz(self):
        """
        Get the yaw (ROS rz) error from box to ee in radians.

        :return: rz_err of box minus ee (radians)
        """
        _, _, ee_rz = self.get_ee_orient_euler()

        ##########################################################
        # Transform ee and box rz (yaw) rotation to our needs   #
        # see the clip function implementation for more details  #
        ##########################################################
        box_rz = self.init_object_pose[5]  # ROS rz (already in radians)
        clipped_ee_rz, clipped_box_rz = self.clip_ee_rz_and_box_rz(ee_rz, box_rz)

        return clipped_box_rz - clipped_ee_rz

    def get_error_ee_box_rz_normalized(self):
        """
        Get the normalized yaw error from box to ee.

        :return: rz_err / (2*pi) normalized yaw error
        """
        return self.get_error_ee_box_rz() / (2 * np.pi)

    def clip_ee_rz_and_box_rz(self, ee_rz, box_rz):
        """
        Fix the yaw (rz) rotation for both box and end-effector.

        In the ROS convention, yaw is the rotation around the Z-axis (rz).
        - Input for the ee is the raw DART Euler rz component.
        - Input for the box is the rz returned from the boxGenerator (radians).
        - The ee starts at +-pi yaw rotation.
        - Some observations are returned with a sign change and should be corrected.

        Important: We assume that when the ee is at +-pi and the box at 0 rad,
                   the error is zero (no rotation needed — 'highest reward').

        Note: We define only one correct rotation direction for grasping:
              - Box rz in [-pi/2, -pi/4): ee should turn clock-wise
              - Box rz in [-pi/4, 0]: ee should turn counter-clock-wise

        Warning: If the ee starts from a different rotation than +-pi or the box
                 spawn range of [-pi/2, 0] rad is different — adapt.

        :param ee_rz: ee yaw rotation in radians (from DART Euler)
        :param box_rz: box yaw rotation in radians (from boxGenerator)

        :return: clipped_ee_rz corrected ee yaw in radians
        :return: clipped_box_rz corrected box yaw in radians
        """
        if (self.init_object_pose[5] >= -np.pi/4):  # Counter-clock-wise rotation
            if (ee_rz > 0):
                ee_rz *= -1
            else:
                ee_rz = -np.pi - (np.pi + ee_rz)

            clipped_box_rz = -np.pi - box_rz

        elif (self.init_object_pose[5] < -np.pi/4):  # Clock-wise rotation
            if (ee_rz < 0):
                ee_rz *= -1
            else:
                ee_rz = np.pi + (np.pi - ee_rz)

            clipped_box_rz = np.pi + (-np.pi / 2 - box_rz)

        clipped_ee_rz = ee_rz

        return clipped_ee_rz, clipped_box_rz

    def clip_rz(self, ee_rz):
        """
        Individual yaw clipping function. See clip_ee_rz_and_box_rz() for details.
        This variant is for agents that need to clip ee_rz and box_rz separately.

        :param ee_rz: ee yaw rotation in radians

        :return: clipped_ee_rz corrected ee yaw rotation (radians)
        """
        if (self.init_object_pose[5] >= -np.pi/4):
            if (ee_rz > 0):
                ee_rz *= -1
            else:
                ee_rz = -np.pi - (np.pi + ee_rz)

        elif (self.init_object_pose[5] < -np.pi/4):
            if (ee_rz < 0):
                ee_rz *= -1
            else:
                ee_rz = np.pi + (np.pi - ee_rz)

        return ee_rz

    def clip_box_rz(self, box_rz):
        """
        Individual box yaw clipping function. See clip_ee_rz_and_box_rz() for details.
        This variant is for agents that need to clip ee_rz and box_rz separately.

        :param box_rz: box yaw rotation in radians (from boxGenerator)

        :return: clipped_box_rz corrected box yaw rotation in radians
        """
        if (self.init_object_pose[5] >= -np.pi/4):
            clipped_box_rz = -np.pi - box_rz
        elif (self.init_object_pose[5] < -np.pi/4):
            clipped_box_rz = np.pi + (-np.pi / 2 - box_rz)

        return clipped_box_rz
