"""
A model-based numerical planar grasping Env class inheriting from the IiwaNumericalPlanarGraspingEnv for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment

Note: update_action() receives velocities genereted by ruckig. For more refer to ruckig_planar_model.py

Unity is used as the main simulator for physics/rendering computations.
The Unity interface receives joint velocities as commands and returns joint positions and velocities

DART is used to calculate inverse kinematics of the iiwa chain.
DART changes the agent action space from the joint space to the cartesian space (position-only or pose/SE(3)) of the end-effector.

action_by_pd_control method can be called to implement a Proportional-Derivative control law instead of an RL policy.

Note: All data exchanged with Unity is now in DART/ROS convention (x=forward, y=left, z=up).
Coordinate conversions are handled entirely on the Unity (C#) side.
"""

import numpy as np
import math

from gym import spaces

from envs_advanced.iiwa_numerical_planar_grasping_env import IiwaNumericalPlanarGraspingEnv

class IiwaRuckigPlanarGraspingEnv(IiwaNumericalPlanarGraspingEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns, state_type, enable_render=False,
                 with_objects=False, target_mode="None", goal_type="box", randomBoxesGenerator=None, 
                 joints_safety_limit=10, max_joint_vel=20, max_ee_cart_vel=0.035, max_ee_cart_acc =10, max_ee_rot_vel=0.15, max_ee_rot_acc=10,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2], noise_enable_rl_obs=False, noise_rl_obs_ratio=0.05,
                 reward_dict=None, agent_kp=0.5, agent_kpr=1.5, threshold_p_model_based=0.01,
                 robotic_tool=None, end_effector_model=None, manipulator_config=None, manipulator_gym_config=None, env_id=0):

        ################################################################################################
        # the init of the parent class should be always called, this will in the end call reset() once #
        ################################################################################################
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

        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns, state_type=state_type, enable_render=enable_render,
                        with_objects=with_objects, target_mode=target_mode, goal_type=goal_type, randomBoxesGenerator=randomBoxesGenerator,
                        joints_safety_limit=joints_safety_limit, max_joint_vel=max_joint_vel, max_ee_cart_vel=max_ee_cart_vel, max_ee_cart_acc=max_ee_cart_acc, max_ee_rot_vel=max_ee_rot_vel, max_ee_rot_acc=max_ee_rot_acc,
                        random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions,noise_enable_rl_obs=False,noise_rl_obs_ratio=0.05,
                        reward_dict=reward_dict,agent_kp=agent_kp, agent_kpr=agent_kpr, robotic_tool=robotic_tool, manipulator_config=manipulator_config, manipulator_gym_config=manipulator_gym_config, env_id=env_id)

        #############################################################################################
        # Joints are normalized in [-1, 1] at the get_state() function - for planar envs            #
        # The below settings are necessary to be set so that in iiwa_dart.py when                   #
        # calling the self.observation_space.sample() function, we denormalize the joints positions #
        #############################################################################################
        self.normalized_rl_obs = True
        self.observation_indices = {'obs_len': 0}

        # Whether to apply a threshold to the model-based controller - see update_action() #
        self.threshold_p_model_based = threshold_p_model_based 

        ########################################
        # Action and observation space related #
        ########################################
        if np.isclose(np.asarray([self.reward_pose_weight]), np.asarray([0])): # No rotation control - 2DoF
            ##################
            # Viewer-related #
            ##################
            self.observation_indices['joint_pos'] = 0 # Unused -> no joints position information in the state
            self.observation_indices['obs_len'] = 5

            ###############
            # Gym-related #
            ###############
            self.action_space_dimension = 2
            self.observation_space_dimension = 5   # [reset, x_ee_d, y_ee_d, x_box_d, y_box_d] - see get_state() 

            # 4 DoF that are being controlled by the P-controller - keep moving planar during the episode #
            self.config_p_controller = {"kpr_x": 1,
                                        "kpr_y": 1,
                                        "kpr_z": 1,
                                        "kp_x": 0,  # Controlled by the model-based agent
                                        "kp_y": 0,
                                        "kp_z": 1
                                       }

            tool_length = 0.2 # Tolerance

            ##########################################################################
            # Important: adapt if the ranges change in the randomBoxesGenerator, or  #
            # e.g. the manipulator is in a different initial planar position,    or  #
            #      the boxes are spawned behind/left/right and not in-front          #     
            ##########################################################################
            low = np.asarray([0, -(0.95 + tool_length), -(0.95 + tool_length), -(0.95 + tool_length), -(0.95 + tool_length)])

            high = np.asarray([1, 0.95 + tool_length, 0.95 + tool_length, 0.95 + tool_length, 0.95 + tool_length])

            low_multi = np.tile(low, self.robot_count)
            high_multi = np.tile(high, self.robot_count)
            self.observation_space = spaces.Box(low=low_multi, high=high_multi, dtype=np.float32)

            single_low = np.asarray([-self.MAX_EE_CART_VEL[1], -self.MAX_EE_CART_VEL[0]], dtype=np.float32)
            single_high = np.asarray([self.MAX_EE_CART_VEL[1], self.MAX_EE_CART_VEL[0]], dtype=np.float32)
            self.action_space = spaces.Box(
                low=np.tile(single_low, (self.robot_count, 1)),
                high=np.tile(single_high, (self.robot_count, 1)),
                dtype=np.float32,
            )

        else: # 3DoF control, rotation is active
            ##################
            # Viewer-related #
            ##################
            self.observation_indices['joint_pos'] = 0 # Unused 
            self.observation_indices['obs_len'] = 6

            ###############
            # Gym-related #
            ###############

            # x (forward), y (lateral), rz (yaw) in ROS #
            self.action_space_dimension = 3
            self.observation_space_dimension = 6      # [reset, rz_ee_d, x_ee_d, y_ee_d, rz_box_d, x_box_d, y_box_d] - see get_state()

            # 3 DoF that are being controlled by the P-controller - keep moving planar during the episode #
            self.config_p_controller = {"kpr_x": 1,
                                        "kpr_y": 1,
                                        "kpr_z": 0,
                                        "kp_x": 0,
                                        "kp_y": 0,
                                        "kp_z": 1
                                    }

            tool_length = 0.2 # Tolerance
            low = np.asarray([0, -2*np.pi, -(0.95 + tool_length), -(0.95 + tool_length), -2*np.pi, -(0.95 + tool_length), -(0.95 + tool_length)])

            high = np.asarray([1, 2*np.pi, 0.95 + tool_length, 0.95 + tool_length, 2*np.pi, 0.95 + tool_length, 0.95 + tool_length])

            low_multi = np.tile(low, self.robot_count)
            high_multi = np.tile(high, self.robot_count)
            self.observation_space = spaces.Box(low=low_multi, high=high_multi, dtype=np.float32)

            single_low = np.asarray([-self.MAX_EE_ROT_VEL[2], -self.MAX_EE_CART_VEL[1], -self.MAX_EE_CART_VEL[1]], dtype=np.float32)
            single_high = np.asarray([self.MAX_EE_ROT_VEL[2], self.MAX_EE_CART_VEL[1], self.MAX_EE_CART_VEL[0]], dtype=np.float32)
            self.action_space = spaces.Box(
                low=np.tile(single_low, (self.robot_count, 1)),
                high=np.tile(single_high, (self.robot_count, 1)),
                dtype=np.float32,
            )

    def get_state(self):
        """
            defines the environment state with shape(num_envs, 5 or 6) - depending on the active DoF (rotation control is active)
                Format: [reset, rz_ee_d, x_ee_d, y_ee_d, rz_box_d, x_box_d, y_box_d]
                        - Should reset, rotation of ee in dart, x position of the ee in dart, etc
                        - box pose is used only once (at reset_ruckig_model()) as the box does not move during the planar episode
                        - Not normalized in [-1, 1] as in most gym agents

           :return: state for the model-based agent (no RL training)
        """
        state = np.empty(0)
        if(self.init_object_pose == None):
            return state

        # Reset ruckig model (flag). Set target pose of ee only once       #
        # at the first step - see how it is used in ruckig_planar_model.py #
        if(self.time_step == 1):
            reset_ruckig_model = 1
        else:
            reset_ruckig_model = 0

        state = np.append(state, np.asarray([reset_ruckig_model]))

        state = np.append(state, self.get_ruckig_current_pose()) # Current pose of the ee
        state = np.append(state, self.get_ruckig_target_pose())  # Pose of the box - changes only per episode. Assume the box is not moved during the planar motion

        # the lines below should stay as it is
        self.observation_state = np.array(state)

        return self.observation_state

    def update_action(self, action):
        """
           converts env action to the required unity action by using inverse kinematics from DART
               - action dim: 3, or 2 (in case of no rotation control)
               - the remaining DoF are controlled by the P-controller.
               - the Gripper is not controlled by the RL agent only during the manual and at the end of the episode - see simulator_vec_env.py

           :param action: The action vector decided by the model-based model
                              - Important: the input action is not normalized in [-1, 1] in this case - m/sec

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
            # Rotation is controlled by the model-based agent
            task_vel = np.zeros(3 if self.action_space_dimension == 3 else 2)

            ############################################################################
            # The model-based agent controls the x, y and rotational z-axis (optional) #
            ############################################################################

            ###########################################################################################
            # Calculate the errors for the P-controller. Axis not controlled by the model-based agent #
            # Important: P-controller expects dart coordinates                                        #
            ###########################################################################################
            ee_height = self.get_ee_pos()[2]  # ROS z = height
            z_diff = self.target_z_dart - ee_height                                                               # Height 
            curr_quat = self.get_rot_ee_quat()                                                                    # Current orientation of the ee in quaternions
            rx_diff, ry_diff, rz_diff = self.get_rot_error_from_quaternions(self.target_rot_quat_dart, curr_quat) # Orientation error from the target pose
            ################################################################################################

            # Get poses #
            curr_pose = self.get_ruckig_current_pose()  # Current ee pose
            target_pose = self.get_ruckig_target_pose() # Current pose of the box - does not change during the episode

            ##############
            # Dart cords #
            ##############
            if self.action_space_dimension == 3: # Rotation is active - 3DoF control by the model-based agent
                task_vel[0] = env_action[0]
                task_vel[1] = env_action[1]
                task_vel[2] = env_action[2]

                # Distance to the goal #
                dist = np.linalg.norm(np.array([rx_diff, ry_diff, z_diff, target_pose[0] - curr_pose[0], target_pose[1] - curr_pose[1], target_pose[2] - curr_pose[2]]))
                if dist < self.threshold_p_model_based: # Threshold is reached - stop, if active only
                    joint_vel = np.zeros(7)
                else:
                    ###############################################################################################
                    # P-controller + inverse kinematics                                                           #
                    #   - The DoF that are controlled by the model-based agent are unaffected by the P-controller #
                    #   - see config_p_controller dictionary                                                      #                        
                    ###############################################################################################
                    joint_vel = self.action_by_p_controller_custom(
                        rx_diff, ry_diff, task_vel[0], task_vel[1], task_vel[2], z_diff,
                        self.agent_kpr, self.agent_kp, self.config_p_controller
                    )
            else: # Rotation is not active - 2DoF control by the model-based agent
                task_vel[0] = env_action[0]
                task_vel[1] = env_action[1]
                dist = np.linalg.norm(np.array([rx_diff, ry_diff, z_diff, rz_diff, target_pose[0] - curr_pose[0], target_pose[1] - curr_pose[1]]))
                if dist < self.threshold_p_model_based: # Threshold is reached
                    joint_vel = np.zeros(7)
                else:
                    joint_vel = self.action_by_p_controller_custom(
                        rx_diff, ry_diff, rz_diff, task_vel[0], task_vel[1], z_diff,
                        self.agent_kpr, self.agent_kp, self.config_p_controller
                    )

            ###################################################################################################
            # Gripper is not controlled via the model-based agent - manual actions - see simulator_vec_env.py # 
            ###################################################################################################
            unity_actions.append(np.append(joint_vel, [float(0.0)]))

        return np.asarray(unity_actions, dtype=np.float32)

    ###########
    # Helpers #
    ###########
    def get_ruckig_current_pose(self):
        """
        Get the current state pose of the ee in DART/ROS coordinates — needed in get_state().
        Refer also to the clip methods in iiwa_numerical_planar_grasping_env.py.

        :return: ee_rz, ee_x, ee_y — or the first value is skipped if only 2DoF are controlled
        """
        state = np.empty(0)

        # Yaw (rz) rotation of the ee — only if 3DoF are active #
        if(self.action_space_dimension == 3):
            _, _, ee_rz = self.get_ee_orient_euler()
            ee_rz = self.clip_rz(ee_rz)
            state = np.append(state, np.asarray([ee_rz]))

        # Forward (x) and lateral (y) position of the ee in ROS convention #
        x, y, _ = self.get_ee_pos()
        state = np.append(state, np.array([x, y]))

        return state

    def get_ruckig_target_pose(self):
        """
        Get the target state pose of the box — needed in get_state().
        Refer also to the clip methods in iiwa_numerical_planar_grasping_env.py.

        :return: rz_box, x_box, y_box — or first value is skipped if only 2DoF are controlled
        """
        state = np.empty(0)

        # Yaw (rz) rotation of the box — if 3DoF are active #
        if(self.action_space_dimension == 3):
            box_rz_rad = self.init_object_pose[5]  # ROS rz (index 5)
            box_rz_rad = self.clip_box_rz(box_rz_rad)
            state = np.append(state, np.asarray([box_rz_rad]))

        # Forward (x) and lateral (y) position of the box in ROS convention #
        state = np.append(state, np.array([self.init_object_pose[0], self.init_object_pose[1]]))

        return state
