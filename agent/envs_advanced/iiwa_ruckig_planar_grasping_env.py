"""
A model-based numerical planar grasping Env class inheriting from the IiwaNumericalPlanarGraspingEnv for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment

Note: update_action() receives velocities genereted by ruckig. For more refer to ruckig_planar_model.py

Unity is used as the main simulator for physics/rendering computations.
The Unity interface receives joint velocities as commands and returns joint positions and velocities

DART is used to calculate inverse kinematics of the iiwa chain.
DART changes the agent action space from the joint space to the cartesian space (position-only or pose/SE(3)) of the end-effector.

action_by_pd_control method can be called to implement a Proportional-Derivative control law instead of an RL policy.

Note: Coordinates in the Unity simulator are different from the ones in DART which used here:
The mapping is [X, Y, Z] of Unity is [-y, z, x] of DART
"""

import numpy as np
import math

from gym import spaces

from envs_advanced.iiwa_numerical_planar_grasping_env import IiwaNumericalPlanarGraspingEnv

class IiwaRuckigPlanarGraspingEnv(IiwaNumericalPlanarGraspingEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns, state_type, enable_render=False, task_monitor=False,
                 with_objects=False, target_mode="None", goal_type="box", randomBoxesGenerator=None, 
                 joints_safety_limit=10, max_joint_vel=20, max_ee_cart_vel=0.035, max_ee_cart_acc =10, max_ee_rot_vel=0.15, max_ee_rot_acc=10,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2], noise_enable_rl_obs=False, noise_rl_obs_ratio=0.05,
                 reward_dict=None, agent_kp=0.5, agent_kpr=1.5, threshold_p_model_based=0.01,
                 robotic_tool="3_gripper", env_id=0):

        ################################################################################################
        # the init of the parent class should be always called, this will in the end call reset() once #
        ################################################################################################
        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns, state_type=state_type, enable_render=enable_render, task_monitor=task_monitor,
                         with_objects=with_objects, target_mode=target_mode, goal_type=goal_type, randomBoxesGenerator=randomBoxesGenerator,
                         joints_safety_limit=joints_safety_limit, max_joint_vel=max_joint_vel, max_ee_cart_vel=max_ee_cart_vel, max_ee_cart_acc=max_ee_cart_acc, max_ee_rot_vel=max_ee_rot_vel, max_ee_rot_acc=max_ee_rot_acc,
                         random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions,noise_enable_rl_obs=False,noise_rl_obs_ratio=0.05,
                         reward_dict=reward_dict,agent_kp=agent_kp, agent_kpr=agent_kpr, robotic_tool=robotic_tool, env_id=env_id)

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

            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

            self.action_space = spaces.Box(low=np.asarray([-self.MAX_EE_CART_VEL[1], -self.MAX_EE_CART_VEL[0]]), 
                                           high=np.asarray([self.MAX_EE_CART_VEL[1], self.MAX_EE_CART_VEL[0]]), 
                                           dtype=np.float32)

        else: # 3DoF control, rotation is active
            ##################
            # Viewer-related #
            ##################
            self.observation_indices['joint_pos'] = 0 # Unused 
            self.observation_indices['obs_len'] = 6

            ###############
            # Gym-related #
            ###############

            # X, Y, RY in unity #
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

            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

            self.action_space = spaces.Box(low=np.asarray([-self.MAX_EE_ROT_VEL[2], -self.MAX_EE_CART_VEL[1], -self.MAX_EE_CART_VEL[1]]), 
                                           high=np.asarray([self.MAX_EE_ROT_VEL[2], self.MAX_EE_CART_VEL[1], self.MAX_EE_CART_VEL[0]]), 
                                           dtype=np.float32)

        ########################################################################
        # Set-up the task_monitor. The re-implementation resides in this class #
        # since the dims of action, and obs spaces have been overridden above  #
        ########################################################################
        if self.task_monitor:
            self._create_task_monitor()

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

        #################
        # Unity in DART #
        # Yu = Zd       #
        # Xu = -Yd      #
        # Zu = Xd       #
        #################

        ##############################################################################
        # Reset the P-controller - save the initial pose of the manipulator          #
        # for moving in a planar manner during the episode e.g. keep the same height #
        ##############################################################################
        if (self.time_step == 1):
            self.reset_agent_p_controller()

        # the lines below should stay as it is
        self.action_state = action
        env_action = action

        # Rotation is controlled by the model-based agent
        if(self.action_space_dimension == 3):
            task_vel = np.zeros(3)
        else:
            task_vel = np.zeros(2)

        ############################################################################
        # The model-based agent controls the x, y and rotational z-axis (optional) #
        ############################################################################

        ###########################################################################################
        # Calculate the errors for the P-controller. Axis not controlled by the model-based agent #
        # Important: P-controller expects dart coordinates                                        #
        ###########################################################################################
        _, ee_y, _ = self.get_ee_pos_unity()
        z_diff = self.target_z_dart - ee_y                                                                    # Height 
        curr_quat = self.get_rot_ee_quat()                                                                    # Current orientation of the ee in quaternions
        rx_diff, ry_diff, rz_diff = self.get_rot_error_from_quaternions(self.target_rot_quat_dart, curr_quat) # Orientation error from the target pose
        ################################################################################################

        # Get poses #
        curr_pose = self.get_ruckig_current_pose()  # Current ee pose
        target_pose = self.get_ruckig_target_pose() # Current pose of the box - does not change during the episode

        ##############
        # Dart cords #
        ##############
        if(self.action_space_dimension == 3): # Rotation is active - 3DoF control by the model-based agent 
            task_vel[0] = env_action[0]
            task_vel[1] = env_action[1]
            task_vel[2] = env_action[2]

            # Distance to the goal #
            dist = np.linalg.norm(np.array([rx_diff, ry_diff, z_diff, target_pose[0] - curr_pose[0], target_pose[1] - curr_pose[1], target_pose[2] - curr_pose[2]]))

            if(dist < self.threshold_p_model_based): # Threshold is reached - stop, if active only
                joint_vel = np.zeros(7)
            else:
                ###############################################################################################
                # P-controller + inverse kinematics                                                           #
                #   - The DoF that are controlled by the model-based agent are unaffected by the P-controller #
                #   - see config_p_controller dictionary                                                      #                        
                ###############################################################################################
                joint_vel = self.action_by_p_controller_custom(rx_diff, ry_diff, task_vel[0],
                                                               task_vel[1], task_vel[2], z_diff,
                                                               self.agent_kpr, self.agent_kp,
                                                               self.config_p_controller)
        else: # Rotation is not active - 2DoF control by the model-based agent
            task_vel[0] = env_action[0]
            task_vel[1] = env_action[1]

            dist = np.linalg.norm(np.array([rx_diff, ry_diff, z_diff, rz_diff, target_pose[0] - curr_pose[0], target_pose[1] - curr_pose[1]]))

            if(dist < self.threshold_p_model_based): # Threshold is reached
                joint_vel = np.zeros(7)
            else:
                joint_vel = self.action_by_p_controller_custom(rx_diff, ry_diff, rz_diff,
                                                               task_vel[0], task_vel[1], z_diff,
                                                               self.agent_kpr, self.agent_kp,
                                                               self.config_p_controller)

        ###################################################################################################
        # Gripper is not controlled via the model-based agent - manual actions - see simulator_vec_env.py # 
        ###################################################################################################
        unity_action = np.append(joint_vel, [float(0.0)])

        return unity_action

    ###########
    # Helpers #
    ###########
    def get_ruckig_current_pose(self):
        """
            get the current state pose of the ee in dart coordinates - needed in the get_state()
                - refer also to the clip methods in iiwa_numerical_planar_grasping_env.py

           :return: ee_rz_d, ee_x_d, ee_y_d - or the first value is skipped if only 2DoF are controlled by the model-based agent
        """
        state = np.empty(0)

        # Rotation part of the ee - only if 3DoF are active #
        if(self.action_space_dimension == 3):
            _, ee_ry, _ = self.get_ee_orient_unity()
            ee_ry = self.clip_ry(ee_ry)
            state = np.append(state, np.asarray([ee_ry]))

        # x and z position of the ee #
        x, _, z = self.get_ee_pos_unity()
        state = np.append(state, np.array([z, -x]))

        return state

    def get_ruckig_target_pose(self):
        """
            get the target state pose of the box - needed in the get_state()
                - refer also to the clip methods in iiwa_numerical_planar_grasping_env.py

           :return: rz_box_d, x_box_d, y_box_d - or first value is skipped if only 2DoF are controlled by the model-based agent

        """
        state = np.empty(0)

        # Rotation part of the box - if 3DoF are active #
        if(self.action_space_dimension == 3):
            box_ry_rad = np.deg2rad(self.init_object_pose_unity[4])
            box_ry_rad = self.clip_box_ry(box_ry_rad)
            state = np.append(state, np.asarray([box_ry_rad]))

        # x and z position of the box #
        state = np.append(state, np.array([self.init_object_pose_unity[2], -self.init_object_pose_unity[0]]))

        return state

    ###########
    # Monitor #
    ###########
    def _create_task_monitor(self, plot_joint_position=True, plot_joint_velocity=True, plot_joint_acceleration=False, plot_joint_torque=True, plot_agent_state=True, plot_agent_action=True, plot_agent_reward=True):
        """
            Override the task monitor definition since we have changed the get_state() function and the actions dimensions
            refer to iiwa_dart.py and task_monitor.py for more
        """
        from PySide2.QtWidgets import QApplication
        from utils.task_monitor import TaskMonitor

        if not QApplication.instance():
            self.monitor_app = QApplication([])
        else:
            self.monitor_app = QApplication.instance()

        self.monitor_n_states = 5
        state_chart_categories = ['RESET', 'X_EE', 'Y_EE', 'X_B', 'Y_B']
        if self.action_space_dimension == 3:
            self.monitor_n_states += 2
            state_chart_categories = ['RESET', 'RZ_EE', 'X_EE', 'Y_EE', 'RZ_B', 'X_B', 'Y_B']

        self.monitor_window = \
            TaskMonitor(plot_joint_position=plot_joint_position,
                        param_joint_position={'dim': self.n_links,
                                              'min': np.rad2deg(self.MIN_JOINT_POS),
                                              'max': np.rad2deg(self.MAX_JOINT_POS),
                                              'cat': [str(1 + value) for value in list(range(self.n_links))],
                                              'zone': 10.0,
                                              'title': "Joint Position [" + u"\N{DEGREE SIGN}" + "]"},
                        plot_joint_velocity=plot_joint_velocity,
                        param_joint_velocity={'dim': self.n_links,
                                              'min': np.rad2deg(-self.MAX_JOINT_VEL),
                                              'max': np.rad2deg(+self.MAX_JOINT_VEL),
                                              'cat': [str(1 + value) for value in list(range(self.n_links))],
                                              'zone': 5.0,
                                              'title': "Joint Velocity [" + u"\N{DEGREE SIGN}" + "/s]"},
                        plot_joint_acceleration=plot_joint_acceleration,
                        param_joint_acceleration={'dim': self.n_links,
                                                  'min': np.rad2deg(-self.MAX_JOINT_ACC),
                                                  'max': np.rad2deg(+self.MAX_JOINT_ACC),
                                                  'cat': [str(1 + value) for value in list(range(self.n_links))],
                                                  'title': "Joint Acceleration [" + u"\N{DEGREE SIGN}" + "/s^2]"},
                        plot_joint_torque=plot_joint_torque,
                        param_joint_torque={'dim': self.n_links,
                                            'min': -self.MAX_JOINT_TORQUE,
                                            'max': +self.MAX_JOINT_TORQUE,
                                            'cat': [str(1 + value) for value in list(range(self.n_links))],
                                            'zone': 5.0,
                                            'title': "Joint Torque [Nm]"},
                        plot_agent_state=plot_agent_state,
                        param_agent_state={'dim': self.monitor_n_states,
                                           'min': self.observation_space.low[0:self.monitor_n_states],
                                           'max': self.observation_space.high[0:self.monitor_n_states],
                                           'cat': state_chart_categories,
                                           'title': "Reaching Target Error"},
                        plot_agent_action=plot_agent_action,
                        param_agent_action={'dim': self.action_space_dimension,
                                            'min': self.action_space.low,
                                            'max': self.action_space.high,
                                            'cat': [str(1 + value) for value in
                                                    list(range(self.action_space_dimension))],
                                            'title': "Normalized Command"},
                        plot_agent_reward=plot_agent_reward,
                        param_agent_reward={'dim': 1,
                                            'min': -3.0,
                                            'max': +3.0,
                                            'cat': ['r'],
                                            'title': "Step Reward"},
                        )

        self.monitor_window.show()
        self.monitor_window.correct_size()
