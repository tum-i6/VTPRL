"""
A numerical planar grasping Env class inheriting from the IiwaSampleEnv for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment

Unity is used as the main simulator for physics/rendering computations.
The Unity interface receives joint velocities as commands and returns joint positions and velocities

DART is used to calculate inverse kinematics of the iiwa chain.
DART changes the agent action space from the joint space to the cartesian space (position-only or pose/SE(3)) of the end-effector.

Notes: - Coordinates in the Unity simulator are different from the ones in DART which used here:
         The mapping is [X, Y, Z] of Unity is [-y, z, x] of DART
       - In numerical planar grasping envs, a P-controller (agent controller) keeps fixed the uncontrolled DoF during the episode. E.g. height of the ee
       - At the end of the episode, manual actions help the agent to grasp the box
         In that case, the P-controller (manual_actions controller) does not correct the controlled DoF by the RL agent. It keeps them fixed. E.g. rotation of the ee
         see simulator_vec_env.py for more details
       - gym functions use UNITY coordinates - e.g. get_state(), get_reward()
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import math
import dartpy as dart
from gym import spaces

from envs.iiwa_sample_env import IiwaSampleEnv

class IiwaNumericalPlanarGraspingEnv(IiwaSampleEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns, state_type, enable_render=False, task_monitor=False,
                 with_objects=False, target_mode="None", goal_type="box", randomBoxesGenerator=None,
                 joints_safety_limit=10, max_joint_vel=20, max_ee_cart_vel=0.035, max_ee_cart_acc =10, max_ee_rot_vel=0.15, max_ee_rot_acc=10,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2], noise_enable_rl_obs=False, noise_rl_obs_ratio=0.05,
                 reward_dict=None, agent_kp=0.5, agent_kpr=1.5,
                 robotic_tool="3_gripper", env_id=0):

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
        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns, state_type=state_type, enable_render=enable_render, task_monitor=task_monitor, 
                         with_objects=with_objects, target_mode=target_mode,  goal_type=goal_type, joints_safety_limit=joints_safety_limit, max_joint_vel=max_joint_vel, max_ee_cart_vel=max_ee_cart_vel, 
                         max_ee_cart_acc=max_ee_cart_acc, max_ee_rot_vel=max_ee_rot_vel, max_ee_rot_acc=max_ee_rot_acc, random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions,
                         robotic_tool=robotic_tool, env_id=env_id)

        # Box dataset to spawn boxes #
        self.randomBoxesGenerator = randomBoxesGenerator

        # Keep object initial pose. Easier calculations for rotations - box does not moved during the planar episode #
        self.init_object_pose_unity = None

        ###################################################################
        # Reward related                                                  #
        ###################################################################
        self.reward_collision = reward_dict["reward_collision"]              # Value
        self.reward_height_goal = reward_dict["reward_height_goal"]          # Height in meters

        self.reward_pose_weight = reward_dict["reward_pose_weight"]          # Weight for the pose error displacement
        self.reward_pose_norm_const = reward_dict["reward_pose_norm_const"]  # Normalization constant for the pose error displacement

        self.reward_x_weight = reward_dict["reward_x_weight"]
        self.reward_x_norm_const = reward_dict["reward_x_norm_const"]

        self.reward_z_weight = reward_dict["reward_z_weight"]
        self.reward_z_norm_const = reward_dict["reward_z_norm_const"]

        self.collision_flag = False                                          # Give collision reward one once per episode

        # Save previous distance ee/box for calculting the displacements - see reward definition #
        self.prev_dist_ee_box_z = np.inf
        self.prev_dist_ee_box_x = np.inf
        self.prev_dist_ee_boxry = np.inf
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

            # X, and Z in unity - no rotation #
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

            # X, Y, and RY in unity #
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
        self.action_space = spaces.Box(-np.ones(self.action_space_dimension), np.ones(self.action_space_dimension), dtype=np.float32)

        self.observation_space = spaces.Box(-np.ones(self.observation_space_dimension), np.ones(self.observation_space_dimension), dtype=np.float32)

        ########################################################################
        # Set-up the task_monitor. The re-implementation resides in this class #
        # since the dims of action, and obs spaces have been overridden above  #
        ########################################################################
        if self.task_monitor:
            self._create_task_monitor()

    def get_state(self):
        """
            defines the environment state:
                Format: [x_error, y_error, ry_error, j1, .., j7] normalized in [-1, 1]
                        - error from the target pose: ee to the target box (x, y, ry axis)
                        - joints positions
                        - unity coords system

                Note: if ry rotation is not controlled by the RL agent, then the state will not include the ry_error part

           :return: observation state for the policy training.
        """
        state = np.empty(0)

        # Relative position normalized errors of ee to the box #
        dx_ee_b, dz_ee_b = self.get_error_ee_box_x_z_normalized_unity()

        # Add to state #
        state = np.append(state, np.array([dx_ee_b, dz_ee_b]))

        # Rotation control is active #
        if(self.action_space_dimension == 3):
            # Normalized rotation error - y axis #
            dry_ee_b = self.get_error_ee_box_ry_normalized_unity()

            # Add to state #
            state = np.append(state, np.array([dry_ee_b]))

        # Get joints positions #
        joint_positions = self.dart_sim.chain.getPositions()
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
        if(self.get_object_height_unity() >= self.reward_height_goal and self.collided_env != 1):
            return True

        return False

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
        env_action = np.clip(action, -1., 1.)

        # Rotation is controlled by the RL-agent
        if(self.action_space_dimension == 3):
            task_vel = np.zeros(3)
        else:
            task_vel = np.zeros(2)

        ###################################################################
        # The RL agent controls the x, y and rotational z-axis (optional) #
        ###################################################################

        ##################################################################################
        # Calculate the errors for the P-controller. Axis not controlled by the RL-agent #
        # Important: P-controller expects dart coordinates                               #
        ##################################################################################
        _, ee_y, _ = self.get_ee_pos_unity()
        z_diff = self.target_z_dart - ee_y                                                                    # Height 
        curr_quat = self.get_rot_ee_quat()                                                                    # Current orientation of the ee in quaternions
        rx_diff, ry_diff, rz_diff = self.get_rot_error_from_quaternions(self.target_rot_quat_dart, curr_quat) # Orientation error from the target pose
        ################################################################################################

        if(self.action_space_dimension == 3): # Rotation is active - 3DoF control by the RL agent
            task_vel[0] = self.MAX_EE_VEL[2] * env_action[0]
            task_vel[1] = self.MAX_EE_VEL[3] * env_action[1]
            task_vel[2] = self.MAX_EE_VEL[4] * env_action[2]

            ######################################################################################
            # P-controller + inverse kinematics                                                  #
            #   - The DoF that are controlled by the RL-agent are unaffected by the P-controller #
            #   - see config_p_controller dictionary                                             #                        
            ######################################################################################
            joint_vel = self.action_by_p_controller_custom(rx_diff, ry_diff, task_vel[0],
                                                           task_vel[1], task_vel[2], z_diff,
                                                           self.agent_kpr, self.agent_kp,
                                                           self.config_p_controller)
        else: # Rotation is not active - 2DoF control by the RL agent
            task_vel[0] = self.MAX_EE_VEL[3] * env_action[0]
            task_vel[1] = self.MAX_EE_VEL[4] * env_action[1]

            joint_vel = self.action_by_p_controller_custom(rx_diff, ry_diff, rz_diff,
                                                           task_vel[0], task_vel[1], z_diff,
                                                           self.agent_kpr, self.agent_kp,
                                                           self.config_p_controller)

        ##########################################################################################
        # Gripper is not controlled via the RL-agent - manual actions - see simulator_vec_env.py # 
        ##########################################################################################
        unity_action = np.append(joint_vel, [float(0.0)])

        return unity_action

    def update(self, observation, time_step_update=True):
        """
            converts the unity observation to the required env state defined in get_state()
            with also computing the reward value from the get_reward(...) and done flag,
            it increments the time_step, and outputs done=True when the environment should be reset

            important: it also updates the dart kinematic chain of the robot using the new unity simulator observation.
                       always call this function, once you have send a new command to unity to synchronize the agent environment

            :param observation: is the observation received from the Unity simulator within its [X,Y,Z] coordinate system
                                'joint_values': indices [0:7],
                                'joint_velocities': indices [7:14],
                                'ee_position': indices [14:17],
                                'ee_orientation': indices [17:20],
                                'target_position': indices [20:23],
                                'target_orientation': indices [23:26],
                                'object_position': indices [26:29],
                                'object_orientation': indices [29:32],
                                'gripper_position': indices [32:33],
                                'collision_flag': indices [33:34],

            :param time_step_update: whether to increase the time_step of the agent - during manual actions call with False

            :return: The state, reward, episode termination flag (done), and an empty info dictionary
        """

        self.current_obs = observation['Observation']

        # Collision or joints limits overpassed                                      #
        # env will not be reseted - wait until the end of the episoe for planar envs #
        if observation['Observation'][-1] == 1.0 or self.joints_limits_violation():
            self.collided_env = 1

        # the method below handles synchronizing states of the DART kinematic chain with the #
        # observation from Unity hence it should be always called                            #
        self._unity_retrieve_joint_values(observation['Observation'])

        # Class attributes below exist in the parent class, hence the names should not be changed
        if(time_step_update == True):
            self.time_step += 1

        state = self.get_state()
        reward = self.get_reward(self.action_state)
        done = bool(self.get_terminal())
        info = {"success": False}                   # Episode was successful. It is set at simulator_vec_env.py before reseting


        # Keep track the previous distance of the ee to the box - used in the reward function #
        self.prev_dist_ee_box_z = self.get_relative_distance_ee_box_z_unity()
        self.prev_dist_ee_box_x = self.get_relative_distance_ee_box_x_unity()
        self.prev_dist_ee_box_ry = self.get_relative_distance_ee_box_ry_unity() 

        self.prev_action = self.action_state

        return state, reward, done, info

    ###########
    # Monitor #
    ###########
    def _create_task_monitor(self, plot_joint_position=True, plot_joint_velocity=True, plot_joint_acceleration=False, plot_joint_torque=True, 
                                   plot_agent_state=True, plot_agent_action=True, plot_agent_reward=True):
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

        self.monitor_n_states = 2
        state_chart_categories = ['X', 'Z']
        if self.action_space_dimension == 3:
            self.monitor_n_states += 1
            state_chart_categories = ['X', 'Z', 'RY']

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

    def reset(self, temp=False):
        """
            resets the DART-gym environment and creates the reset_state vector to be sent to Unity

            :param temp: not relevant

            :return: The initialized state
        """
        self.current_obs = np.ones(34)

        ############################################################
        # Spawn the next box and fix the target (for task monitor) #
        ############################################################
        object_X, object_Y, object_Z, object_RX, object_RY, object_RZ = self.randomBoxesGenerator()
        self.init_object_pose_unity = [object_X, object_Y, object_Z, object_RX, object_RY, object_RZ]

        ################################################################################################
        # Align the target with the box. This is done for visualization purposes for the dart viewer   #
        # In the Unity simulator, however, we spawn the target far away                                #
        # Vision-based models might get confused if there is in the image a red target - if the target #
        # is used in vision-based envs -> adapt                                                        #
        ################################################################################################

        # Align with the box
        target_object_X, target_object_Y, target_object_Z = self.init_object_pose_unity[0], self.init_object_pose_unity[1], self.init_object_pose_unity[2]

        # Note in the dart viewer the ee at the goal is more up as we assume that we have a gripper (urdf) but the gripper is not #
        # yet visualized in the viewer. The self.dart_sim.get_pos_distance() returns 0 correctly at the goal                      #
        tool_length = 0.0 
        target_object_RX, target_object_RY, target_object_RZ = self.get_box_rotation_in_target_dart_coords_angle_axis()
        target = [target_object_RX, target_object_RY, target_object_RZ, target_object_Z, -target_object_X, target_object_Y + tool_length]

        # takes care of resetting the DART chain and should stay as it is
        state = super().reset()

        # sets the initial reaching target for the current episode,
        # should be always called in the beginning of each episode,
        # you might need to call it even during the episode run to change the reaching target for the IK-P controller
        self.set_target(target)

        # initial position for the gripper state, accumulates the tool_action velocity received in update_action
        self.tool_target = 0.0  # should be in range [0.0,90.0]

        # movement control of each joint can be disabled by setting zero for that joint index
        active_joints = [1] * 7

        # the lines below should stay as it is, Unity simulator expects these joint values in radians
        joint_positions = self.dart_sim.chain.getPositions().tolist()
        joint_velocities = self.dart_sim.chain.getVelocities().tolist()

        #######################################################################################
        # Spawn the target rectangle outside of the Unity simulator -> since it is not used   #
        # or can harm the vision-based methods                                                #
        # in dart we map the target to be in the same pose as the pose of the box (see above) #
        #######################################################################################
        #target_positions = self.dart_sim.target.getPositions().tolist()
        target_positions = [0, 0, 0, 0, -5, -200]

        target_X, target_Y, target_Z = -target_positions[4], target_positions[5], target_positions[3]
        target_RX, target_RY, target_RZ = np.rad2deg([-target_positions[1], target_positions[2], target_positions[0]])
        target_positions_mapped = [target_X, target_Y, target_Z, target_RX, target_RY, target_RZ]

        object_positions_mapped = [object_X, object_Y, object_Z, object_RX, object_RY, object_RZ]

        self.reset_state = active_joints + joint_positions + joint_velocities\
                           + target_positions_mapped + object_positions_mapped + [self.tool_target]

        self.collision_flag = False

        #######################################################################################
        # Keep track the previous distance of the ee to the box - used in the reward function #
        # Initialize -during the __init__() they are set to np.inf                            #
        #######################################################################################
        self.prev_dist_ee_box_z = self.get_relative_distance_ee_box_z_unity()
        self.prev_dist_ee_box_x = self.get_relative_distance_ee_box_x_unity()
        self.prev_dist_ee_box_ry = self.get_relative_distance_ee_box_ry_unity() 

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

        _, ee_y, __ = self.get_ee_pos_unity()

        self.target_z_dart = ee_y
        self.target_rot_quat_dart = self.get_rot_ee_quat()

    ################
    # Reward terms #
    ################
    def get_reward_term_displacement(self):
        """
            returns the reward value for the current observation

            uses a displacements logic:
                - the current ee distance to the box in x, z, and ry axis minus the previous ee distance to box for the same axis (see implementation for more)

            :return: reward displacement term (float)
        """
        reward = 0

        # Z axis #
        curr_dist_ee_box_z = self.get_relative_distance_ee_box_z_unity()
        dz = (self.prev_dist_ee_box_z - curr_dist_ee_box_z)              # Displacement

        dz /= self.reward_z_norm_const # Normalize
        dz = np.clip(dz, -1, 1)        # Clip for safety
        dz *= self.reward_z_weight     # Weight this term

        # X axis #
        curr_dist_ee_box_x = self.get_relative_distance_ee_box_x_unity()
        dx = (self.prev_dist_ee_box_x - curr_dist_ee_box_x)

        dx /= self.reward_x_norm_const
        dx = np.clip(dx, -1, 1)
        dx *= self.reward_x_weight

        # RY axis #
        curr_dist_ee_box_ry = self.get_relative_distance_ee_box_ry_unity()
        dry = (self.prev_dist_ee_box_ry - curr_dist_ee_box_ry)

        dry /= self.reward_pose_norm_const
        dry = np.clip(dry, -1, 1)
        dry *= self.reward_pose_weight

        reward = dz + dx + dry

        return reward


    #############
    # Accessors #
    #############
    def get_ee_orient_unity(self):
        """
            Override it for planar grasping envs - easier calculations

            :return: x, y, z orientation of the ee in Euler
        """
        rot_mat = self.dart_sim.chain.getBodyNode('iiwa_link_ee').getTransform().rotation()
        rx, ry, rz = dart.math.matrixToEulerXYZ(rot_mat)

        rx_unity = -ry
        ry_unity = rz
        rz_unity = rx

        return rx_unity, ry_unity, rz_unity

    def get_box_rotation_in_target_dart_coords_angle_axis(self):
        """
            return in angle-axis dart coordinates the orientation of the box
                  - read the unity coordinates of the box and then convert
                  - unity uses degrees [-90, 0] for rotation in our case)
                  - if the orientation of the box is in a different range adapt
 
            :return: a, b, c (in rad) angle-axis dart coordinates the orientation of the box
        """

        object_RY = self.init_object_pose_unity[4]
        object_RY = -object_RY if object_RY >= -45 else -object_RY - 90
        r = R.from_euler('xyz', [-180, 0, -180 + object_RY], degrees=True)
        r = r.as_matrix()
        a, b, c = dart.math.logMap(r)

        return a, b, c

    def get_object_pos_unity(self):
        """
            get the position of the box in unity coords

            :return: x, y, z coords of box in unity
        """
        return self.current_obs[26], self.current_obs[27], self.current_obs[28]

    def get_object_orient_unity(self):
        """
            get the orientation of the box in unity coords
                Important: works for planar grasping envs only - assume the box does not move during the RL episode

            :return: rx, ry, rz coords of box in unity
        """
        return self.init_object_pose_unity[3], self.init_object_pose_unity[4], self.init_object_pose_unity[5]

    def get_object_height_unity(self):
        """
            get the height of the box in unity coords

            :return: y coords of box
        """
        return self.current_obs[27]

    def get_object_pose_unity(self):
        """
            get the pose of the box in unity coords
                Important: works for planar grasping envs only - assume the box does not move during the RL episode

            :return: x, y, z, rx, ry, rz coords of box
        """
        return self.current_obs[26], self.current_obs[27], self.current_obs[28], \
               self.init_object_pose_unity[3], self.init_object_pose_unity[4], self.init_object_pose_unity[5]

    def get_collision_flag(self):
        """
            get the collision flag value

            :return: 0 (no collision) or 1 (collision)
        """
        return self.current_obs[33]

    def get_relative_distance_ee_box_x_unity(self):
        """
            get the relative distance from the box to the ee in x axis in unity coords

            :return: dist (float) of the box to the ee in x axis
        """
        x_err, _, _ = self.get_error_ee_box_pos_unity()
        dist = abs(x_err)

        return dist

    def get_relative_distance_ee_box_z_unity(self):
        """
            get the relative distance from the box to the ee in z axis in unity coords

            :return: dist of the box to the ee in z axis
        """
        _, _, z_err = self.get_error_ee_box_pos_unity()
        dist = abs(z_err)

        return dist

    def get_relative_distance_ee_box_ry_unity(self):
        """
            get the relative distance from the box to the ee in ry axis in unity coords

            :return: dist of the box to the ee in ry axis
        """
        ry_err = self.get_error_ee_box_ry_unity()
        dist = abs(ry_err)

        return dist

    def get_error_ee_box_pos_unity(self):
        """
            get the error from the box to the ee in unity coords

            :return: x_err, y_err, z_err of the box to the ee
        """
        object_x, object_y, object_z = self.get_object_pos_unity()
        ee_x, ee_y, ee_z = self.get_ee_pos_unity()

        return object_x - ee_x, object_y - ee_y, object_z - ee_z

    def get_error_ee_box_x_z_unity(self):
        """
            get the error from the box to the ee in x and z axis in unity coords

            :return: x_err, z_err of the box to the ee
        """
        x_err, _, z_err = self.get_error_ee_box_pos_unity()

        return x_err, z_err

    def get_error_ee_box_x_z_normalized_unity(self):
        """
            get the normalized error from the box to the ee in x and z axis in unity coords
                Important: hard-coded manner - adapt if the ee starts from different initial position, or
                           the boxes are not spawned in front of the robot

            :return: x_err, z_err normalized of the box to the ee
        """
        x_err, z_err = self.get_error_ee_box_x_z_unity()

        return x_err / 1.4, z_err / 0.73

    def get_error_ee_box_ry_unity(self):
        """
            get the relative error from the box to the ee in ry axis in unity coords

            :return: ry_err of the box to the ee - in radians
        """
        _, ee_ry, _ = self.get_ee_orient_unity()

        #####################################################
        # Transorm ee and box ry rotation to our needs      #
        # see the function implementation for more details  #
        #####################################################
        box_ry = np.deg2rad(self.init_object_pose_unity[4]) # Deg -> rad  
        clipped_ee_ry, clipped_box_ry = self.clip_ee_ry_and_box_ry(ee_ry, box_ry)

        ry_error = clipped_box_ry - clipped_ee_ry

        return ry_error

    def get_error_ee_box_ry_normalized_unity(self):
        """
            get the normalized relative error from the box to the ee in ry axis in unity coords
                Important: hard-coded normalization - adapt if needed

            :return: ry_err normalized of the box to the ee
        """
        error_ry = self.get_error_ee_box_ry_unity()

        return error_ry / (2 * np.pi)

    def clip_ee_ry_and_box_ry(self, ee_ry, box_ry):
        """
            Fix the rotation in y axis for both box and end-effector
                - Input for the ee is the raw observation of rotation returned from the Unity simulator
                - Input for the box is the rotation returned from the boxGenerator but transformed in radians
                - The ee starts at +-np.pi rotation in y-axis.
                - Important: some observations are returned with a change of sign and they should be corrected.

            Important: In this clipping behaviour, we assume that when the ee is at +-np.pi and the box
                       at 0 radians, then the error between them is zero. No rotation should be performed - 'highest reward'
                           - Hence, in this case, the function returns for ee_ry -np.pi and for box_ry -np.pi so that their difference is 0.

            Note:      We also define only one correct rotation for grasping the box.
                           - The Box rotation ranges from [-90, 0]
                           - The ee should (only) turn clock-wise when the box is between [-90, -45), and
                             counter-clock-wise when the box is between [-45, 0].

            Warning:   If the ee starts from different rotation than +-np.pi or the box rotation spawn range of [-90, 0] is different - adapt.

            :param ee_ry:  ee rotation returned from the unity simulator in radians
            :param boy_ry: box rotation returned from the boxGenerator, but transformed in radians - does not change during the RL episode

            :return: clipped_ee_ry corrected ee ry rotation in radians
            :return: clipped_box_ry: corrected box ry rotation in radians
        """

        if (self.init_object_pose_unity[4] >= -45): # Counter-clock-wise rotation should be performed
            if (ee_ry > 0): # Change of sign
                ee_ry *= -1
            else:
                # ee is turning in the wrong direction. ee rotation is #
                # decreasing but the error to the box is increasing.   #
                # add the difference to correct the error              #
                ee_ry = -np.pi - (np.pi + ee_ry) 

            # When box_ry is at 0 rad - no rotation of the ee is needed #
            clipped_box_ry = -np.pi - box_ry

        elif (self.init_object_pose_unity[4] < -45): # Clock-wise rotation should be performed
            if (ee_ry < 0):
                ee_ry *= -1
            else:
                ee_ry = np.pi + (np.pi - ee_ry)

            clipped_box_ry = np.pi + (-np.pi / 2 - box_ry)

        clipped_ee_ry = ee_ry

        return clipped_ee_ry, clipped_box_ry

    def clip_ry(self, ee_ry):
        """
            Individual cliping function. For more information refer to the self.clip_ee_ry_and_box_ry() definition
                - This re-definition is for agents that need to clip ee_ry and box_ry seperately 

            :param ee_ry:  ee rotation returned from the unity simulator in radians

            :return: clipped_ee_ry corrected ee rotation (rad)
        """

        if (self.init_object_pose_unity[4] >= -45): 
            if (ee_ry > 0): 
                ee_ry *= -1
            else:
                ee_ry = -np.pi - (np.pi + ee_ry) 

        elif (self.init_object_pose_unity[4] < -45): 
            if (ee_ry < 0):
                ee_ry *= -1
            else:
                ee_ry = np.pi + (np.pi - ee_ry)

        clipped_ee_ry = ee_ry

        return clipped_ee_ry 

    def clip_box_ry(self, box_ry):
        """
            Individual cliping function. For more information refer to the self.clip_ee_ry_and_box_ry() definition
                - This re-definition is for agents that need to clip ee_ry and box_ry seperately 

            :param boy_ry: box rotation returned from the boxGenerator, but transformed in radians - does not change during the RL episode

            :return: clipped_box_ry: corrected box ry rotation in radians
        """

        if (self.init_object_pose_unity[4] >= -45):
            clipped_box_ry = -np.pi - box_ry 
        elif (self.init_object_pose_unity[4] < -45):
            clipped_box_ry = np.pi + (-np.pi / 2 - box_ry) 

        return clipped_box_ry
