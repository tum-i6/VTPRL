"""
A sample Env class inheriting from the DART-Unity Env for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment
Unity is used as the main simulator for physics/rendering computations.
The Unity interface receives joint velocities as commands and returns joint positions and velocities

DART is used to calculate inverse kinematics of the iiwa chain.
DART changes the agent action space from the joint space to the cartesian space
(position-only or pose/SE(3)) of the end-effector.

action_by_pd_control method can be called to implement a Proportional-Derivative control law instead of an RL policy.

Note: Coordinates in the Unity simulator are different from the ones in DART which used here:
      The mapping is [X, Y, Z] of Unity is [-y, z, x] of DART
"""

import numpy as np
import os

from gym import spaces
from envs_dart.iiwa_dart_unity import IiwaDartUnityEnv

# Import when images are used as state representation
# import cv2
# import base64

class IiwaSampleEnv(IiwaDartUnityEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns,
                 state_type, enable_render=False, task_monitor=False, 
                 with_objects=False, target_mode="random", target_path="/misc/generated_random_targets/cart_pose_7dof.csv", goal_type="target",
                 joints_safety_limit=0.0, max_joint_vel=20.0, max_ee_cart_vel=10.0, max_ee_cart_acc =3.0, max_ee_rot_vel=4.0, max_ee_rot_acc=1.2,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, 0, 0, 0, 0],
                 robotic_tool="3_gripper", env_id=0):

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
        # and torques of the iiwa kinematic chain                          #
        self.MAX_JOINT_POS = np.deg2rad([170, 120, 170, 120, 170, 120, 175]) - self.JOINT_POS_SAFE_LIMIT  # [rad]: based on the specs
        self.MIN_JOINT_POS = -self.MAX_JOINT_POS

        # Joint space #
        self.MAX_JOINT_VEL = np.deg2rad(np.full(7, max_joint_vel))                                        # np.deg2rad([85, 85, 100, 75, 130, 135, 135])  # [rad/s]: based on the specs
        self.MAX_JOINT_ACC = 3.0 * self.MAX_JOINT_VEL                                                     # [rad/s^2]: just approximation due to no existing data
        self.MAX_JOINT_TORQUE = np.array([320, 320, 176, 176, 110, 40, 40])                               # [Nm]: based on the specs

        # admissible range for Cartesian pose translational and rotational velocities, #
        # and accelerations of the end-effector                                        #
        self.MAX_EE_CART_VEL = np.full(3, max_ee_cart_vel)                                                # np.full(3, 10.0) # [m/s] --- not optimized values for sim2real transfer
        self.MAX_EE_CART_ACC = np.full(3, max_ee_cart_acc)                                                # np.full(3, 3.0) # [m/s^2] --- not optimized values
        self.MAX_EE_ROT_VEL = np.full(3, max_ee_rot_vel)                                                  # np.full(3, 4.0) # [rad/s] --- not optimized values
        self.MAX_EE_ROT_ACC = np.full(3, max_ee_rot_acc)                                                  # np.full(3, 1.2) # [rad/s^2] --- not optimized values

        ##################################################################################
        # End set limits -> Important: Must be set before calling the super().__init__() #
        ##################################################################################

        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns,
                         state_type=state_type, robotic_tool=robotic_tool, enable_render=enable_render, task_monitor=task_monitor,
                         with_objects=with_objects, target_mode=target_mode, target_path=target_path, viewport=viewport,
                         random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions, env_id=env_id)

        # Collision happened for this env when set to 1. In case the manipulator is spanwed in a different position than the #
        # default vertical, for the remaining of the episode zero velocities are sent to UNITY                               #
        self.collided_env = 0
        self.reset_flag = False

        ################################
        # Set initial joints positions #
        ################################
        self.random_initial_joint_positions = random_initial_joint_positions                                                 # True, False
        self.initial_positions = np.asarray(initial_positions, dtype=float) if initial_positions is not None else None

        # Initial positions flag for the manipulator after reseting. 1 means different than the default vertical position #
        # In that case, the environments should terminate at the same time step due to UNITY synchronization              #
        if((self.initial_positions is None or np.count_nonzero(initial_positions) == 0) and self.random_initial_joint_positions == False):
            self.flag_zero_initial_positions = 0
        else:
            self.flag_zero_initial_positions = 1

        # Clip gripper action to this limit #
        if(self.robotic_tool.find("3_gripper") != -1):
            self.gripper_clip = 90
        elif(self.robotic_tool.find("2_gripper") != -1):
            self.gripper_clip = 250
        else:
            self.gripper_clip = 90

        self.transform_ee_initial = None
        self.save_image = False  # change it to True to save images received from Unity into jpg files

        if self.save_image:
            self.save_image_folder = self.get_save_image_folder()

        # Some attributes that are initialized in the parent class:
        # self.reset_counter --> keeps track of the number of resets performed
        # self.reset_state   --> reset vector for initializing the Unity simulator at each episode
        # self.tool_target   --> initial position for the gripper state

        # helper methods that can be called from the parent class:
        # self._convert_vector_dart_to_unity(vector) -- transforms a [x, y, z] position vector
        # self._convert_vector_unity_to_dart(vector)
        # self._convert_rotation_dart_to_unity(matrix) -- transforms a 3*3 rotation matrix
        # self._convert_rotation_unity_to_dart(matrix)
        # self._convert_quaternion_dart_to_unity(quaternion) -- transforms a [w, x, y, z] quaternion vector
        # self._convert_quaternion_unity_to_dart(quaternion)
        # self._convert_angle_axis_dart_to_unity(vector) -- transforms a [rx, ry, rz] logmap representation of an Angle-Axis
        # self._convert_angle_axis_unity_to_dart(vector)
        # self._convert_pose_dart_to_unity(dart_pose, unity_in_deg=True) -- transforms a pose -- DART pose order [rx, ry, rz, x, y, z] -- Unity pose order [X, Y, Z, RX, RY, RZ]
        # self.get_rot_error_from_quaternions(target_quat, current_quat) -- calculate the rotation error in angle-axis given orientation inputs in quaternion format
        # self.get_rot_ee_quat()  -- DART coords -- get the orientation of the ee in quaternion
        # self.get_ee_pos()       -- DART coords -- get the position of the ee
        # self.get_ee_orient()    -- DART coords -- get the orientation of the ee in angle-axis
        # self.get_ee_pose()      -- DART coords -- get the pose of the ee
        # self.get_ee_pos_unity()           -- get the ee position in unity coords
        # self.get_ee_orient_unity()        -- get the ee orientation in angle-axis format in unity coords
        # self.get_ee_orient_euler_unity()  -- get the ee orientation in XYZ euler angles in unity coords

        # the lines below should stay as it is
        self.MAX_EE_VEL = self.MAX_EE_CART_VEL
        self.MAX_EE_ACC = self.MAX_EE_CART_ACC
        if orientation_control:
            self.MAX_EE_VEL = np.concatenate((self.MAX_EE_ROT_VEL, self.MAX_EE_VEL))
            self.MAX_EE_ACC = np.concatenate((self.MAX_EE_ROT_ACC, self.MAX_EE_ACC))

        # the lines below wrt action_space_dimension should stay as it is
        self.action_space_dimension = self.n_links  # there would be 7 actions in case of joint-level control
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

        # the lines below should stay as it is.                                                                        #
        # Important:        Adapt only if you use images as state representation, or your task is more complicated     #
        # Good practice:    If you need to adapt several methods, inherit from IiwaSampleEnv and define your own class #
        self.action_space = spaces.Box(-np.ones(self.action_space_dimension, dtype=np.float32), np.ones(self.action_space_dimension, dtype=np.float32), dtype=np.float32)

        self.observation_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

    def _update_env_flags(self):
        ###########################################################################################
        # collision happened or joints limits overpassed                                          #
        # Important: in case the manipulator is spanwed in a different position than the default  #
        #            vertical, for the remaining of the episode zero velocities are sent to UNITY #
        #            see _send_actions_and_update() method in simulator_vec_env.py                #
        ###########################################################################################
        if self.unity_observation['collision_flag'] == 1.0 or self.joints_limits_violation():
            self.collided_env = 1
            # Reset when we have a collision only when we spawn the robot to the default #
            # vertical position, else wait the episode to finish                         #
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

            _random_target_gen_joint_level(): generate a random sample target in the reachable workspace of the iiwa

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

    def get_reward(self, action):
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
        if self.prev_pos_distance is not None:
            reward += self.prev_pos_distance - curr_pos_distance
        self.prev_pos_distance = curr_pos_distance

        if self.dart_sim.orientation_control:
            curr_rot_distance = self.dart_sim.get_rot_distance()
            if self.prev_rot_distance is not None:
                reward += 0.3 * (self.prev_rot_distance - curr_rot_distance)
            self.prev_rot_distance = curr_rot_distance

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
                                                   to velocities in the joint space of the kinematic chain (j1,...,j7)

           :param action: The action vector decided by the RL agent, acceptable range: [-1,+1]

                          It should be a numpy array with the following shape: [arm_action] or [arm_action, tool_action]
                          in case 'robotic_tool' is a gripper, tool_action is always a dim-1 scalar value representative of the normalized gripper velocity

                          arm_action has different dim based on each case of control level:
                              in case of use_ik=False    -> is dim-7 representative of normalized joint velocities
                              in case of use_ik=True     -> there would be two cases:
                                  orientation_control=False  -> of dim-3: Normalized EE Cartesian velocity in x,y,z DART coord
                                  orientation_control=True   -> of dim-6: Normalized EE Rotational velocity in x,y,z DART coord followed by Normalized EE Cartesian velocity in x,y,z DART coord

           :return: the command to send to the Unity simulator including joint velocities and possibly the gripper position
        """

        # the lines below should stay as it is
        self.action_state = action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # This updates the gripper target by accumulating the tool velocity from the action vector   #
        # Note: adapt if needed: e.g. accumulating the tool velocity may not work well for your task #
        if self.with_gripper:
            tool_action = action[-1]
            self.tool_target = np.clip((self.tool_target + tool_action), 0.0, self.gripper_clip)

            # This removes the gripper velocity from the action vector for inverse kinematics calculation
            action = action[:-1]

        # use INVERSE KINEMATICS #
        if self.dart_sim.use_ik:
            task_vel = self.MAX_EE_VEL * action
            joint_vel = self.dart_sim.command_from_action(task_vel, normalize_action=False)
        else:
            joint_vel = self.MAX_JOINT_VEL * action

        # append tool action #
        unity_action = joint_vel
        if self.with_gripper:
            unity_action = np.append(unity_action, [float(self.tool_target)])

        return unity_action

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

            :param time_step_update: whether to increase the time_step of the agent

            :return: The state, reward, episode termination flag (done), and an info dictionary
        """
        self.current_obs = observation['Observation']

        if self.save_image:
            self._unity_retrieve_observation_image(observation['ImageData'])

        # the methods below handles synchronizing states of the DART kinematic chain with the observation from Unity
        # hence it should be always called
        self._unity_retrieve_observation_numeric(observation['Observation'])
        self._update_dart_chain()
        self._update_env_flags()

        # Class attributes below exist in the parent class, hence the names should not be changed
        if(time_step_update == True):
            self.time_step += 1

        self._state = self.get_state()
        self._reward = self.get_reward(self.action_state)
        self._done = bool(self.get_terminal())
        self._info = {"success": False}                   # Episode was successful. For now it is set at simulator_vec_env.py before reseting, step() method. Adapt if needed

        self.prev_action = self.action_state

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
        random_target = self.create_target() # draw a red target

        # sets the initial reaching target for the current episode,
        # should be always called in the beginning of each episode,
        # you might need to call it even during the episode run to change the reaching target for the IK-P controller
        self.set_target(random_target)

        # movement control of each joint can be disabled by setting zero for that joint index
        active_joints = [1] * 7

        # the lines below should stay as it is, Unity simulator expects these joint values in radians
        joint_positions = self.dart_sim.chain.getPositions().tolist()
        joint_velocities = self.dart_sim.chain.getVelocities().tolist()

        # the mapping below for the target should stay as it is, unity expects orientations in degrees
        target_positions = self.dart_sim.target.getPositions().tolist()
        target_positions_mapped = self._convert_pose_dart_to_unity(target_positions)

        # spawn the object in UNITY: by default a green box is spawned
        # mapping from DART to Unity coordinates -- Unity expects orientations in degrees
        object_positions = self.generate_object()
        object_positions_mapped = self._convert_pose_dart_to_unity(object_positions)
  
        self.reset_state = active_joints + joint_positions + joint_velocities\
                           + target_positions_mapped + object_positions_mapped
 
        if self.with_gripper:
            # initial position for the gripper state, accumulates the tool_action velocity received in update_action
            self.tool_target = 0.0  # should be in range [0.0,90.0] for 3-gripper or [0.0,250.0] for 2-gripper
            self.reset_state = self.reset_state + [self.tool_target]

        self.reset_counter += 1
        self.reset_flag = False

        return self._state

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

        action_lin = coeff_kp_lin * self.dart_sim.get_pos_error()
        action = action_lin

        if self.dart_sim.orientation_control:
            action_rot = coeff_kp_rot * self.dart_sim.get_rot_error()
            action = np.concatenate(([action_rot, action]))

        if self.with_gripper:
            tool_vel = 0.0                         # zero velocity means no gripper movement - should be adapted for the task
            action = np.append(action, [tool_vel])

        return action
