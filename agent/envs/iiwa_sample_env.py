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

import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym import spaces
from envs_dart.iiwa_dart_unity import IiwaDartUnityEnv

# Import when images are used as state representation
# import cv2
# import base64

class IiwaSampleEnv(IiwaDartUnityEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns,
                 state_type, enable_render=False, task_monitor=False, 
                 with_objects=False, target_mode="random", target_path="/misc/generated_random_targets/cart_pose_7dof.csv", goal_type="target",
                 joints_safety_limit=0, max_joint_vel=20, max_ee_cart_vel=10.0, max_ee_cart_acc =3.0, max_ee_rot_vel=4.0, max_ee_rot_acc=1.2,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, 0, 0, 0, 0],
                 robotic_tool="3_gripper", env_id=0):

        # range of vertical, horizontal pixels for the DART viewer
        viewport = (0, 0, 500, 500)

        self.state_type = state_type
        self.reset_flag = False
        self.goal_type = goal_type # Target or box 

        # Collision happened for this env when set to 1. In case the manipulator is spanwed in a different position than the #
        # default vertical, for the remaining of the episode zero velocities are sent to UNITY                               #
        self.collided_env = 0

        ################################
        # Set initial joints positions #
        ################################
        self.random_initial_joint_positions = random_initial_joint_positions                                                 # True, False
        self.initial_positions = np.asarray(initial_positions, dtype=float) if initial_positions is not None else None

        # Initial positions flag for the manipulator after reseting. 1 means different than the default vertical position #
        # In that case, the environments should terminate at the same time step due to UNITY synchronization              #
        if((self.initial_positions is None or np.count_nonzero(initial_positions) != 0) and self.random_initial_joint_positions == False):
            self.flag_zero_initial_positions = 0
        else:
            self.flag_zero_initial_positions = 1

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
                         robotic_tool=robotic_tool, enable_render=enable_render, task_monitor=task_monitor,
                         with_objects=with_objects, target_mode=target_mode, target_path=target_path, viewport=viewport,
                         random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions, env_id=env_id)

        # Clip gripper action to this limit #
        if(self.robotic_tool.find("3_gripper") != -1):
            self.gripper_clip = 90
        elif(self.robotic_tool.find("2_gripper") != -1):
            self.gripper_clip = 250
        else:
            self.gripper_clip = 90

        # Some attributes that are initialized in the parent class:
        # self.reset_counter --> keeps track of the number of resets performed
        # self.reset_state   --> reset vector for initializing the Unity simulator at each episode
        # self.tool_target   --> initial position for the gripper state

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

        if self.robotic_tool.find("gripper") != -1:
            self.action_space_dimension += 1  # gripper velocity

        # Variables below exist in the parent class, hence the names should not be changed
        tool_length = 0.2  # [m] allows for some tolerances in maximum observation

        # x,y,z of TCP: maximum reach of arm plus tool length in meters
        ee_pos_high = np.array([0.95 + tool_length, 0.95 + tool_length, 1.31 + tool_length])
        ee_pos_low = -np.array([0.95 + tool_length, 0.95 + tool_length, 0.39 + tool_length])

        high = np.empty(0)
        low = np.empty(0)
        if orientation_control:
            # rx,ry,rz of TCP: maximum orientation in radians without considering dexterous workspace
            ee_rot_high = np.full(3, np.pi)
            # observation space is distance to target orientation (rx,ry,rz), [rad]
            high = np.append(high, ee_rot_high)
            low = np.append(low, -ee_rot_high)

        # and distance to target position (dx,dy,dz), [m]
        high = np.append(high, ee_pos_high - ee_pos_low)
        low = np.append(low, -(ee_pos_high - ee_pos_low))

        # and joint positions [rad] and possibly velocities [rad/s]
        if 'a' in self.state_type:
            high = np.append(high, self.MAX_JOINT_POS)
            low = np.append(low, self.MIN_JOINT_POS)
        if 'v' in self.state_type:
            high = np.append(high, self.MAX_JOINT_VEL)
            low = np.append(low, -self.MAX_JOINT_VEL)

        # the lines below should stay as it is.                                                                        #
        # Important:        Adapt only if you use images as state representation, or your task is more complicated     #
        # Good practice:    If you need to adapt several methods, inherit from IiwaSampleEnv and define your own class # 
        self.action_space = spaces.Box(-np.ones(self.action_space_dimension), np.ones(self.action_space_dimension),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low, high, dtype=np.float32)

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

                if 0.4 < math.sqrt(math.pow(x, 2) + math.pow(y, 2)) < 0.8:
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

        else:
            target = [0, 0, 0, 0, -5, -200] # Dummy will be defined from the user later - advise define it in the reset() function 

        return target

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
            state = np.append(state, self.dart_sim.chain.getPositions())
        if 'v' in self.state_type:
            state = np.append(state, self.dart_sim.chain.getVelocities())

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
        reward = -self.dart_sim.get_pos_distance()

        # stands for reducing orientation error
        if self.dart_sim.orientation_control:
            reward -= 0.5 * self.dart_sim.get_rot_distance()

        # stands for avoiding abrupt changes in actions
        reward -= 0.1 * np.linalg.norm(action - self.prev_action)

        # stands for shaping the reward to increase when target is reached to balance at the target
        if self.get_terminal_reward():
            reward += 1.0 * (np.linalg.norm(np.ones(self.action_space_dimension)) - np.linalg.norm(action))

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
        if self.dart_sim.get_pos_distance() < self.MIN_POS_DISTANCE:
            if not self.dart_sim.orientation_control:
                return True
            if self.dart_sim.get_rot_distance() < self.MIN_ROT_DISTANCE:
                return True

        return False

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
        action = np.clip(action, -1., 1.)

        # This updates the gripper target by accumulating the tool velocity from the action vector   #
        # Note: adapt if needed: e.g. accumulating the tool velocity may not work well for your task #
        if self.robotic_tool.find("gripper") != -1:
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
        if self.robotic_tool.find("gripper") != -1:
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
                                'ee_orientation':     indices [17:20],
                                'target_position':    indices [20:23],
                                'target_orientation': indices [23:26],
                                'object_position':    indices [26:29],
                                'object_orientation': indices [29:32],
                                'gripper_position':   indices [32:33], ---(it is optional, in case a gripper is enabled)
                                'collision_flag':     indices [33:34], ---([32:33] in case of without gripper)

            :param time_step_update: whether to increase the time_step of the agent

            :return: The state, reward, episode termination flag (done), and an info dictionary
        """
        # use commented lines below in case you'd like to save the image observations received from the Unity
        # base64_bytes = observation['ImageData'][0].encode('ascii')
        # image_bytes = base64.b64decode(base64_bytes)
        # image = np.frombuffer(image_bytes, np.uint8)
        # image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        # print(image.shape)
        # cv2.imwrite("test.jpg", image)

        self.current_obs = observation['Observation']

        ###########################################################################################
        # collision happened or joints limits overpassed                                          #
        # Important: in case the manipulator is spanwed in a different position than the default  #
        #            vertical, for the remaining of the episode zero velocities are sent to UNITY #
        #            see _send_actions_and_update() method in simulator_vec_env.py                #
        ###########################################################################################
        if(observation['Observation'][-1] == 1.0 or self.joints_limits_violation()):
            self.collided_env = 1
            # Reset when we have a collision only when we spawn the robot to the default #
            # vertical position, else wait the episode to finish                         #
            # Important: you may want to reset anyway depending on your task - adapt     #
            if(self.flag_zero_initial_positions == 0): 
                self.reset_flag = True 

        # the method below handles synchronizing states of the DART kinematic chain with the #
        # observation from Unity hence it should be always called                            #
        self._unity_retrieve_joint_values(observation['Observation'])

        # Class attributes below exist in the parent class, hence the names should not be changed
        if(time_step_update == True):
            self.time_step += 1

        state = self.get_state()
        reward = self.get_reward(self.action_state)
        done = bool(self.get_terminal())
        info = {"success": False}                   # Episode was successful. For now it is set at simulator_vec_env.py before reseting, step() method. Adapt if needed

        self.prev_action = self.action_state

        return state, reward, done, info

    def reset(self, temp=False):
        """
            resets the DART-gym environment and creates the reset_state vector to be sent to Unity

            :param temp: not relevant

            :return: The initialized state
        """
        # takes care of resetting the DART chain and should stay as it is
        state = super().reset(random_initial_joint_positions=self.random_initial_joint_positions, initial_positions=self.initial_positions)

        self.collided_env = 0
        random_target = self.create_target() # draw a red target

        # sets the initial reaching target for the current episode,
        # should be always called in the beginning of each episode,
        # you might need to call it even during the episode run to change the reaching target for the IK-P controller
        self.set_target(random_target)

        # initial position for the gripper state, accumulates the tool_action velocity received in update_action
        self.tool_target = 0.0  # should be in range [0.0,90.0]

        # movement control of each joint can be disabled by setting zero for that joint index
        active_joints = [1] * 7

        # the lines below should stay as it is, Unity simulator expects these joint values in radians
        joint_positions = self.dart_sim.chain.getPositions().tolist()
        joint_velocities = self.dart_sim.chain.getVelocities().tolist()

        # the mapping below for the target should stay as it is, unity expects orientations in degrees
        target_positions = self.dart_sim.target.getPositions().tolist()

        target_X, target_Y, target_Z = -target_positions[4], target_positions[5], target_positions[3]
        target_RX, target_RY, target_RZ = np.rad2deg([-target_positions[1], target_positions[2], target_positions[0]])
        target_positions_mapped = [target_X, target_Y, target_Z, target_RX, target_RY, target_RZ]

        # spawn the object in UNITY: by default a green box is spawned     
        # depending on your task, positioning the object might be necessary, start from the following sample code
        # important: unity expects orientations in degrees
        object_X, object_Y, object_Z = np.random.uniform(-1.0, 1.0), 0.05, np.random.uniform(-1.0, 1.0)
        object_RX, object_RY, object_RZ = 0.0, 0.0, 0.0
        object_positions_mapped = [object_X, object_Y, object_Z, object_RX, object_RY, object_RZ]

        self.reset_state = active_joints + joint_positions + joint_velocities\
                           + target_positions_mapped + object_positions_mapped
 
        if self.robotic_tool.find("gripper") != -1:
            self.reset_state = self.reset_state + [self.tool_target]

        self.reset_counter += 1
        self.reset_flag = False

        return state

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

        if self.robotic_tool.find("gripper") != -1:
            tool_vel = 0.0                         # zero velocity means no gripper movement - should be adapted for the task
            action = np.append(action, [tool_vel])

        return action
