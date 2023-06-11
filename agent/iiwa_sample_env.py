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
DART uses Eigen library - Geometry module to retrieve transformation matrices of chains of rigidbodies
Documentation: https://eigen.tuxfamily.org/dox/group__Geometry__Module.html
Background details: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation
Further code details: https://github.com/dartsim/dart/blob/main/dart/math/Geometry.cpp
"""

import numpy as np
import dartpy as dart
# import cv2
# import base64
from gym import spaces
from dart_envs.iiwa_dart_unity import IiwaDartUnityEnv


class IiwaSampleEnv(IiwaDartUnityEnv):

    # Variables below exist in the parent class, hence the names should not be changed

    # Min distance to declare that the target is reached by the end-effector, adapt the values based on your task
    MIN_POS_DISTANCE = 0.05  # [m]
    MIN_ROT_DISTANCE = 0.1  # [rad]

    # admissible range for joint positions, velocities, accelerations and torques of the iiwa kinematic chain
    MAX_JOINT_POS = np.deg2rad([170, 120, 170, 120, 170, 120, 175])  # [rad]: based on the specs
    MIN_JOINT_POS = -MAX_JOINT_POS
    MAX_JOINT_VEL = np.deg2rad([85, 85, 100, 75, 130, 135, 135])  # [rad/s]: based on the specs
    MAX_JOINT_ACC = 3.0 * MAX_JOINT_VEL  # [rad/s^2]: just approximation due to no existing data
    MAX_JOINT_TORQUE = np.array([320, 320, 176, 176, 110, 40, 40])  # [Nm]: based on the specs

    # admissible range for Cartesian pose translational and rotational velocities and accelerations of the end-effector
    MAX_EE_CART_VEL = np.full(3, 10.0)  # [m/s] --- not optimized values
    MAX_EE_CART_ACC = np.full(3, 3.0)  # [m/s^2] --- not optimized values
    MAX_EE_ROT_VEL = np.full(3, 4.0)  # [rad/s] --- not optimized values
    MAX_EE_ROT_ACC = np.full(3, 1.2)  # [rad/s^2] --- not optimized values

    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns,
                 enable_render, state_type, with_gripper=True, env_id=0):

        # range of vertical, horizontal pixels for the DART viewer
        viewport = (0, 0, 500, 500)

        self.state_type = state_type
        self.reset_flag = False
        self.with_gripper = with_gripper

        # the init of the parent class should be always called, this will in the end call reset() once
        super().__init__(max_ts=max_ts, orientation_control=orientation_control,
                         use_ik=use_ik, ik_by_sns=ik_by_sns, enable_render=enable_render,
                         viewport=viewport, env_id=env_id)

        # Some attributes that are initialized in the parent class:
        # self.reset_counter --> keeps track of the number of resets performed
        # self.reset_state --> reset vector for initializing the Unity simulator at each episode
        # self.tool_target --> initial position for the gripper state

        # the lines below should stay as it is
        self.MAX_EE_VEL = self.MAX_EE_CART_VEL
        self.MAX_EE_ACC = self.MAX_EE_CART_ACC
        if self.ORIENTATION_CONTROL:
            self.MAX_EE_VEL = np.concatenate((self.MAX_EE_ROT_VEL, self.MAX_EE_VEL))
            self.MAX_EE_ACC = np.concatenate((self.MAX_EE_ROT_ACC, self.MAX_EE_ACC))

        # the lines below wrt action_space_dimension should stay as it is
        self.action_space_dimension = self.n_links  # there would be 7 actions in case of joint-level control
        if self.USE_IK:
            # There are three cartesian coordinates x,y,z for inverse kinematic control
            self.action_space_dimension = 3
            if self.ORIENTATION_CONTROL:
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
        if self.ORIENTATION_CONTROL:
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

        # the lines below should stay as it is
        self.action_space = spaces.Box(-np.ones(self.action_space_dimension), np.ones(self.action_space_dimension),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.reset()

    def create_target(self):
        """
            defines the target to reach per episode, this should be adapted by the task
            should always return rx,ry,rz,x,y,z in order,
            i.e., first target orientation rx,ry,rz in radians, and then target position x,y,z in meters
            in case of orientation_control=False --> rx,ry,rz are irrelevant and can be set to zero
            _random_target_gen_joint_level(): generate a random sample target in the reachable workspace of the iiwa

            :return: Cartesian pose of the target to reach in the task space
        """
        # a sample fixed target on the right side of iiwa with orientation pointing downwards
        # rx, ry, rz = 0.0, np.pi, 0.0
        # x, y, z = 0.0, 0.57, 0.655
        # target = rx, ry, rz, x, y, z

        # a sample reachable target generator
        target = self._random_target_gen_joint_level()

        return target

    def get_pos_error(self):
        """
           calculates the Euclidean error from the end-effector to the target position
           body_node.getTransform().translation() --> Cartesian position (x,y,z) of the center of that body_node

           :return: Euclidean error from the end-effector to the target position
        """
        ee = self.dart_chain.getBodyNode('iiwa_link_ee')  # The end-effector rigid-body node
        target = self.dart_target.getBodyNode(0)  # The target rigid-body node
        position_error = target.getTransform().translation() - ee.getTransform().translation()
        return position_error

    def get_pos_distance(self):
        """
           :return: L2-norm of the Euclidean error from the end-effector to the target position
        """
        distance = np.linalg.norm(self.get_pos_error())
        return distance

    def get_rot_error(self):
        """
           calculates the Quaternion error from the end-effector to the target orientation
           body_node.getTransform().rotation() --> Orientation of the body_node in a 3*3 rotation matrix
           body_node.getTransform().quaternion() --> Orientation of the body_node in a unit quaternion form (w,x,y,z)
           dart.math.logMap() --> calculates an angle-axis representation of a rotation matrix

           :return: returns Quaternion error from the end-effector to the target orientation
        """
        ee = self.dart_chain.getBodyNode('iiwa_link_ee')  # The end-effector rigid-body node
        target = self.dart_target.getBodyNode(0)  # The target rigid-body node
        quaternion_error = target.getTransform().quaternion().multiply(ee.getTransform().quaternion().inverse())
        orientation_error = dart.math.logMap(quaternion_error.rotation())
        return orientation_error

    def get_rot_distance(self):
        """
           :return: L2-norm of the Quaternion error from the end-effector to the target orientation
        """
        distance = np.linalg.norm(self.get_rot_error())
        return distance

    def get_state(self):
        """
           defines the environment state, this should be adapted by the task
           get_pos_error(): returns Euclidean error from the end-effector to the target position
           get_rot_error(): returns Quaternion error from the end-effector to the target orientation

           :return: state for the policy training
        """
        state = np.empty(0)
        if self.ORIENTATION_CONTROL:
            state = np.append(state, self.get_rot_error())
        state = np.append(state, self.get_pos_error())
        if 'a' in self.state_type:
            state = np.append(state, self.dart_chain.getPositions())
        if 'v' in self.state_type:
            state = np.append(state, self.dart_chain.getVelocities())

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
        reward = -self.get_pos_distance()

        # stands for reducing orientation error
        if self.ORIENTATION_CONTROL:
            reward -= 0.5 * self.get_rot_distance()

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

           :return: a boolean value representing if the target is reached within the defined threshold
        """
        if self.get_pos_distance() < self.MIN_POS_DISTANCE:
            if not self.ORIENTATION_CONTROL:
                return True
            if self.get_rot_distance() < self.MIN_ROT_DISTANCE:
                return True

        return False

    def get_terminal(self):
        """
           checks the terminal conditions for the episode

           :return: a boolean value indicating if the episode should be terminated
        """

        if self.time_step > self.max_ts:
            self.reset_flag = True

        return self.reset_flag

    def update_action(self, action):
        """
           converts env action to the required unity action by possibly using inverse kinematics from DART
           _dart_calc_inv_kinematics() --> changes the action from velocities in the task space (position-only 'x,y,z'
           or complete pose 'rx,ry,rz,x,y,z') to velocities in the joint space of the kinematic chain (j1,...,j7)

           :param action: The action vector decided by the RL agent, acceptable range: [-1,+1]
                          It should be a numpy array with the following shape: [arm_action] or [arm_action, tool_action]
                          in case of with_gripper=True, tool_action is always a dim-1 scalar value representative of the normalized gripper velocity
                          arm_action has different dim based on each case of control level:
                          in case of use_ik=False -> is dim-7 representative of normalized joint velocities
                          in case of use_ik=True -> there would be two cases:
                          orientation_control=False -> of dim-3: Normalized EE Cartesian velocity in x,y,z DART coord
                          orientation_control=True -> of dim-6: Normalized EE Rotational velocity in x,y,z DART coord
                          followed by Normalized EE Cartesian velocity in x,y,z DART coord

           :return: the command to send to the Unity simulator including joint velocities and possibly gripper position
        """

        # the lines below should stay as it is
        self.action_state = action
        action = np.clip(action, -1., 1.)

        if self.with_gripper:
            # This updates the gripper target by accumulating the tool velocity from the action vector
            tool_action = action[-1]
            self.tool_target = np.clip((self.tool_target + tool_action), 0.0, 90.0)
            # This removes the gripper velocity from the action vector for inverse kinematics calculation
            action = action[:-1]

        if self.USE_IK:
            task_vel = self.MAX_EE_VEL * action
            joint_vel = self._dart_calc_inv_kinematics(task_vel)
        else:
            joint_vel = self.MAX_JOINT_VEL * action

        unity_action = joint_vel
        if self.with_gripper:
            unity_action = np.append(unity_action, [float(self.tool_target)])

        return unity_action

    def update(self, observation):
        """
           converts the unity observation to the required env state defined in get_state()
           with also computing the reward value from the get_reward(...) and done flag,
           it increments the time_step, and outputs done=True when the environment should be reset

           :param observation: is the observation received from the Unity simulator within its [X,Y,Z] coordinate system
                'joint_values': indices [0:7],
                'joint_velocities': indices [7:14],
                'ee_position': indices [14:17],
                'ee_orientation': indices [17:20],
                'target_position': indices [20:23],
                'target_orientation': indices [23:26],
                'object_position': indices [26:29],
                'object_orientation': indices [29:32],
                'gripper_position': indices [32:33], ---(it is optional, in case a gripper is enabled)
                'collision_flag': indices [33:34], ---([32:33] in case of without gripper)

           :return: The state, reward, episode termination flag (done), and an empty info dictionary
        """

        # use commented lines below in case you'd like to save the image observations received from the Unity
        # base64_bytes = observation['ImageData'][0].encode('ascii')
        # image_bytes = base64.b64decode(base64_bytes)
        # image = np.frombuffer(image_bytes, np.uint8)
        # image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        # print(image.shape)
        # cv2.imwrite("test.jpg", image)

        if observation['Observation'][-1] == 1.0:
            self.reset_flag = True

        # the method below handles synchronizing states of the DART kinematic chain with the observation from Unity
        # hence it should be always called
        self._unity_retrieve_joint_values(observation['Observation'])

        # Class attributes below exist in the parent class, hence the names should not be changed
        self.time_step += 1

        state = self.get_state()
        reward = self.get_reward(self.action_state)
        done = bool(self.get_terminal())
        info = {}

        self.prev_action = self.action_state

        return state, reward, done, info

    def reset(self, temp=False):
        """
            resets the DART-gym environment and creates the reset_state vector to be sent to Unity

            :param temp: not relevant

            :return: The initialized state
        """

        # takes care of resetting the DART chain and should stay as it is
        state = super().reset()

        # creates a random target for reaching task, should be adapted for your specific task
        random_target = self.create_target()

        # sets the initial reaching target for the current episode,
        # should be always called in the beginning of each episode,
        # you might need to call it even during the episode run to change the reaching target for the IK-P controller
        self.set_target(random_target)

        # initial position for the gripper state, accumulates the tool_action velocity received in update_action
        self.tool_target = 0.0  # should be in range [0.0,90.0]

        # movement control of each joint can be disabled by setting zero for that joint index
        active_joints = [1] * 7

        # the lines below should stay as it is, Unity simulator expects these joint values in radians
        joint_positions = self.dart_chain.getPositions().tolist()
        joint_velocities = self.dart_chain.getVelocities().tolist()

        # the mapping below for the target should stay as it is, unity expects orientations in degrees
        target_positions = self.dart_target.getPositions().tolist()
        target_X, target_Y, target_Z = -target_positions[4], target_positions[5], target_positions[3]
        target_RX, target_RY, target_RZ = np.rad2deg([-target_positions[1], target_positions[2], target_positions[0]])
        target_positions_mapped = [target_X, target_Y, target_Z, target_RX, target_RY, target_RZ]

        # depending on your task, positioning the object might be necessary,
        # start from the following sample code, unity expects orientations in degrees
        object_X, object_Y, object_Z = np.random.uniform(-1.0, 1.0), 0.05, np.random.uniform(-1.0, 1.0)
        object_RX, object_RY, object_RZ = 0.0, 0.0, 0.0
        object_positions_mapped = [object_X, object_Y, object_Z, object_RX, object_RY, object_RZ]

        self.reset_state = active_joints + joint_positions + joint_velocities\
                           + target_positions_mapped + object_positions_mapped
        
        if self.with_gripper:
            self.reset_state = self.reset_state + [self.tool_target]

        self.reset_counter += 1
        self.reset_flag = False

        return state

    def action_by_p_control(self, coeff_kp_lin, coeff_kp_rot):
        """
            computes the task-space velocity commands proportional to the reaching target error

            :param coeff_kp_lin: proportional coefficient for the translational error
            :param coeff_kp_rot: proportional coefficient for the rotational error

            :return: The action in task space
        """

        action_lin = coeff_kp_lin * self.get_pos_error()
        action = action_lin

        if self.ORIENTATION_CONTROL:
            action_rot = coeff_kp_rot * self.get_rot_error()
            action = np.concatenate(([action_rot, action]))

        if self.with_gripper:
            tool_vel = 0.0  # zero velocity means no gripper movement - should be adapted for the task
            action = np.append(action, [tool_vel])

        return action
