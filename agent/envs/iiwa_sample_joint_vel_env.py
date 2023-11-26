"""
A sample Env class inheriting from basic gym.Env for the Kuka LBR iiwa manipulator with 7 links and a Gripper.

Important: There is no inverse kinematics calculation supported here (see iiwa_sample_env for IK), only joint velocity control is supported with it.

Unity is used as the main simulator for physics/rendering computations. The Unity interface receives joint velocities as commands
and returns joint positions and velocities. The class presents a way to alternate between numeric observations and image observations,
how to parse images returned from the Unity simulator and how to reset the environments in the simulator
"""

import math
import time

import cv2
import base64

import numpy as np
from numpy import newaxis

from gym import spaces, core
from gym.utils import seeding

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

        self.collided_env = 0
        self.seed()

        # number of joints to be controlled from the 7 available
        self.num_joints = config["num_joints"]

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
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # the action space is in the range [-1, 1] for the controllable joints and a 0 for the gripper opening as we
        # do not need to control the gripper for the reaching task
        temp = [1] * self.num_joints
        temp.append(0)
        high = np.array(temp)
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)

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

        # create agent state for the specific task from the generic simulator observation that was passed
        self._convert_observation(observation['Observation'])

        info = {"success": False} # Episode was successful. For now it is set at simulator_vec_env.py before the reset. Adapt if needed
        terminal = self._get_terminal()
        reward = self._get_reward()

        if self.use_images:
            return self._retrieve_image(observation), reward, terminal, info
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
        if self._get_distance() < self.MIN_POS_DISTANCE:
            return True

        return False

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
                - x, y, and z for t and o are the initial coordinates of the target and the object in meters (note that y is
                  vertical axis in unity)
                - rx, ry, and rz for t and o are the euler angles in degrees for the rotation of the object and the target
                - g_p is the position (opening) of the gripper (0 is open, value up to 90 supported)
        """
        self.reset_counter += 1

        self.ts = 0
        self.collided_env = 0

        self.target_x = 0
        self.target_y = 0.38  # np.random.uniform(0, 1.15)
        self.target_z = -0.9

        self.object_x = 1  # np.random.uniform(-1., 1.)
        self.object_y = 1  # 0.05
        self.object_z = 1  # np.random.uniform(-1., 1.)

        # the joints to be enabled have value 1 and the others have value 0
        self.reset_state = [1] * self.num_joints + [0] * (7 - self.num_joints)

        # zeros for initial joint angles and velocities 
        # Note: adapt this to reset to different initial positions if needed 
        self.reset_state.extend([0] * 14)

        # initial target and object position and orientation and gripper opening
        self.reset_state.extend([
            self.target_x, self.target_y, self.target_z, 90, 0, 0,
            self.object_x, self.object_y, self.object_z, 90, 0, 0,
            0])

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
