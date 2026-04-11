"""
A vision-based end-to-end RL planar grasping Env class inheriting from the IiwaNumericalPlanarGraspingEnv for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment

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
import cv2
import base64

class IiwaEndToEndPlanarGraspingEnv(IiwaNumericalPlanarGraspingEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns, state_type, use_images=True, enable_render=False, target_mode="None", goal_type="box",
                 randomBoxesGenerator=None, joints_safety_limit=10, max_joint_vel=20, max_ee_cart_vel=0.035, max_ee_cart_acc =10, max_ee_rot_vel=0.15, max_ee_rot_acc=10,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2],
                 noise_enable_rl_obs=False, noise_rl_obs_ratio=0.05, reward_dict=None, agent_kp=0.5, agent_kpr=1.5,
                 robotic_tool=None, end_effector_model=None, manipulator_config=None, manipulator_gym_config=None, image_size=128, env_id=0):

        if(use_images != True):
            raise Exception("End-to-end vision-based env requires use_images to be set to True - abort")

        # the init of the parent class should be always called, this will in the end call reset() once
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

        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns, state_type=state_type, enable_render=enable_render, target_mode=target_mode, goal_type=goal_type, 
                        randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=joints_safety_limit, max_joint_vel=max_joint_vel, max_ee_cart_vel=max_ee_cart_vel,
                        max_ee_cart_acc=max_ee_cart_acc, max_ee_rot_vel=max_ee_rot_vel, max_ee_rot_acc=max_ee_rot_acc,
                        random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions,noise_enable_rl_obs=False,noise_rl_obs_ratio=0.05,
                        reward_dict=reward_dict, agent_kp=agent_kp, agent_kpr=agent_kpr, robotic_tool=robotic_tool, manipulator_config=manipulator_config, manipulator_gym_config=manipulator_gym_config, env_id=env_id)

        self.use_images = use_images
        self.image_size = image_size
        self.current_obs_img = None

        ###################################
        # RGB image                       #
        # Unormalized observation for now #
        # Adapt to your project           #
        ###################################
        if self.robot_count > 1:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.robot_count, self.image_size, self.image_size, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8)


    def get_state(self):
        """
            End-to-end planar grasping agent receives a whole image as state input

           :return: state for the policy training
        """
        return self.current_obs_img

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

        per_robot_images = []
        per_robot_rewards = []
        per_robot_success = []

        for ridx in range(self.robot_count):
            robot_payload = robots_by_index.get(ridx)
            if robot_payload is None and ridx < len(robots):
                candidate = robots[ridx]
                if isinstance(candidate, dict):
                    robot_payload = candidate
            if robot_payload is None:
                raise ValueError(f"Missing robot payload for robot index {ridx}.")

            self.current_obs = robot_payload.get('Numeric', {}) if isinstance(robot_payload, dict) else {}
            if isinstance(self.init_object_pose_per_robot, list) and ridx < len(self.init_object_pose_per_robot):
                self.init_object_pose = self.init_object_pose_per_robot[ridx]

            ###############################################################################################
            # Parse the image observation from Unity                                                      #
            # This is a sample code - to make it faster process images in batches in simulator_vec_env.py #
            # Decoding is done faster with pytorch -> then detach().numpy() and pass to the update()      #
            # the decoded image instead of decoding the images inside each env individually               #
            ###############################################################################################
            per_robot_images.append(self.parse_image_observation(observation, robot_payload, robot_index=ridx))

            # the methods below handles synchronizing states of the DART kinematic chain with the observation from Unity
            # hence it should be always called
            self._unity_retrieve_observation_numeric(robot_payload, observation, robot_index=ridx)
            self._update_dart_chain()
            self._update_env_flags()

            self.prev_dist_ee_box_x = self.prev_dist_ee_box_x_per_robot[ridx]
            self.prev_dist_ee_box_y = self.prev_dist_ee_box_y_per_robot[ridx]
            self.prev_dist_ee_box_rz = self.prev_dist_ee_box_rz_per_robot[ridx]

            action_i = self.action_state[ridx] if isinstance(self.action_state, np.ndarray) and self.action_state.ndim == 2 else self.action_state
            per_robot_rewards.append(float(self.get_reward(action_i)))
            self.prev_dist_ee_box_x = self.get_relative_distance_ee_box_x()   # forward distance (ROS x)
            self.prev_dist_ee_box_y = self.get_relative_distance_ee_box_y()   # lateral distance (ROS y)
            self.prev_dist_ee_box_rz = self.get_relative_distance_ee_box_rz()  # yaw distance (ROS rz)
            self.prev_dist_ee_box_x_per_robot[ridx] = self.prev_dist_ee_box_x
            self.prev_dist_ee_box_y_per_robot[ridx] = self.prev_dist_ee_box_y
            self.prev_dist_ee_box_rz_per_robot[ridx] = self.prev_dist_ee_box_rz
            per_robot_success.append(bool(self.get_object_height() >= self.reward_height_goal and self.collided_env != 1))

        if self.robot_count > 1:
            self.current_obs_img = np.stack([img for img in per_robot_images if img is not None], axis=0) if any(img is not None for img in per_robot_images) else None
        else:
            self.current_obs_img = per_robot_images[0] if per_robot_images else None

        if(time_step_update == True):
            self.time_step += 1

        self._state = self.get_state()
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

    def parse_image_observation(self, observation, robot_payload, robot_index=0):
        """
           Read the unity observation and decode the RGB image

           :param observation: is the observation received from the Unity simulator
           :param robot_payload: per-robot payload from Unity

           :return: decoded RGB image
        """

        overhead_images = observation.get('OverheadImages', None)
        robot_image = robot_payload.get('RobotImage', {}) if isinstance(robot_payload, dict) else {}
        robot_image_data = robot_image.get('Data') if isinstance(robot_image, dict) else None
        image_list = []
        if overhead_images is not None:
            for entry in overhead_images:
                if isinstance(entry, dict):
                    image_list.append(entry.get('Data'))
                else:
                    image_list.append(entry)
        if robot_image_data is not None:
            image_list.append(robot_image_data)
        if not image_list:
            return None

        base64_bytes = image_list[0].encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # cv2.IMREAD_GRAYSCALE - for faster processing but with lower performance (depending on the task)

        #print(image.shape)
        #cv2.imwrite("test.jpg", image)

        return image

    """
    -> Advanced code:
           - Sample code to parallelize the image-proccessing logic. Should be adapted, can not be run
           - Add this code/logic in the simulator_vec_env.py
           - Ideas of Ludwig Graef, TUM

       Advise:
           - Parse the images from the response of the json message, and then call the env.update() with the decode images
           - More advanced idea: try in addition to use nvidia dali. jpeg decoding is faster

        # Sample code starts #

        # Put this in the simulator_vec_env.py #
        observations_preprocessed = preprocessImage(observations_as_literal)
        for env, observation, observation_img in zip(self.envs, observations_as_literal, observations_preprocessed):
            obs, rew, done, info = env.update(observation, observation_img)
            observations_converted.append(obs)

        def preprocessImage(self, observation):
            mean = torch.tensor([0.8, 0.9. 0.99])
            std = torch.tensor([0.8, 0.9. 0.99])
            norm_transform = T.Normalize(mean, std)
            device = "cuda"
            imgs = torch.stack([self.observation_to_image_tensor(obs) for obs in observation])
            imgs = norm_transform(imgs/255).to(device)
            return imgs.numpy()

        def observation_to_image_tensor(self, observation) -> torch.tensor:
            base64_bytes = observation['ImageData'][0].encode('ascii')
            image_bytes = base64.b64decode(base64_bytes)
            buffer2 = bytearray(len(image_bytes))
            buffer2[:] = image_bytes
            image = torch.frombuffer(buffer2, dtype=torch.uint8)
            img = torchvision.io.decode_image(image, torchvision.io.ImageReadMode.RGB)
            return img.float()
        # Sample code ends #
    """
    def render(self, mode='human', monitor_real_values=False, joint_torques=None):
        """
            Override the render definition since we have changed the get_state() function and the actions dimensions
                Note: we do not keep the joints positions in the state of the agent
            refer to iiwa_dart.py for more
        """
        if not self.dart_sim.enable_viewer:
            return False

        return self.dart_sim.render(mode=mode)
