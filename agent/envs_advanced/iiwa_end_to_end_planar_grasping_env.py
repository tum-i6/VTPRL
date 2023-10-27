"""
A vision-based end-to-end RL planar grasping Env class inheriting from the IiwaNumericalPlanarGraspingEnv for the Kuka LBR iiwa manipulator with 7 links and a Gripper
The parent class takes care of integrating DART Engine with Unity simulation environment

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
import cv2
import base64

class IiwaEndToEndPlanarGraspingEnv(IiwaNumericalPlanarGraspingEnv):
    def __init__(self, max_ts, orientation_control, use_ik, ik_by_sns, state_type, use_images=True, enable_render=False, task_monitor=False, target_mode="None", goal_type="box",
                 randomBoxesGenerator=None, joints_safety_limit=10, max_joint_vel=20, max_ee_cart_vel=0.035, max_ee_cart_acc =10, max_ee_rot_vel=0.15, max_ee_rot_acc=10,
                 random_initial_joint_positions=False, initial_positions=[0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2],
                 noise_enable_rl_obs=False, noise_rl_obs_ratio=0.05, reward_dict=None, agent_kp=0.5, agent_kpr=1.5,
                 robotic_tool="3_gripper", image_size=128, env_id=0):

        if(use_images != True):
            raise Exception("End-to-end vision-based env requires use_images to be set to True - abort")

        # the init of the parent class should be always called, this will in the end call reset() once
        super().__init__(max_ts=max_ts, orientation_control=orientation_control, use_ik=use_ik, ik_by_sns=ik_by_sns, state_type=state_type, enable_render=enable_render, task_monitor=task_monitor, target_mode=target_mode, goal_type=goal_type, 
                         randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=joints_safety_limit, max_joint_vel=max_joint_vel, max_ee_cart_vel=max_ee_cart_vel,
                         max_ee_cart_acc=max_ee_cart_acc, max_ee_rot_vel=max_ee_rot_vel, max_ee_rot_acc=max_ee_rot_acc,
                         random_initial_joint_positions=random_initial_joint_positions, initial_positions=initial_positions,noise_enable_rl_obs=False,noise_rl_obs_ratio=0.05,
                         reward_dict=reward_dict, agent_kp=agent_kp, agent_kpr=agent_kpr, robotic_tool=robotic_tool, env_id=env_id)

        self.use_images = use_images
        self.image_size = image_size
        self.current_obs_img = None

        ###################################
        # RGB image                       #
        # Unormalized observation for now #
        # Adapt to your project           #
        ###################################
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8)

        #######################################################
        # Set-up the task_monitor                             #
        #######################################################
        if self.task_monitor:
            self._create_task_monitor()

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

        ###############################################################################################
        # Parse the image observation from Unity                                                      #
        # This is a sample code - to make it faster process images in batches in simulator_vec_env.py #
        # Decoding is done faster with pytorch -> then detach().numpy() and pass to the update()      #
        # the decoded image instead of decoding the images inside each env individually               #
        ###############################################################################################
        self.current_obs_img = self.parse_image_observation(observation)

        # Collision or joints limits overpassed                                      #
        # env will not be reseted - wait until the end of the episoe for planar envs #
        if observation['Observation'][-1] == 1.0 or self.joints_limits_violation():
            self.collided_env = 1
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
        info = {"success": False}                    # Episode was successful. It is set at simulator_vec_env.py before reseting

        # Keep track the previous distance of the ee to the box - used in the reward function #
        self.prev_dist_ee_box_z = self.get_relative_distance_ee_box_z_unity()
        self.prev_dist_ee_box_x = self.get_relative_distance_ee_box_x_unity()
        self.prev_dist_ee_box_ry = self.get_relative_distance_ee_box_ry_unity() 

        self.prev_action = self.action_state

        return state, reward, done, info

    def parse_image_observation(self, observation):
        """
           Read the unity observation and decode the RGB image

           :param observation: is the observation received from the Unity simulator

           :return: decoded RGB image
        """

        base64_bytes = observation['ImageData'][0].encode('ascii')
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

    ##############
    # Monitoring #
    ##############
    def _create_task_monitor(self, plot_joint_position=True, plot_joint_velocity=True, plot_joint_acceleration=False, plot_joint_torque=True, plot_agent_state=False, plot_agent_action=True, plot_agent_reward=True):
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

        plot_agent_state = False
        # Unused -> state observation is an image #
        # Do not visualize the agent state        #
        self.monitor_n_states = 2
        state_chart_categories = ['X', 'Z']
        if self.action_space_dimension == 3:
            self.monitor_n_states += 1
            state_chart_categories = ['X', 'Z', 'RY']
        # Unused ###################################

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

    def render(self, mode='human', monitor_real_values=False, joint_torques=None):
        """
            Override the render definition since we have changed the get_state() function and the actions dimensions
                Note: we do not keep the joints positions in the state of the agent
            refer to iiwa_dart.py and task_monitor.py for more
        """
        if not self.dart_sim.enable_viewer:
            return False

        if self.task_monitor:
            if self.monitor_window.plot_joint_torque:
                if monitor_real_values and joint_torques is not None:
                    joint_torques = joint_torques
                else:
                    self.dart_sim.chain.computeInverseDynamics()
                    joint_torques = self.dart_sim.chain.getForces()

            self.monitor_window.update_values(values_joint_position=np.rad2deg(self.dart_sim.chain.getPositions()),
                                              values_joint_velocity=np.rad2deg(self.dart_sim.chain.getVelocities()),
                                              values_joint_acceleration=np.rad2deg(self.dart_sim.chain.getAccelerations()),
                                              values_joint_torque=joint_torques,
                                              values_agent_action=self.action_state,
                                              values_agent_reward=[self.reward_state])
            self.monitor_app.processEvents()

        return self.dart_sim.render(mode=mode)
