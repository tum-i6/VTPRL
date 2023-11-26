"""
Configuration dict for dart and gym agents

Run this script to update the UNITY configuration.xml (or update the fields manually). See the main method below for instructions

Symbol (*): If you have changed an option in the Config class where there is a "(*)" symbol in the comments, please run this script or
            update the corresponding UNITY field in the .xml file to apply the changes - you may need to restart the simulator
"""
import os
import numpy as np
from utils.simulator_configuration import update_simulator_configuration

class Config:
    def __init__(self):
        self.dart_dict           = self.get_dart_env_dict()      # Dart-related
        self.config_dict         = self.get_config_dict()        # General and environment settings (e.g. number of envs, velocity limits, joints)
        self.reward_dict         = self.get_reward_dict()        # Reward dict values
        self.goal_dict           = self.get_goal_dict()          # UNITY options: Box dimensions
        self.randomization_dict  = self.get_randomization_dict() # UNITY options: latency, appearance, shadows, etc

    @staticmethod
    def get_dart_env_dict():
        dart_env_dict = {
            # episode length in no. of simulation steps
            'max_time_step': 400,

            # to train RL-based models: PPO
            'total_timesteps': 10000,

            # agents and UNITY control cycles
            'control_cycle': 0.02,                             # (*) agents-related
            'unity_cycle':   0.02,                             # (*) UNITY-related

            # should control end-effector orientation or not
            'orientation_control': True,

            # when True: actions in task space, when False: actions in joint space
            'use_inverse_kinematics': True,

            # when True: SNS algorithm is used for inverse kinematics to conserve optimal linear motion in task space
            # note: might conflict with training agents in task space velocities, in such cases set it to False
            'linear_motion_conservation': False,

            # when True: the task can be also rendered in the DART viewer
            # Important: Use it when evaluating an agent (e.g. checkpoint) . Only for debugging when training an RL agent ('simulation_mode': 'train') - set to False in this case
            # Advanced:  with 'weights & biases', you can log videos during training
            'enable_dart_viewer': False,

            # enable task monitor to visualize states, velocities, agent actions, reward of the robot.
            # 'enable_dart_viewer' should be set to True
            'task_monitor': False,

            # whether to load additional objects in the DART simulation and viewer - ground, background, etc.
            'with_objects': False,

            # how to spawn the red targets in the dart simulation
            # Options: 'random', 'random_joint_level', 'import', 'fixed', 'None'
            # import   -> import targets from a .csv file (see 'target_path' below)
            # None     -> default behaviour. Can be adapted: See iiwa_sample_dart_unity_env, create_target() method
            'target_mode': "random_joint_level",

            # when target_mode is 'import': load targets from a .csv file
            'target_path': "/misc/generated_random_targets/cart_pose_7dof.csv"
        }

        return dart_env_dict

    @staticmethod
    def get_config_dict():
        config_dict = {
            # whether the RL algorithm uses customized or stable-baseline specific parameters
            'custom_hps': True,

            # agent model
            'model': 'PPO',

            # options: 'train', 'evaluate', 'evaluate_model_based" -> only for 'iiwa_sample_dart_unity_env' environment (see below). Refer to the main.py for more details
            # 'evaluate':             load an RL saved checkpoint
            # 'evaluate_model_based': use a p-controller to solve the task for debugging
            'simulation_mode': 'evaluate_model_based',

            # available environments
            'env_key': 'iiwa_sample_dart_unity_env',   # for control in task space with dart
            #'env_key': 'iiwa_joint_vel',              # without dart (only joint velocity control) -> enable gripper (see below). Sample env for testing images state representation

            # environment-specific parameter, e.g, number of links, only relevant for 'iiwa_joint_vel' env
            # currently, num_joints should be always 7 when the DART based environment is used
            'num_joints': 7,

            # camera and images settings: gym and UNITY #
            'use_images':      False,                   # (*) Use images as state representation. Example code at 'iiwa_joint_vel', _retrieve_image() and 'iiwa_sample_dart_unity_env', update() functions
            'image_size':      128,                     # (*) Gym observation space dimension. See 'iiwa_joint_vel', __init__()
            'camera_position': [0.0, 3.0, 0.0],         # (*) UNITY
            'camera_rotation': [90.0, 0.0, -90.0],      # (*)

            # end effector
            'enable_end_effector': True,                # (*) Set to False if no tool is attached. Important: in that case, set 'robotic_tool' to None
            'robotic_tool':        "3_gripper",         # (*) Options: '3_gripper', '2_gripper', 'calibration_pin', 'None' (string). Also, 'end_effector' should be set to True.
                                                        #               -> For 'iiwa_sample_joint_vel_env' select a gripper

            # GRPC, ROS or ZMQ
            'communication_type': 'GRPC',               # (*)

            # the ip address of the server, for windows and docker use 'host.docker.internal',
            # for linux use 'localhost' for connections on local machine
            'ip_address': 'localhost',  # 'host.docker.internal',  # 'localhost'

            # port number for communication with the simulator
            'port_number': '9092', # (*)

            # the seed used for generating pseudo-random sequences
            'seed': 1235,

            # the state of the RL agent in case of numeric values for manipulators 'a' for angles only
            # or 'av' for angles and velocities
            'state': 'a',

            # number of environments to run in parallel, 8, 16, ...
            # you may need to restart unity
            'num_envs': 1,

            #########################################################################
            # Logging settings during training                                      #
            #########################################################################
            'log_dir': "./agent/logs/",                                             # Folder to save the model checkpoints and tensorboard logs
            'tb_log_name': "ppo_log_tb",                                            # Name of the tb folder
            ########################################################################

            ##########################################################################################################
            # Manipulator related                                                                                    #
            # Important: these functionallities are not supported for the standalone env 'iiwa_sample_joint_vel_env' #
            #            - see reset() and __init__ to adapt if needed                                               #
            ##########################################################################################################

            ##################################################################################################################
            # Important: When the manipulator is spawned to a different position than the vertical position,                 #
            #            the parallel envs should terminate at the same timestep due to Unity synchronization behaviour      #
            # Imporant:  when the observation space is an image, can not sample random_initial_joint_positions, set to False #
            ##################################################################################################################
            'random_initial_joint_positions': False,                        # If set to True, it overrides the values set in 'initial_positions'.
            'initial_positions': None,                                      # Example options: [0, 0, 0, 0, 0, 0, 0] same as None, [0, 0, 0, -np.pi/2, 0, np.pi/2, 0]

            ##################################################################################################################################
            # Default velocity and joint limits for 'iiwa_sample_dart_unity_env'                                                             #
            # Note: 'joints_safety_limit' -> set to higher value depending on your task and velocity ranges                                  #
            #                                the UNITY behaviour may be unstable when having a 0.0 safety limit with high velocities         #
            #                                e.g. nan tensor joint error -> the robot is in an invalid configuration - reset the manipulator #                
            ##################################################################################################################################
            'joints_safety_limit': 0.0,  # [deg]
            'max_joint_vel':       20.0, # [deg/s] - Joint space
            'max_ee_cart_vel':     10.0, # [m/s]   - Task space  -- Not optimized values for sim2real transfer 
            'max_ee_cart_acc':     3.0,  # [m/s^2]
            'max_ee_rot_vel':      4.0,  # [rad/s]
            'max_ee_rot_acc':      1.2   # [rad/s^2]
        }

        return config_dict

    @staticmethod
    def get_reward_dict():
        reward_dict = {
            'reward_terminal': 0.0 # Given at the end of the episode if it was successful - see simulator_vec_env.py, step() method during reset
        }

        return reward_dict

    def get_goal_dict(self):
        """
            (*) Box dimensions - UNITY
        """
        goal_dict = {
            'box_dim': [0.05, 0.1, 0.05]    # (*) x, y, z
        }

        return goal_dict

    def get_randomization_dict(self):
        """
            (*) Randomization settings - UNITY
        """

        randomization_dict = {
            'randomize_latency':    False,  # (*) 
            'randomize_torque':     False,  # (*)
            'randomize_appearance': False,  # (*)
            'enable_shadows':       True,   # (*) Note: For sim2real transfer (images) try also False
            'shadow_type':          "Soft", # (*) Options: Soft/Hard
        }

        return randomization_dict

if __name__ == "__main__":
    """
        If you have updated the Config class, you can run this script to update the configuration.xml of the
        UNITY simulator instead of updating the fields manually

        Important: In this case the simulator folder should be located inside the vtprl/ folder
                   with name: simulator/Linux/ManipulatorEnvironment/ManipulatorEnvironment.x86_64, or update the path
                   of the 'xml_file' variable below

        Note:      You many need to restart the UNITY simulator after updating the .xml file
    """
    config = Config()

    #os.path.dirname(os.path.realpath(__file__)) +
    xml_file = "./simulator/Linux/ManipulatorEnvironment/configuration.xml" # Change the path if needed

    # Update the .xml based on the new changes in the Config class #
    update_simulator_configuration(config, xml_file)
