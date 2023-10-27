"""
Advanced Configuration dict for dart and gym agents

Run this script to update the UNITY configuration.xml (or update the fields manually). See the main method below for instructions

Symbol (*): If you have changed an option in the Config class where there is a "(*)" symbol in the comments, please run this script or
            update the corresponding UNITY field in the .xml file to apply the changes - you may need to restart the simulator

Important:  First fully understand the logic behind config.py and 'iiwa_sample_dart_unity_env', and 'iiwa_joint_vel' envs
"""
import os
import numpy as np
from utils.simulator_configuration import update_simulator_configuration

class ConfigAdvanced:

    def __init__(self):
        self.dart_dict = self.get_dart_env_dict()                 # Dart-related
        self.config_dict = self.get_config_dict()                 # General settings (e.g. number of envs, velocity limits, joints)
        self.env_dict = self.get_env_dict()                       # Environment specific settings
        self.reward_dict = self.get_reward_dict()                 # Reward dict values
        self.hyper_dict = self.get_hyper_dict()                   # Model settings (e.g. PPO, RUCKIG Model-based)
        self.goal_dict = self.get_goal_dict()                     # Green box and red rectangle target settings
        self.manual_actions_dict = self.get_manual_actions_dict() # Hard-coded actions settings (applied at the end of the agent episode)
        self.randomization_dict = self.get_randomization_dict()   # UNITY and agent randomization settings

    @staticmethod
    def get_dart_env_dict():
        dart_env_dict = {
            # episode length in no. of simulation steps
            'max_time_step': 400,

            # to train RL-based models: PPO
            'total_timesteps': 1500000,

            # agents and UNITY control cycles
            'control_cycle': 0.05,                                # (*) agents-related
            'unity_cycle':   0.025,                               # (*) UNITY-related 

            # should control end-effector orientation or not
            'orientation_control': True,

            # when True: actions in task space, when False: actions in joint space
            'use_inverse_kinematics': True,

            # when True: SNS algorithm is used for inverse kinematics to conserve optimal linear motion in task space
            # note: might conflict with training agents in task space velocities, in such cases set it to False
            'linear_motion_conservation': False,

            # when True: the task can be also rendered in the DART viewer
            # Important: Use it when evaluating an agent (e.g. checkpoint). Only for debugging when training an RL agent ('simulation_mode': 'train') - set to False in this case
            # Advanced:  with 'weights & biases', you can log videos during training
            'enable_dart_viewer': False,

            # enable task monitor to visualize states, velocities, agent actions, reward of the robot. 
            # 'enable_dart_viewer' should be set to True
            'task_monitor': True,

            # whether to load additional objects in the DART simulation and viewer - ground, background, etc.
            'with_objects': False,

            # how to spawn the red targets in the dart simulation
            # Options: 'random', 'random_joint_level', 'import', 'fixed', 'None'
            # import   -> import targets from a .csv file (see 'target_path' below)
            # None     -> default behaviour. Can be adapted: See iiwa_sample_dart_unity_env.py, create_target() method
            'target_mode': "None",

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
            # options: 'PPO', 'RUCKIG'
            # RUCKIG is a model-based trajectory-generator. Set env_key to 'iiwa_ruckig_planar_grasping_dart_unity_env'
            'model': 'PPO', 

            # options: 'train', 'evaluate', 'evaluate_model_based" -> only for 'iiwa_sample_dart_unity_env' environment (see below). Refer to the main.py and evaluate.py for more details
            # 'evaluate', for grasping: if the number of test boxes can not be divided by the 'num_envs' (see below), some boxes will not be evaluated 
            'simulation_mode': 'train',

            # available environments
            #'env_key': 'iiwa_sample_dart_unity_env',                     # for control in task space with dart
            #'env_key': 'iiwa_joint_vel',                                 # without dart (only joint velocity control) -> enable gripper (see below). Sample env for testing images state representation
            'env_key': 'iiwa_numerical_planar_grasping_dart_unity_env',  # RL-based planar grasping. Enable manual actions (see below)
            #'env_key': 'iiwa_end_to_end_planar_grasping_dart_unity_env', # image-based planar grasping. Enable manual actions, enable images in the Unity simulator, and set 'use_images' to True
            #'env_key': 'iiwa_ruckig_planar_grasping_dart_unity_env',      # time-optimal planar grasping. Enable manual actions, set 'model' to RUCKIG, and 'simulation_mode' to evaluate

            # environment-specific parameter, e.g, number of links, only relevant for iiwa_joint_vel env
            # currently, num_joints should be always 7 when the DART based environment is used
            'num_joints': 7,

            # camera and images settings: gym and UNITY #
            'use_images':      False,                                     # (*) Use images as state representation. Example code at 'iiwa_joint_vel', _retrieve_image() and 'iiwa_sample_dart_unity_env', update() function
            'image_size':      128,                                       # (*) Gym observation space dimension. See 'iiwa_joint_vel', __init__()
            'camera_position': [0.0, 1.7, 1.6],                           # (*) UNITY
            'camera_rotation': [125.0, 0.0, 180.0],                       # (*) 

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
            # Usage in: 'iiwa_sample_env', and 'iiwa_sample_joint_vel_env'
            'state': 'a',

            # number of environments to run in parallel, 8, 16, ...
            # you may need to restart unity
            # Note: for RUCKIG model set to 1 - no Neural Network - for parallel GPU support 
            'num_envs': 4,

            #########################################################################
            # Logging settings during training: see monitoring_agent.py             #
            #########################################################################
            'log_dir':          "./agent/logs/",                                    # Folder to save the model checkpoints and tensorboard logs
            'save_model_freq':  3125,                                               # Save model checkpoints at this frequency
            'check_freq':       3125,                                               # Frequency to update the 'best_mean_reward' and 'best_model.zip'
            'tb_log_name':      "ppo_log_tb",                                       # Name of the tb folder to save custom metrics/stats. e.g. success rate
            'best_mean_reward': "inf",                                              # Continue training from a checkpoint -> set 'best_mean_reward' to the 'best_mean_reward' value of the loaded model
            #########################################################################

            ###################################################################################################################################
            # Evaluation                                                                                                                      #
            # Important: For base envs              -> support to evaluate only one model per time - do not use 'model_evaluation_type': all' #
            #            For planar grasping envs   -> use 8 envs for 80 boxes, 10 envs for 50 boxes, etc.                                    #
            ###################################################################################################################################

            # Options for 'model_evaluation_type': 'all', or './agent/logs/run_0/best_model', or './agent/logs/run_1/model_25000_0', etc
            #         - 'all'                           -> Scan the agent/logs/ folder and evaulate all the checkpoints that exists in each agen/logs/run_i/ folder
            #         - './agent/logs/run_0/best_model' -> Single model evalaluation. Do not append .zip
            #         - Note:                           -> Use 'all' for model-based evaluation - RUCKIG
            'model_evaluation_type': 'all',
            'evaluation_name':       'planar_grasping_eval',                                                 # Pandas column to name your evaluation experiment
            'reward_baseline':       -95,                                                                    # (Adapt if needed). Reward at time step 0. "untrained" agent. For ploting.
            ##################################################################################################

            ##########################################################################################################
            # Manipulator related                                                                                    #
            # Important: these functionallities are not supported for the standalone env 'iiwa_sample_joint_vel_env' #
            #            - see reset() and __init__ to adapt if needed                                               #
            ##########################################################################################################

            ############################################################################################################################
            # Important: When the manipulator is spawned to a different position than the vertical position,                           #
            #            the parallel envs should terminate at the same timestep due to Unity synchronization behaviour                #
            # Imporant:  when the observation space is an image, can not sample random_initial_joint_positions, set to False           #
            #            if you change the initial position - grasping envs should be adapted - e.g. height goal during manual actions #
            ############################################################################################################################
            'random_initial_joint_positions': False,                         # If set to True, it overrides the values set in 'initial_positions'
            'initial_positions': [0, 0, 0, -np.pi/2, 0, np.pi/2, 0],         # Other options: [0, 0, 0, 0, 0, 0, 0] or None (same as the zero list)

            # Planar Limits - optimized values for sim2real transfer #
            # Note: these values can be increased further            #
            'joints_safety_limit': 10,
            'max_joint_vel':       20,
            'max_ee_cart_vel':     0.035,
            'max_ee_cart_acc':     0.35,
            'max_ee_rot_vel':      0.15,
            'max_ee_rot_acc':      15.0                              # Remove limits to match ruckig

            ##################################################################################################################################
            # Default velocity and joint limits for 'iiwa_sample_dart_unity_env'                                                             #
            # Note: 'joints_safety_limit' -> set to higher value depending on your task and velocity ranges                                  #
            #                                the UNITY behaviour may be unstable when having a 0.0 safety limit with high velocities         #
            #                                e.g. nan tensor joint error -> the robot is in an invalid configuration - reset the manipulator #                
            ##################################################################################################################################
            #'joints_safety_limit': 0.0,
            #'max_joint_vel':       20,   # Joint space
            #'max_ee_cart_vel':     10.0, # Task space -- Not optimized values for sim2real transfer
            #'max_ee_cart_acc':     3.0,
            #'max_ee_rot_vel':      4.0,
            #'max_ee_rot_acc':      1.2
        }

        return config_dict

    def get_env_dict(self):
        """
            Environment specific settings
        """

        env_dict = None

        if(self.config_dict["env_key"].find("planar") != -1): # Planar environment
            env_dict = {
                ########################################################################
                # Agents p-controller settings                                         #
                # Planar RL-agents are using a p-controller to keep fixed some DoF     #
                # so that the robot moves in a planar manner during the episode        # 
                ########################################################################
                'agent_kp':                0.5,                                        # P-controller gain position
                'agent_kpr':               1.5,                                        # P-controller gain rotation
                'threshold_p_model_based': 0.01                                        # Threshold for model-based agents - RUCKIG
                ########################################################################
            }

        return env_dict

    def get_goal_dict(self):
        """
            Goal Settings: used mainly by the "advanced" envs (planar grasping) #
            (*)
        """
        goal_dict = {
            'goal_type': 'box',                                               # 'target' -> red rectangle or 'box' -> green box
            'box_dim':   [0.05, 0.1, 0.05]                                    # (*) UNITY option x, y, z - if you change the height of the box, then adapt in boxes_generator.py the _get_samples() method
        }

        if(goal_dict["goal_type"] == "box"):
            goal_dict_box = {
                'box_type':              'small',                             # 'big' (10x10x10cm), or 'small' (5x10x5). Set 'box_dim' to the correct values
                'box_mode':              "train",                             # 'train', 'val', 'debug'
                'box_max_distance_base': 0.67,                                # Max distance of the box to the base of the robot 
                'box_min_distance_base': 0.475,                               # Min distance of the box to the base of the robot
                'box_folder':            "./agent/logs/dataset/",             # Folder to save the generated dataset
                'box_samples':           200000,                              # Train and val
                'box_split':             0.00025,                             # Val split ratio
                'box_save_val':          True,                                # Save the dataset
                'box_load_val':          False,                               # Load the validation boxes from a saved dataset. Train boxes are always random (seed)
                'box_radius_val':        0.0,                                 # Exclude spawned train boxes that are close to the val boxes within this radius (threshold) in meters. e.g. 0.01
                'box_x_active':          True,                                # If set to False:  x coord of the box will be always 0.0 meters
                'box_x_min':             -0.67,                               # Minimum range for x box coordinate
                'box_x_max':             0.67,                                # Maximum range for x box coordinate
                'box_z_active':          True,
                'box_z_min':             0.42,
                'box_z_max':             0.67,
                'box_ry_active':         True,                                # If set to False -> Boxes will have a fixed rotation (0.0 deg)
                'box_ry_min':            -90,
                'box_ry_max':            0,
                'box_debug':             [-0.1, 0.05, 0.55, 0.0, -90.0, 0.0]  # 'box_mode': 'debug' -> spawn a fixed box for all envs
            }

            if(goal_dict_box['box_type'] == 'small'):
                goal_dict['box_dim'] = [0.05, 0.1, 0.05]
            elif(goal_dict_box['box_type'] == 'big'):
                goal_dict['box_dim'] = [0.1, 0.1, 0.1]

            goal_dict.update(goal_dict_box)

        return goal_dict

    def get_hyper_dict(self):
        """
            Model settings: PPO or RUCKIG
        """

        if(self.config_dict['model'] == 'PPO'):
            hyper_dict = {

                #################################### MOST COMMON #################################################################################################
                'learning_rate':  0.0005,      # Default 0.0003
                'clip_range':     0.25,        # Default 0.2
                'policy':         "MlpPolicy", # Try 'CnnPolicy' with 'iiwa_end_to_end_planar_grasping_dart_unity_env'
                'policy_network': None,        # None (means default), or 'PolicyNetworkVanillaReLU' -> default SB3 network but with ReLU act. func. instead of Tanh
                #################################################################################################################################################

                'ent_coef':       0.02,        # Default 0.0
                'gamma':          0.96,        # Default 0.99
                'gae_lambda':     0.94,        # Default 0.95
                'n_steps':        512,         # Default 2048
                'n_epochs':       10,          # Default 10
                'vf_coef':        0.5,         # Default 0.5
                'max_grad_norm':  0.5,         # Default 0.5
                'batch_size':     32           # Default 64
            }

        # For 'iiwa_ruckig_planar_grasping_dart_unity_env'. Set 'simulation_mode': 'evaluate', and 'model': 'RUCKIG'
        # Set 'num_envs': 1 -> the code can not be parallelized in the GPU (no neural network)
        elif(self.config_dict['model'] == 'RUCKIG'): 
            hyper_dict = {
                'dof':            [1, 1, 1],              # Active dof. [1,1,1] or [0,1,1]. Important: for 011: set 'reward_pose_weight': 0.0  (see below). Activate only 2DoF control
                'target_vel':     [0, 0, 0],              # Velocity of the ee in the target pose (rz, x, y - dart coords)
                'target_acc':     [0, 0, 0],              # Acceleration of the ee in the target pose (rz, x, y - dart coords)
                'max_vel':        [0.15, 0.035, 0.035],   # Limits task-space
                'max_acc':        [1000, 1000, 1000],     # Disabled -> can also be set to some smaller values
                'max_jerk':       [1000, 1000, 1000],
            }

        else:
            return None

        return hyper_dict

    @staticmethod
    def get_reward_dict():
        """
            Novel reward that uses the displacements of the ee to the box (previous distance of the ee to the box minus currrent distance of ee to the box)

            see 'iiwa_numerical_planar_grasping_dart_unity_env'

            Note: adapt if needed to your task

        """

        reward_dict = {
            "reward_collision":       -100,        # One time penalty. When: collision with the box, floor, itself or joints position limits have been violated
            "reward_terminal":        0.0,         # If the box is in the air after the episode -> give this reward. Also, used in the default envs for a successful episode
            "reward_height_goal":     0.25,        # Goal height in meters                      -> then give 'reward_terminal' for planar envs only
            'reward_pose_weight':     1.0,         # Important: Set 'reward_pose_weight': 0.0  to activate 2DoF control
            'reward_pose_norm_const': 0.04,        # Normalization constant
            'reward_x_weight':        1.0,         # Importance of this term
            'reward_x_norm_const':    0.01,
            'reward_z_weight':        1.0,
            'reward_z_norm_const':    0.01
        }

        return reward_dict

    def get_manual_actions_dict(self):
        """
            Manual actions settings.
            Whether to perform manual actions at the end of the RL-episode
            (*)
        """

        manual_actions_dict = {
            'manual':           True,                # Activate manual actions
            'manual_rewards':   True,                # Give collision penalties during the manual actions
            'manual_behaviour': "planar_grasping",   # Option 1: "planar_grasping" -> only for planar envs. Go 'down', 'close', and then 'up' at the end of the episode
                                                     # Option 2: "close_gripper"   -> close the gripper at the end of the episode. Works for non-planar agents too
                                                     # The envs should terminate at the same time step always, see simulator_vec_env.py for explanations - self.step() method
        }

        # Extend the 'manual_actions_dict' dict depending on the user-defined 'manual_behaviour' option #
        behaviour_dict = None
        if(manual_actions_dict["manual_behaviour"] == "planar_grasping"):
            behaviour_dict = self.get_manual_actions_planar_dict()

        elif(manual_actions_dict["manual_behaviour"] == "close_gripper"):
            behaviour_dict = self.get_manual_actions_close_gripper_dict()

        # Extend manual_actions_dict #
        if(behaviour_dict is not None):
            manual_actions_dict.update(behaviour_dict)

        return manual_actions_dict

    def get_manual_actions_planar_dict(self):
        """
            Manual actions dictionary settings for planar grasping

            Actions:
                    1. down
                    2. close
                    3. up
            (*)
        """
        manual_actions_planar_dict = None

        ######################################################################
        # Box 10x10x10 -> (0.055, 4, 15) / (down, close, close_vel)          #
        # Box 5x10x5   -> (0.055, 6, 15) / (down, close, close_vel)          #
        # (*) Change the configuration.xml of the Unity simulator:           #
        # (gripper type <EndEffectorModel> and box dimensions <ItemSize>)    #
        ######################################################################
        manual_actions_planar_dict = {
            'manual_kp':               1.0,                                  # P-controller gain when going down/up for position error
            'manual_kpr':              3.0,                                  # P-controller gain when going down/up for orientation error
            'manual_tol':              0.015,                                # Theshold of the P-controller
            'manual_up_height':        0.35,                                 # Target height
            'manual_steps_same_state': 30,                                   # How many steps the ee is allowed to not moved during the manual actions down/up
            'manual_tol_same_state':   0.01,                                 # If the gripper has not moved more than this distance threshold value within 'steps_same_state' -> it means collision
        }

        if(self.config_dict["robotic_tool"] == "None"):
            manual_actions_planar_dict = None

        # 3-finger gripper #
        elif(self.config_dict["robotic_tool"] == "3_gripper"):
            manual_actions_planar_dict["manual_down_height"] = 0.055         # How much down to move to reach the box
            manual_actions_planar_dict["manual_close_vel"] = 15              # Velocity value to close the gripper per each step

            if(self.goal_dict["box_type"] == "big"):                         # Big box
                manual_actions_planar_dict["manual_close_steps"] = 4         # How many steps to close the gripper (with 'manual_close_vel' per each step)

            elif(self.goal_dict["box_type"] == "small"):                     # Small box
                manual_actions_planar_dict["manual_close_steps"] = 6

            else:
                manual_actions_planar_dict = None

        # 2-finger gripper #
        elif(self.config_dict['robotic_tool'] == "2_gripper"):
            if(self.goal_dict["box_type"] == "small"):                       # Small box
                manual_actions_planar_dict["manual_down_height"] = 0.055
                manual_actions_planar_dict["manual_close_steps"] = 10
                manual_actions_planar_dict["manual_close_vel"] = 15 

            else:                                                            # Important: Gripper 2 can not manipulate the big box
                manual_actions_planar_dict = None

        else:
            manual_actions_planar_dict = None

        return manual_actions_planar_dict

    def get_manual_actions_close_gripper_dict(self):
        """
            Manual actions dictionary settings for closing the gripper at the end of the episode
            (*)
        """
        manual_actions_close_gripper_dict = None

        ######################################################################
        # Box 10x10x10 -> (4, 15) / (close, close_vel)                       #
        # Box 5x10x5   -> (6, 15) / (close, close_vel)                       #
        # (*) Change the configuration.xml of the Unity simulator:           #
        # (gripper type <EndEffectorModel> and box dimensions <ItemSize>)    #
        ######################################################################
        manual_actions_close_gripper_dict = {
        }

        if(self.config_dict["robotic_tool"] is "None"):
            manual_actions_close_gripper_dict = None

        # 3-finger gripper #
        elif(self.config_dict["robotic_tool"] == "3_gripper"):
            manual_actions_close_gripper_dict["manual_close_vel"] = 15       # Velocity value to close the gripper per each step

            if(self.goal_dict["box_type"] == "big"):                         # Big box
                manual_actions_close_gripper_dict["manual_close_steps"] = 4  # How many steps to close the gripper (with 'manual_close_vel' per each step)

            elif(self.goal_dict["box_type"] == "small"):
                manual_actions_close_gripper_dict["manual_close_steps"] = 6

            else:
                manual_actions_close_gripper_dict = None

        # 2-finger gripper #
        elif(self.config_dict['robotic_tool'] == "2_gripper"):
            if(self.goal_dict["box_type"] == "small"):                       # Small box
                manual_actions_close_gripper_dict["manual_close_steps"] = 10
                manual_actions_close_gripper_dict["manual_close_vel"] = 15

            else:                                                            # Important: Gripper 2 can not manipulate the big box
                manual_actions_close_gripper_dict = None

        else:
            manual_actions_close_gripper_dict = None

        return manual_actions_close_gripper_dict

    def get_randomization_dict(self):
        """
            1. Apply noise to the agent state observation see 'iiwa_numerical_planar_grasping_dart_unity_env' - self.get_state() method

            2. (*) UNITY Randomization settings
        """

        randomization_dict = {
            'noise_enable_rl_obs':  False,  # Whether to apply noise in the agent state
            'noise_rl_obs_ratio':   0.05,   # How much noise to apply in % percentage

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

    config = ConfigAdvanced()

    #os.path.dirname(os.path.realpath(__file__)) +
    xml_file = "./simulator/Linux/ManipulatorEnvironment/configuration.xml" # Change the path if needed

    # Update the .xml based on the new changes in the Config class #
    update_simulator_configuration(config, xml_file)

