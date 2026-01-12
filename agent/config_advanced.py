"""
Advanced Configuration dict for dart and gym agents

Run this script to update the UNITY configuration.xml (or update the fields manually). See the main method below for instructions

Symbol (*): If you have changed an option in the Config class where there is a '(*)' symbol in the comments, please run this script or
            update the corresponding UNITY field in the .xml file to apply the changes - you may need to restart the simulator

Important:  First fully understand the logic behind config.py and 'iiwa_sample_dart_unity_env', and 'iiwa_joint_vel' envs
"""
import os
import numpy as np
from utils.simulator_configuration import update_simulator_configuration
from config import Config

class ConfigAdvanced(Config):
    def __init__(self):
        # Initialize the base structured configuration (agent/gym + Unity XML groups)
        super().__init__()

        ########################################################################################################
        # Override/extend agent-related groups to include advanced settings (hyper, logging, noise, manual)    #
        ########################################################################################################
        self.agent_dict = self.get_agent_dict()

        ########################################################################################################
        # Override/extend gym environment groups (env_key options, manipulator planar settings, goals, reward) #
        ########################################################################################################
        # Build advanced gym dict
        gym_adv = self.get_gym_environment_dict()
        # Compute goal using simulator manipulator settings and inject it
        goal_dict = ConfigAdvanced.get_goal_dict(self.manipulator_environment_dict)
        gym_adv.get('manipulator_gym_environment', {}).update({'goal': goal_dict})
        self.gym_environment_dict = gym_adv

        # Finalize manual actions now that ee model and goal/box type are known
        self.agent_dict['manual_actions'] = self.get_manual_actions_dict(self.manipulator_environment_dict, goal_dict)

    ####################################
    # Agent-related groups (overrides) #
    ####################################
    @staticmethod
    def get_agent_dict():
        """
        Agent-level configuration (algorithm, runtime, logging, hyper-parameters, and Python-side randomization).

        Includes algorithm selection and high-level runtime settings.
        """
        base = Config.get_agent_dict()
        # agent model
        # options: 'PPO', 'RUCKIG'
        # RUCKIG is a model-based trajectory-generator. Set env_key to 'iiwa_ruckig_planar_grasping_dart_unity_env'
        base['model'] = 'PPO'

        #########################################################################
        # Logging settings during training: see monitoring_agent.py             #
        #########################################################################
        base['save_model_freq']  = 3125                                         # Save model checkpoints at this frequency
        base['check_freq']       = 3125                                         # Frequency to update the 'best_mean_reward' and 'best_model.zip'
        base['best_mean_reward'] = "inf"                                        # Continue training from a checkpoint -> set 'best_mean_reward' to the 'best_mean_reward' value of the loaded model

        ###################################################################################################################################
        # Evaluation                                                                                                                      #
        # Important: For base envs              -> support to evaluate only one model per time - do not use 'model_evaluation_type': all' #
        #            For planar grasping envs   -> use 8 envs for 80 boxes, 10 envs for 50 boxes, etc.                                    #
        ###################################################################################################################################
        # Options for 'model_evaluation_type': 'all', or './agent/logs/run_0/best_model', or './agent/logs/run_1/model_25000_0', etc
        #         - 'all'                           -> Scan the agent/logs/ folder and evaulate all the checkpoints that exists in each agen/logs/run_i/ folder
        #         - './agent/logs/run_0/best_model' -> Single model evalaluation. Do not append .zip
        #         - Note:                           -> Use 'all' for model-based evaluation - RUCKIG
        base['model_evaluation_type'] = 'all'
        base['evaluation_name']       = 'planar_grasping_eval'                  # Pandas column to name your evaluation experiment
        base['reward_baseline']       = -95                                     # (Adapt if needed). Reward at time step 0. "untrained" agent. For ploting.

        #############################################
        # Model settings (e.g., PPO or RUCKIG)      #
        #############################################
        base['hyper_dict'] = ConfigAdvanced.get_hyper_dict(base['model'])

        ###########################################################################################################
        # Python-side randomization (noise) – not simulator randomization (which stays in Unity XML dictionaries) #
        ###########################################################################################################
        base['noise_enable_rl_obs'] = False                                    # Whether to apply noise in the agent state
        base['noise_rl_obs_ratio']  = 0.05                                     # How much noise to apply in % percentage

        #############################################
        # Manual actions (advanced, used by agents) #
        #############################################
        # Manual actions depend on simulator gripper and goal; set in __init__ later
        base['manual_actions'] = None
        return base

    ###########################################
    # Gym/DART groups (overrides and extends) #
    ###########################################
    @staticmethod
    def get_gym_environment_dict():
        """
        Gym environment settings for Python agents (advanced). This aggregates
        common parameters and specialized sub-configs for manipulator planar tasks.
        """
        gym = Config.get_gym_environment_dict()

        # available manipulator environments (*) 'env_key' should include 'iiwa' or 'so100' -- relevant to set Unity's manipulator_model
        gym['env_key'] = 'iiwa_numerical_planar_grasping_dart_unity_env'        # RL-based planar grasping. Enable manual actions (see below)
        # gym['env_key'] = 'iiwa_end_to_end_planar_grasping_dart_unity_env'       # image-based planar grasping. Enable manual actions, enable images in the Unity simulator, and set 'use_images' to True
        # gym['env_key'] = 'iiwa_ruckig_planar_grasping_dart_unity_env'           # time-optimal planar grasping. Enable manual actions, set 'model' to RUCKIG, and 'simulation_mode' to evaluate

        # Note: for RUCKIG model set 'num_envs' to 1 - no Neural Network - for parallel GPU support 

        # Manipulator gym environment overrides and extensions
        mg = gym['manipulator_gym_environment']
        gym['task_monitor'] = True
        
        mg['initial_positions'] = [0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/2]

        ###############################################
        # Planar controller parameters (advanced)     #
        ###############################################
        mg['planar'] = {
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

        # DART-specific advanced overrides
        d = mg['dart']
        d['target_mode'] = 'None'

        ##########################################################
        # Planar Limits - optimized values for sim2real transfer #
        # Note: these values can be increased further            #
        ##########################################################
        d['joints_safety_limit'] = 10.0
        d['max_joint_vel']       = 20.0
        d['max_ee_cart_vel']     = 0.035
        d['max_ee_cart_acc']     = 0.35
        d['max_ee_rot_vel']      = 0.15
        d['max_ee_rot_acc']      = 15.0  # Remove limits to match ruckig

        # Placeholders; goal injected in __init__, reward static here
        mg['goal'] = {}
        mg['reward'] = ConfigAdvanced.get_reward_dict()

        return gym

    @staticmethod
    def get_hyper_dict(model_name):
        """
            Model settings: PPO or RUCKIG
        """
        if(model_name == 'PPO'):
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
        elif(model_name == 'RUCKIG'): 
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

    @staticmethod
    def get_goal_dict(manipulator_environment_dict):
        """
            Goal Settings: used mainly by the "advanced" envs (planar grasping) #
            (*)
        """
        # Use simulator's manipulator.item settings to drive goal defaults
        items_list = manipulator_environment_dict.get('items', []) if isinstance(manipulator_environment_dict, dict) else []
        item_dict = items_list[0] if (isinstance(items_list, list) and len(items_list) > 0 and isinstance(items_list[0], dict)) else {}

        if(item_dict.get('item_type') == "BOX"):
            goal_dict_box = {
                'goal_type':            'box',                                # Keep advanced envs focused on box targets
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

            # Mirror box type into simulator item size to keep them consistent
            if isinstance(item_dict, dict):
                if(goal_dict_box['box_type'] == 'small'):
                    item_dict['item_size'] = [0.05, 0.1, 0.05]
                elif(goal_dict_box['box_type'] == 'big'):
                    item_dict['item_size'] = [0.1, 0.1, 0.1]

            return goal_dict_box

        # Default when no BOX item is present
        return {
            'goal_type':    'box',
            'box_type':        'small',
            'box_mode':        'train',
            'box_samples':     0,
            'box_save_val':    False,
            'box_load_val':    False,
            'box_radius_val':  0.0,
            'box_x_active':    True,
            'box_z_active':    True,
            'box_ry_active':   True,
        }

    @staticmethod
    def get_manual_actions_dict(manipulator_environment_dict, goal_dict):
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
        ee_model = (manipulator_environment_dict or {}).get('end_effector_model', None)
        box_type = (goal_dict or {}).get('box_type', 'small')

        if(manual_actions_dict["manual_behaviour"] == "planar_grasping"):
            behaviour_dict = ConfigAdvanced.get_manual_actions_planar_dict(ee_model, box_type)

        elif(manual_actions_dict["manual_behaviour"] == "close_gripper"):
            behaviour_dict = ConfigAdvanced.get_manual_actions_close_gripper_dict(ee_model, box_type)

        # Extend manual_actions_dict #
        if(behaviour_dict is not None):
            manual_actions_dict.update(behaviour_dict)

        return manual_actions_dict

    @staticmethod
    def get_manual_actions_planar_dict(ee_model, box_type):
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

        if(ee_model in (None, 'None')):
            manual_actions_planar_dict = None

        # 3-finger gripper #
        elif(ee_model == "ROBOTIQ_3F"):
            manual_actions_planar_dict["manual_down_height"] = 0.055            # How much down to move to reach the box
            manual_actions_planar_dict["manual_close_vel"] = 15                 # Velocity value to close the gripper per each step

            if(box_type == "big"):                                              # Big box
                manual_actions_planar_dict["manual_close_steps"] = 4            # How many steps to close the gripper (with 'manual_close_vel' per each step)

            elif(box_type == "small"):                                          # Small box
                manual_actions_planar_dict["manual_close_steps"] = 6

            else:
                manual_actions_planar_dict = None

        # 2-finger gripper #
        elif(ee_model == "ROBOTIQ_2F85"):
            if(box_type == "small"):                                            # Small box
                manual_actions_planar_dict["manual_down_height"] = 0.055
                manual_actions_planar_dict["manual_close_steps"] = 10
                manual_actions_planar_dict["manual_close_vel"] = 15 

            else:                                                               # Important: Gripper 2 can not manipulate the big box
                manual_actions_planar_dict = None

        else:
            manual_actions_planar_dict = None

        return manual_actions_planar_dict

    @staticmethod
    def get_manual_actions_close_gripper_dict(ee_model, box_type):
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
        manual_actions_close_gripper_dict = {}

        if(ee_model in (None, 'None')):
            manual_actions_close_gripper_dict = None

        # 3-finger gripper #
        elif(ee_model == "ROBOTIQ_3F"):
            manual_actions_close_gripper_dict["manual_close_vel"] = 15          # Velocity value to close the gripper per each step

            if(box_type == "big"):                                              # Big box
                manual_actions_close_gripper_dict["manual_close_steps"] = 4     # How many steps to close the gripper (with 'manual_close_vel' per each step)

            elif(box_type == "small"):
                manual_actions_close_gripper_dict["manual_close_steps"] = 6

            else:
                manual_actions_close_gripper_dict = None

        # 2-finger gripper #
        elif(ee_model == "ROBOTIQ_2F85"):
            if(box_type == "small"):                                            # Small box
                manual_actions_close_gripper_dict["manual_close_steps"] = 10
                manual_actions_close_gripper_dict["manual_close_vel"] = 15

            else:                                                               # Important: Gripper 2 can not manipulate the big box
                manual_actions_close_gripper_dict = None

        else:
            manual_actions_close_gripper_dict = None

        return manual_actions_close_gripper_dict    


if __name__ == "__main__":
    """
        If you have updated the Config class, you can run this script to update the configuration.xml of the
        UNITY simulator instead of updating the fields manually

        Important: In this case the simulator folder should be located inside the vtprl/ folder
                   with name e.g., simulator/Linux/VTPRL-Simulator.x86_64 (or .exe),
                   or update the path of the 'xml_file' variable below

        Note:      You many need to restart the UNITY simulator after updating the .xml file
    """
    config = ConfigAdvanced()

    # Change the path if needed
    simulator_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/environment/simulator/'
    simulator_version = 'v1.0.0'
    simulator_platform = 'Windows'  # 'Linux', 'Mac'
    xml_file = simulator_path + simulator_version + '/' + simulator_platform + '/configuration.xml'

    # Update the .xml based on the new changes in the Config class #
    update_simulator_configuration(config, xml_file)
    print('Successfully updated the simulator configuration file:')
    print(xml_file)
