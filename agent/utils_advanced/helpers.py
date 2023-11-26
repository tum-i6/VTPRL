import numpy as np

# Import envs #

# Base envs
from envs.iiwa_sample_joint_vel_env import IiwaJointVelEnv
from envs.iiwa_sample_env import IiwaSampleEnv

# Planar grasping #
from envs_advanced.iiwa_numerical_planar_grasping_env import IiwaNumericalPlanarGraspingEnv
from envs_advanced.iiwa_end_to_end_planar_grasping_env import IiwaEndToEndPlanarGraspingEnv
from envs_advanced.iiwa_ruckig_planar_grasping_env import IiwaRuckigPlanarGraspingEnv

# Monitor envs
from stable_baselines3.common.vec_env import VecMonitor
from simulator_vec_env import SimulatorVecEnv

def get_env(config_dict, dart_env_dict, gym_env_dict, log_dir, reward_dict=None, goal_dict=None, manual_actions_dict=None, randomization_dict=None, randomBoxesGenerator=None):
    """
        Set-up the env according to the input dictionary settings
    """

    # Some basic checks #
    if(config_dict["simulation_mode"] == "train" and config_dict["env_key"] == "iiwa_ruckig_planar_grasping_dart_unity_env"):
        raise Exception("Can not train RUCKIG model - aborting")

    if(config_dict["env_key"] == "iiwa_ruckig_planar_grasping_dart_unity_env" and config_dict["model"] != "RUCKIG"):
        raise Exception("For ruckig env use only the RUCKIG model - aborting")

    if(config_dict["env_key"] != "iiwa_ruckig_planar_grasping_dart_unity_env" and config_dict["model"] == "RUCKIG"):
        raise Exception("For non ruckig env do not use the RUCKIG model - aborting")

    if(config_dict["env_key"] == "iiwa_end_to_end_planar_grasping_dart_unity_env" and config_dict["use_images"] == False):
        raise Exception("For end_to_end env enable the images - aborting")

    if(config_dict["env_key"].find("planar") != -1 and ((reward_dict is None) or (goal_dict is None) or (randomization_dict is None)  or (randomBoxesGenerator is None))):
        raise Exception("Missing dictionaries for planar envs - aborting")

    if(config_dict["env_key"].find("planar") != -1 and (manual_actions_dict is None or manual_actions_dict["manual"] == False)):
        raise Exception("Please enable manual actions for planar envs - aborting")

    if(config_dict["env_key"].find("planar") != -1 and (manual_actions_dict["manual_behaviour"] != "planar_grasping")):
        raise Exception("Please enable manual planar grasing actions for planar envs - aborting")

    if(config_dict["env_key"].find("planar") != -1 and (config_dict["enable_end_effector"] == False or config_dict["robotic_tool"].find("gripper") == -1)):
        raise Exception("Please enable the gripper for planar envs - aborting")

    if(config_dict["env_key"].find("planar") != -1 and (goal_dict["goal_type"] != "box")):
        raise Exception("Please enable boxes as targets for planar envs - aborting")

    if(config_dict["env_key"].find("planar") != -1 and (goal_dict["box_ry_active"] == True and np.isclose(reward_dict["reward_pose_weight"], 0.0))):
        raise Exception("Add more weight to the reward pose to activate 3 DoF control for planar envs when the rotation is active in spanwed boxes - aborting")

    if((manual_actions_dict["manual"] == True) and (manual_actions_dict["manual_behaviour"] == "planar_grasping" or manual_actions_dict["manual_behaviour"] == "close_gripper") and (config_dict["enable_end_effector"] == False or config_dict["robotic_tool"].find("gripper") == -1)):
        raise Exception("Please enable the gripper for planar_grasping and close_gripper manual actions - aborting")

    if((manual_actions_dict["manual"] == True) and (manual_actions_dict["manual_behaviour"] == "planar_grasping" or manual_actions_dict["manual_behaviour"] == "close_gripper") and (config_dict["env_key"] == "iiwa_joint_vel")):
        raise Exception("Can not use dart-based manual actions for iiwa_joint_vel env - aborting")
    # End checks #

    env_key = config_dict['env_key']
    def create_env(id=0):

        #################################################################################################################################
        # Important: 'dart' substring should always be included in the 'env_key' for dart-based envs. E.g. 'iiwa_sample_dart_unity_env' #
        # If 'dart' is not included, IK behaviour can not be used                                                                       #
        #################################################################################################################################

        # joints control without dart
        if env_key == 'iiwa_joint_vel':
            env = IiwaJointVelEnv(max_ts=dart_env_dict['max_time_step'], id=id, config=config_dict)

        # Reaching the red target sample env
        # task-space with dart or joint joint space control
        # model-based control with P-controller available
        elif env_key == 'iiwa_sample_dart_unity_env': 
            env = IiwaSampleEnv(max_ts=dart_env_dict['max_time_step'], orientation_control=dart_env_dict['orientation_control'],
                                use_ik=dart_env_dict['use_inverse_kinematics'], ik_by_sns=dart_env_dict['linear_motion_conservation'],
                                state_type=config_dict['state'], enable_render=dart_env_dict['enable_dart_viewer'],
                                task_monitor=dart_env_dict['task_monitor'], with_objects=dart_env_dict['with_objects'],
                                target_mode=dart_env_dict['target_mode'], target_path=dart_env_dict['target_path'],
                                goal_type=goal_dict['goal_type'], joints_safety_limit=config_dict['joints_safety_limit'], 
                                max_joint_vel=config_dict['max_joint_vel'], max_ee_cart_vel=config_dict['max_ee_cart_vel'], 
                                max_ee_cart_acc=config_dict['max_ee_cart_acc'], max_ee_rot_vel=config_dict['max_ee_rot_vel'],
                                max_ee_rot_acc=config_dict['max_ee_rot_acc'], random_initial_joint_positions=config_dict['random_initial_joint_positions'], 
                                initial_positions=config_dict['initial_positions'], robotic_tool=config_dict["robotic_tool"], 
                                env_id=id)

        # Planar RL grasping using the true numeric observations from the UNITY simulator #
        elif env_key == 'iiwa_numerical_planar_grasping_dart_unity_env': 
            env = IiwaNumericalPlanarGraspingEnv(max_ts=dart_env_dict['max_time_step'], orientation_control=dart_env_dict['orientation_control'],
                                use_ik=dart_env_dict['use_inverse_kinematics'], ik_by_sns=dart_env_dict['linear_motion_conservation'],
                                state_type=config_dict['state'], enable_render=dart_env_dict['enable_dart_viewer'],
                                task_monitor=dart_env_dict['task_monitor'], with_objects=dart_env_dict['with_objects'],
                                target_mode=dart_env_dict['target_mode'], goal_type=goal_dict['goal_type'],
                                randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=config_dict['joints_safety_limit'],
                                max_joint_vel=config_dict['max_joint_vel'], max_ee_cart_vel=config_dict['max_ee_cart_vel'],
                                max_ee_cart_acc=config_dict['max_ee_cart_acc'], max_ee_rot_vel=config_dict['max_ee_rot_vel'],
                                max_ee_rot_acc=config_dict['max_ee_rot_acc'], random_initial_joint_positions=config_dict['random_initial_joint_positions'],
                                initial_positions=config_dict['initial_positions'], noise_enable_rl_obs=randomization_dict['noise_enable_rl_obs'],
                                noise_rl_obs_ratio=randomization_dict['noise_rl_obs_ratio'], reward_dict=reward_dict,
                                agent_kp=gym_env_dict['agent_kp'], agent_kpr=gym_env_dict['agent_kpr'],
                                robotic_tool=config_dict["robotic_tool"],
                                env_id=id)

        # Planar RL grasping using image observations as state representation - end-to-end learning #
        elif env_key == 'iiwa_end_to_end_planar_grasping_dart_unity_env': 
            env = IiwaEndToEndPlanarGraspingEnv(max_ts=dart_env_dict['max_time_step'], orientation_control=dart_env_dict['orientation_control'],
                                use_ik=dart_env_dict['use_inverse_kinematics'], ik_by_sns=dart_env_dict['linear_motion_conservation'],
                                state_type=config_dict['state'], enable_render=dart_env_dict['enable_dart_viewer'], task_monitor=dart_env_dict['task_monitor'],
                                target_mode=dart_env_dict['target_mode'], goal_type=goal_dict['goal_type'],
                                randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=config_dict['joints_safety_limit'],
                                max_joint_vel=config_dict['max_joint_vel'], max_ee_cart_vel=config_dict['max_ee_cart_vel'],
                                max_ee_cart_acc=config_dict['max_ee_cart_acc'], max_ee_rot_vel=config_dict['max_ee_rot_vel'],
                                max_ee_rot_acc=config_dict['max_ee_rot_acc'], random_initial_joint_positions=config_dict['random_initial_joint_positions'],
                                initial_positions=config_dict['initial_positions'], noise_enable_rl_obs=randomization_dict['noise_enable_rl_obs'],
                                noise_rl_obs_ratio=randomization_dict['noise_rl_obs_ratio'], reward_dict=reward_dict,
                                agent_kp=gym_env_dict['agent_kp'], agent_kpr=gym_env_dict['agent_kpr'],
                                image_size=config_dict['image_size'], robotic_tool=config_dict["robotic_tool"],
                                gripper_type=config_dict["gripper_type"],
                                env_id=id)

        # Planar grasping using time-optimal trajectory generation method - RUCKIG #
        elif env_key == 'iiwa_ruckig_planar_grasping_dart_unity_env': 
            env = IiwaRuckigPlanarGraspingEnv(max_ts=dart_env_dict['max_time_step'], orientation_control=dart_env_dict['orientation_control'],
                                use_ik=dart_env_dict['use_inverse_kinematics'], ik_by_sns=dart_env_dict['linear_motion_conservation'],
                                state_type=config_dict['state'], enable_render=dart_env_dict['enable_dart_viewer'],
                                task_monitor=dart_env_dict['task_monitor'], with_objects=dart_env_dict['with_objects'],
                                target_mode=dart_env_dict['target_mode'], goal_type=goal_dict['goal_type'],
                                randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=config_dict['joints_safety_limit'],
                                max_joint_vel=config_dict['max_joint_vel'], max_ee_cart_vel=config_dict['max_ee_cart_vel'],
                                max_ee_cart_acc=config_dict['max_ee_cart_acc'], max_ee_rot_vel=config_dict['max_ee_rot_vel'],
                                max_ee_rot_acc=config_dict['max_ee_rot_acc'], random_initial_joint_positions=config_dict['random_initial_joint_positions'],
                                initial_positions=config_dict['initial_positions'], noise_enable_rl_obs=randomization_dict['noise_enable_rl_obs'],
                                noise_rl_obs_ratio=randomization_dict['noise_rl_obs_ratio'], reward_dict=reward_dict,
                                agent_kp=gym_env_dict['agent_kp'], agent_kpr=gym_env_dict['agent_kpr'],
                                threshold_p_model_based=gym_env_dict["threshold_p_model_based"], robotic_tool=config_dict["robotic_tool"],
                                env_id=id)

        # Set env seed #
        env.seed((id * 150) + (id + 11))

        return env

    num_envs = config_dict['num_envs']
    env = [create_env for i in range(num_envs)]
    env = SimulatorVecEnv(env, config_dict, manual_actions_dict, reward_dict) # Set vectorized env
    env = VecMonitor(env, log_dir, info_keywords=("success",))                # Monitor envs 

    return env
