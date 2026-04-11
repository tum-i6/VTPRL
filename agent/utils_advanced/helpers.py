import numpy as np
from utils.config_utils import first_manipulator_instance

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

def get_env(agent_dict, root_dict, sim_dict, gym_environment_dict, manipulator_environment_dict, observation_dict, log_dir, reward_dict=None, goal_dict=None, manual_actions_dict=None, randomBoxesGenerator=None):
    """
        Set up a vectorized environment according to structured configuration dictionaries.

        Args:
            agent_dict: Agent/runtime configuration.
            root_dict: Root simulator configuration.
            sim_dict: Communication/simulator configuration.
            gym_environment_dict: Gym environment configuration.
            manipulator_environment_dict: Manipulator simulator configuration.
            observation_dict: Observation/image configuration.
            log_dir: Directory path used by ``VecMonitor``.
            reward_dict: Optional reward configuration for planar tasks.
            goal_dict: Optional goal/task configuration for planar tasks.
            manual_actions_dict: Optional manual-action configuration.
            randomBoxesGenerator: Optional random-box generator for planar grasping tasks.

        Returns:
            ``VecMonitor``-wrapped ``SimulatorVecEnv`` instance.
    """

    # Some basic checks #
    env_key = gym_environment_dict['env_key']
    mg = gym_environment_dict['manipulator_gym_environment'] if 'manipulator_gym_environment' in gym_environment_dict else {}
    d = mg.get('dart', {})

    if(agent_dict["simulation_mode"] == "train" and env_key == "iiwa_ruckig_planar_grasping_dart_unity_env"):
        raise Exception("Can not train RUCKIG model - aborting")

    if(env_key == "iiwa_ruckig_planar_grasping_dart_unity_env" and agent_dict["model"] != "RUCKIG"):
        raise Exception("For ruckig env use only the RUCKIG model - aborting")

    if(env_key != "iiwa_ruckig_planar_grasping_dart_unity_env" and agent_dict["model"] == "RUCKIG"):
        raise Exception("For non ruckig env do not use the RUCKIG model - aborting")

    if(env_key == "iiwa_end_to_end_planar_grasping_dart_unity_env" and observation_dict['enable_observation_image'] == False):
        raise Exception("For end_to_end env enable the images - aborting")

    if(env_key.find("planar") != -1 and ((reward_dict is None) or (goal_dict is None) or (randomBoxesGenerator is None))):
        raise Exception("Missing dictionaries for planar envs - aborting")

    if(env_key.find("planar") != -1 and (manual_actions_dict is None or manual_actions_dict["manual"] == False)):
        raise Exception("Please enable manual actions for planar envs - aborting")

    if(env_key.find("planar") != -1 and (manual_actions_dict["manual_behaviour"] != "planar_grasping")):
        raise Exception("Please enable manual planar grasing actions for planar envs - aborting")

    manip_cfg = first_manipulator_instance(manipulator_environment_dict)
    ee_enabled = manip_cfg.get("enable_end_effector", False)
    ee_model = manip_cfg.get("end_effector_model", None)
    if(env_key.find("planar") != -1 and (ee_enabled == False or ee_model not in ("ROBOTIQ_3F", "ROBOTIQ_2F85"))):
        raise Exception("Please enable the gripper for planar envs - aborting")

    if(env_key.find("planar") != -1 and (goal_dict["goal_type"] != "box")):
        raise Exception("Please enable boxes as targets for planar envs - aborting")

    if(env_key.find("planar") != -1 and (goal_dict["box_rz_active"] == True and np.isclose(reward_dict["reward_pose_weight"], 0.0))):
        raise Exception("Add more weight to the reward pose to activate 3 DoF control for planar envs when the rotation is active in spanwed boxes - aborting")

    if((manual_actions_dict["manual"] == True) and (manual_actions_dict["manual_behaviour"] == "planar_grasping" or manual_actions_dict["manual_behaviour"] == "close_gripper") and (ee_enabled == False or ee_model not in ("ROBOTIQ_3F", "ROBOTIQ_2F85"))):
        raise Exception("Please enable the gripper for planar_grasping and close_gripper manual actions - aborting")

    if((manual_actions_dict["manual"] == True) and (manual_actions_dict["manual_behaviour"] == "planar_grasping" or manual_actions_dict["manual_behaviour"] == "close_gripper") and (env_key == "iiwa_joint_vel")):
        raise Exception("Can not use dart-based manual actions for iiwa_joint_vel env - aborting")
    # End checks #

    def create_env(id=0):
        """Instantiate one environment instance for a specific vector index.

        Args:
            id: Environment index used for seeding and simulator routing.

        Returns:
            Configured environment instance matching ``env_key``.
        """

        #################################################################################################################################
        # Important: 'dart' substring should always be included in the 'env_key' for dart-based envs. E.g. 'iiwa_sample_dart_unity_env' #
        # If 'dart' is not included, IK behaviour can not be used                                                                       #
        #################################################################################################################################

        # joints control without dart
        if env_key == 'iiwa_joint_vel':
            use_images = bool(observation_dict['enable_observation_image'])
            width = int(observation_dict['observation_image_width'])
            height = int(observation_dict['observation_image_height'])
            img_size = int(min(width, height))
            jv_cfg = {
                'use_images': use_images,
                'image_size': img_size,
                'state': mg['state'],
                'num_joints': gym_environment_dict['num_joints'],
                'manipulator_config': manipulator_environment_dict,
                'manipulator_gym_config': mg,
            }
            env = IiwaJointVelEnv(max_ts=gym_environment_dict['max_time_step'], id=id, config=jv_cfg)

        # Reaching the red target sample env
        # task-space with dart or joint joint space control
        # model-based control with P-controller available
        elif env_key == 'iiwa_sample_dart_unity_env': 
            env = IiwaSampleEnv(max_ts=gym_environment_dict['max_time_step'], orientation_control=d['orientation_control'],
                                use_ik=d['use_inverse_kinematics'], ik_by_sns=d['linear_motion_conservation'],
                                state_type=mg['state'], enable_render=d['enable_dart_viewer'], with_objects=d['with_objects'],
                                target_mode=d['target_mode'], target_path=d['target_path'],
                                goal_type=goal_dict['goal_type'], joints_safety_limit=d['joints_safety_limit'], 
                                max_joint_vel=d['max_joint_vel'], max_ee_cart_vel=d['max_ee_cart_vel'], 
                                max_ee_cart_acc=d['max_ee_cart_acc'], max_ee_rot_vel=d['max_ee_rot_vel'],
                                max_ee_rot_acc=d['max_ee_rot_acc'], random_initial_joint_positions=mg['random_initial_joint_positions'], 
                                initial_positions=mg['initial_positions'], end_effector_model=manip_cfg.get('end_effector_model'), 
                                manipulator_config=manipulator_environment_dict, manipulator_gym_config=mg, env_id=id)

        # Planar RL grasping using the true numeric observations from the UNITY simulator #
        elif env_key == 'iiwa_numerical_planar_grasping_dart_unity_env': 
            env = IiwaNumericalPlanarGraspingEnv(max_ts=gym_environment_dict['max_time_step'], orientation_control=d['orientation_control'],
                                use_ik=d['use_inverse_kinematics'], ik_by_sns=d['linear_motion_conservation'],
                                state_type=mg['state'], enable_render=d['enable_dart_viewer'], with_objects=d['with_objects'],
                                target_mode=d['target_mode'], goal_type=goal_dict['goal_type'],
                                randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=d['joints_safety_limit'],
                                max_joint_vel=d['max_joint_vel'], max_ee_cart_vel=d['max_ee_cart_vel'],
                                max_ee_cart_acc=d['max_ee_cart_acc'], max_ee_rot_vel=d['max_ee_rot_vel'],
                                max_ee_rot_acc=d['max_ee_rot_acc'], random_initial_joint_positions=mg['random_initial_joint_positions'],
                                initial_positions=mg['initial_positions'], noise_enable_rl_obs=agent_dict['noise_enable_rl_obs'],
                                noise_rl_obs_ratio=agent_dict['noise_rl_obs_ratio'], reward_dict=reward_dict,
                                agent_kp=mg['planar']['agent_kp'], agent_kpr=mg['planar']['agent_kpr'],
                                end_effector_model=manip_cfg.get('end_effector_model'),
                                manipulator_config=manipulator_environment_dict, manipulator_gym_config=mg, env_id=id)

        # Planar RL grasping using image observations as state representation - end-to-end learning #
        elif env_key == 'iiwa_end_to_end_planar_grasping_dart_unity_env': 
            width = int(observation_dict['observation_image_width'])
            height = int(observation_dict['observation_image_height'])
            img_size = int(min(width, height))
            env = IiwaEndToEndPlanarGraspingEnv(max_ts=gym_environment_dict['max_time_step'], orientation_control=d['orientation_control'],
                                use_ik=d['use_inverse_kinematics'], ik_by_sns=d['linear_motion_conservation'],
                                state_type=mg['state'], enable_render=d['enable_dart_viewer'],
                                target_mode=d['target_mode'], goal_type=goal_dict['goal_type'],
                                randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=d['joints_safety_limit'],
                                max_joint_vel=d['max_joint_vel'], max_ee_cart_vel=d['max_ee_cart_vel'],
                                max_ee_cart_acc=d['max_ee_cart_acc'], max_ee_rot_vel=d['max_ee_rot_vel'],
                                max_ee_rot_acc=d['max_ee_rot_acc'], random_initial_joint_positions=mg['random_initial_joint_positions'],
                                initial_positions=mg['initial_positions'], noise_enable_rl_obs=agent_dict['noise_enable_rl_obs'],
                                noise_rl_obs_ratio=agent_dict['noise_rl_obs_ratio'], reward_dict=reward_dict,
                                agent_kp=mg['planar']['agent_kp'], agent_kpr=mg['planar']['agent_kpr'],
                                image_size=img_size, end_effector_model=manip_cfg.get('end_effector_model'),
                                manipulator_config=manipulator_environment_dict, manipulator_gym_config=mg, env_id=id)

        # Planar grasping using time-optimal trajectory generation method - RUCKIG #
        elif env_key == 'iiwa_ruckig_planar_grasping_dart_unity_env': 
            env = IiwaRuckigPlanarGraspingEnv(max_ts=gym_environment_dict['max_time_step'], orientation_control=d['orientation_control'],
                                use_ik=d['use_inverse_kinematics'], ik_by_sns=d['linear_motion_conservation'],
                                state_type=mg['state'], enable_render=d['enable_dart_viewer'], with_objects=d['with_objects'],
                                target_mode=d['target_mode'], goal_type=goal_dict['goal_type'],
                                randomBoxesGenerator=randomBoxesGenerator, joints_safety_limit=d['joints_safety_limit'],
                                max_joint_vel=d['max_joint_vel'], max_ee_cart_vel=d['max_ee_cart_vel'],
                                max_ee_cart_acc=d['max_ee_cart_acc'], max_ee_rot_vel=d['max_ee_rot_vel'],
                                max_ee_rot_acc=d['max_ee_rot_acc'], random_initial_joint_positions=mg['random_initial_joint_positions'],
                                initial_positions=mg['initial_positions'], noise_enable_rl_obs=agent_dict['noise_enable_rl_obs'],
                                noise_rl_obs_ratio=agent_dict['noise_rl_obs_ratio'], reward_dict=reward_dict,
                                agent_kp=mg['planar']['agent_kp'], agent_kpr=mg['planar']['agent_kpr'],
                                threshold_p_model_based=mg['planar']["threshold_p_model_based"], end_effector_model=manip_cfg.get('end_effector_model'),
                                manipulator_config=manipulator_environment_dict, manipulator_gym_config=mg, env_id=id)

        # Set env seed #
        env.seed((id * 150) + (id + 11))

        return env

    num_envs = gym_environment_dict['num_envs']
    env = [create_env for i in range(num_envs)]
    env = SimulatorVecEnv(
        env,
        agent_dict,
        root_dict,
        sim_dict,
        gym_environment_dict,
        manipulator_environment_dict,
        reward_dict,
        manual_actions_dict=manual_actions_dict,
        observation_dict=observation_dict,
    ) # Set vectorized env
    env = VecMonitor(env, log_dir, info_keywords=("success",))                # Monitor envs 

    return env
