import os
# import time

from config import Config

import numpy as np
from pathlib import Path

# Import envs #
from simulator_vec_env import SimulatorVecEnv
from envs.iiwa_sample_joint_vel_env import IiwaJointVelEnv
from envs.iiwa_sample_env import IiwaSampleEnv
from envs.warehouse_unity_env import WarehouseUnityEnv
from envs.so100_sample_env import SO100SampleEnv

# Monitor envs 
from stable_baselines3.common.vec_env import VecMonitor

# Models #
from stable_baselines3 import PPO

from utils.helpers import set_seeds

def get_env(agent, root, sim, gym_cfg, manip_env, warehouse_env, observation, reward_dict, log_dir):
    """Create the vec env from structured config dictionaries."""

    env_key = gym_cfg['env_key']
    def create_env(id=0):

        #################################################################################################################################
        # Important: 'dart' substring should always be included in the 'env_key' for dart-based envs. E.g. 'iiwa_sample_dart_unity_env' #
        # If 'dart' is not included, IK behaviour can not be used                                                                       #
        #################################################################################################################################

        # joints control without dart
        if env_key == 'iiwa_joint_vel':
            use_images = bool(observation['enable_observation_image'])
            width = int(observation['observation_image_width'])
            height = int(observation['observation_image_height'])
            img_size = int(min(width, height))
            jv_cfg = {
                'use_images': use_images,
                'image_size': img_size,
                'state': gym_cfg['manipulator_gym_environment']['state'],
                'num_joints': gym_cfg['num_joints'],
            }
            env = IiwaJointVelEnv(max_ts=gym_cfg['max_time_step'], id=id, config=jv_cfg)

        # Reaching the red target sample env
        # task-space with dart or joint space control
        # model-based control with P-controller available
        elif env_key == 'iiwa_sample_dart_unity_env':
            mg = gym_cfg['manipulator_gym_environment']
            d = mg['dart']
            env = IiwaSampleEnv(max_ts=gym_cfg['max_time_step'], orientation_control=d['orientation_control'],
                                use_ik=d['use_inverse_kinematics'], ik_by_sns=d['linear_motion_conservation'],
                                state_type=mg['state'], enable_render=d['enable_dart_viewer'], with_objects=d['with_objects'],
                                target_mode=d['target_mode'], target_path=d['target_path'],
                                goal_type="target", joints_safety_limit=d['joints_safety_limit'],
                                max_joint_vel=d['max_joint_vel'], max_ee_cart_vel=d['max_ee_cart_vel'],
                                max_ee_cart_acc=d['max_ee_cart_acc'], max_ee_rot_vel=d['max_ee_rot_vel'],
                                max_ee_rot_acc=d['max_ee_rot_acc'], random_initial_joint_positions=mg['random_initial_joint_positions'],
                                initial_positions=mg['initial_positions'], end_effector_model=manip_env['end_effector_model'],
                                env_id=id)

        elif env_key == 'so100_sample_dart_unity_env':
            mg = gym_cfg['manipulator_gym_environment']
            d = mg['dart']
            env = SO100SampleEnv(max_ts=gym_cfg['max_time_step'], orientation_control=d['orientation_control'],
                                 use_ik=d['use_inverse_kinematics'], ik_by_sns=d['linear_motion_conservation'],
                                 state_type=mg['state'], enable_render=d['enable_dart_viewer'], with_objects=d['with_objects'],
                                 target_mode=d['target_mode'], target_path=d['target_path'],
                                 goal_type="target", joints_safety_limit=d['joints_safety_limit'],
                                 max_joint_vel=d['max_joint_vel'], max_ee_cart_vel=d['max_ee_cart_vel'],
                                 max_ee_cart_acc=d['max_ee_cart_acc'], max_ee_rot_vel=d['max_ee_rot_vel'],
                                 max_ee_rot_acc=d['max_ee_rot_acc'], random_initial_joint_positions=mg['random_initial_joint_positions'],
                                 initial_positions=mg['initial_positions'], end_effector_model=manip_env['end_effector_model'],
                                 env_id=id)

        elif env_key == 'warehouse_unity_env':
            env = WarehouseUnityEnv(
                max_time_steps=gym_cfg['max_time_step'],
                env_id=id,
                gym_config=gym_cfg['warehouse_gym_environment'],
                warehouse_config=warehouse_env,
                observation_config=observation,
            )

        # Set env seed #
        env.seed((id * 150) + (id + 11))

        return env

    num_envs = gym_cfg['num_envs']
    env = [create_env for i in range(num_envs)]
    env = SimulatorVecEnv(
        env,
        agent,
        root,
        sim,
        gym_cfg,
        manip_env,
        reward_dict=reward_dict,
        observation_dict=observation,
    )   # Set vectorized env
    env = VecMonitor(env, log_dir, info_keywords=("success",))                                  # Monitor envs

    return env


if __name__ == "__main__":

    main_config = Config()

    # Structured config handles
    agent = main_config.agent_dict
    root = main_config.root_dict
    sim = main_config.simulation_dict
    gym = main_config.gym_environment_dict
    manip_env = main_config.manipulator_environment_dict
    warehouse_env = main_config.warehouse_environment_dict
    observation = main_config.observation_dict

    # Reward dict (terminal reward used in SimulatorVecEnv)
    reward_dict = gym['manipulator_gym_environment']['reward']

    # Create new folder if not exists for logging #
    Path(agent["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Build env #
    env = get_env(agent, root, sim, gym, manip_env, warehouse_env, observation, reward_dict, agent["log_dir"])

    try:
        # Train the agent #
        if(agent['simulation_mode'] == 'train'):

            # Set global seeds and get a PPO seed #
            ppo_seed = set_seeds(sim["random_seed"])

            # Define the model and its hyperparameters #
            model = PPO(policy="MlpPolicy", env=env, seed=ppo_seed, tensorboard_log=agent["log_dir"], verbose=1)

            # Play some episodes                                                      #
            # If you retrain the model, you may need to set reset_num_timesteps=False #
            model.learn(total_timesteps=agent["total_timesteps"], reset_num_timesteps=True, tb_log_name=agent["tb_log_name"], log_interval=2)

            print("Training ended. Saving a checkpoint at: " + agent["log_dir"])

            # Save the last model #
            model.save(os.path.join(agent["log_dir"], "ppo_trained"))

            del model  # remove

        elif(agent['simulation_mode'] == 'evaluate'):
            print("===================================================")
            print("RL-based evaluation")
            print("===================================================")

            # Load trained agent #
            model = PPO.load(os.path.join(agent["log_dir"], "ppo_trained"))
            model.policy.set_training_mode(False)

            obs = env.reset()
            for x in range(1000):                                                # Run some steps for each env 
                action, _states = model.predict(obs, deterministic=True)         # Important: set deterministic to True to use the best learned policy (no exploration)

                obs, rewards, dones, info = env.step(action)

                # Render #
                if gym['manipulator_gym_environment']['dart']['enable_dart_viewer'] and gym['env_key'] != 'iiwa_joint_vel':
                    env.render()

        elif(agent['simulation_mode'] == 'evaluate_model_based' and 'dart' in gym['env_key']):
            # check model-based controllers (e.g. P-controller) #
            print("===================================================")
            print("Model-based evaluation")
            print("===================================================")

            control_kp = 1.0 / env.observation_space.high[0]

            obs = env.reset()
            episode_rewards = []
            for _ in range(5): # Play some episodes 
                cum_reward = 0

                while True: # Play until we have a successful episode 
                    if gym['manipulator_gym_environment']['dart']['use_inverse_kinematics']:                        # Generate an action for the current observation using a P-controller
                        action = np.reshape(env.env_method('action_by_p_control', control_kp, 2.0 * control_kp),
                                            (gym['num_envs'], env.action_space.shape[0]))
                    else:                                                                                           # Random action
                        action = np.reshape(env.env_method('random_action'),
                                            (gym['num_envs'], env.action_space.shape[0]))

                    obs, rewards, dones, info = env.step(action)                                                    # Play this action
                    cum_reward += rewards

                    # Render #
                    if gym['manipulator_gym_environment']['dart']['enable_dart_viewer']:
                        env.render()

                    if dones.any():
                        episode_rewards.append(cum_reward)
                        break

            mean_reward = np.mean(episode_rewards)
            print("Mean reward: " + str(mean_reward))

        elif(agent['simulation_mode'] == 'evaluate_model_based' and gym['env_key'] == 'warehouse_unity_env'):
            print("===================================================")
            print("Warehouse model-based evaluation")
            print("===================================================")

            obs = env.reset()
            episode_rewards = []
            for _ in range(3):
                cum_reward = 0.0
                while True:
                    # Use simple P controller provided by the env
                    action = np.reshape(env.env_method('action_by_p_control', 1.0, 2.0), (gym['num_envs'], env.action_space.shape[0]))
                    # step_time = time.time()
                    obs, rewards, dones, info = env.step(action)
                    # print("Step time (ms): " + str((time.time() - step_time)*1000.0))
                    cum_reward += rewards
                    if dones.any():
                        episode_rewards.append(cum_reward)
                        break
            mean_reward = np.mean(episode_rewards)
            print("Mean reward: " + str(mean_reward))

        else:
            print("You have set an invalid simulation_mode or some other settings in the config.py are wrong - aborting")

    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Cleaning up...")
        raise
    finally:
        try:
            env.close()
        except Exception as env_close_exc:
            print(f"Failed to close environment cleanly: {env_close_exc}")
