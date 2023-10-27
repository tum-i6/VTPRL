import os

from config import Config

import numpy as np
from pathlib import Path
import pandas as pd

# Import envs #
from simulator_vec_env import SimulatorVecEnv
from envs.iiwa_sample_joint_vel_env import IiwaJointVelEnv
from envs.iiwa_sample_env import IiwaSampleEnv

# Monitor envs 
from stable_baselines3.common.vec_env import VecMonitor

# Models #
from stable_baselines3 import PPO

from utils.helpers import set_seeds

def get_env(config_dict, dart_env_dict, reward_dict, log_dir):
    """
        Set-up the env according to the input dictionary settings
    """

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
                                goal_type="target", joints_safety_limit=config_dict['joints_safety_limit'],
                                max_joint_vel=config_dict['max_joint_vel'], max_ee_cart_vel=config_dict['max_ee_cart_vel'],
                                max_ee_cart_acc=config_dict['max_ee_cart_acc'], max_ee_rot_vel=config_dict['max_ee_rot_vel'],
                                max_ee_rot_acc=config_dict['max_ee_rot_acc'], random_initial_joint_positions=config_dict['random_initial_joint_positions'],
                                initial_positions=config_dict['initial_positions'], robotic_tool=config_dict["robotic_tool"],
                                env_id=id)

        # Set env seed #
        env.seed((id * 150) + (id + 11))

        return env

    num_envs = config_dict['num_envs']
    env = [create_env for i in range(num_envs)]
    env = SimulatorVecEnv(env, config_dict, manual_actions_dict=None, reward_dict=reward_dict) # Set vectorized env
    env = VecMonitor(env, log_dir, info_keywords=("success",))                                 # Monitor envs 

    return env

if __name__ == "__main__":

    main_config = Config()

    # Parse configs #
    config_dict = main_config.get_config_dict()
    dart_env_dict = main_config.get_dart_env_dict()
    reward_dict = main_config.get_reward_dict()

    # Create new folder if not exists for logging #
    Path(config_dict["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Build env #
    env = get_env(config_dict, dart_env_dict, reward_dict, config_dict["log_dir"])

    # Train the agent #
    if(config_dict['simulation_mode'] == 'train'):

        # Set global seeds and get a PPO seed #
        ppo_seed = set_seeds(config_dict["seed"])

        # Define the model and its hyperparameters #
        model = PPO(policy="MlpPolicy", env=env, seed=ppo_seed, tensorboard_log=config_dict["log_dir"], verbose=1)

        # Play some episodes                                                      #
        # If you retrain the model, you may need to set reset_num_timesteps=False #
        model.learn(total_timesteps=dart_env_dict["total_timesteps"], reset_num_timesteps=True, tb_log_name=config_dict["tb_log_name"], log_interval=2)

        print("Training ended. Saving a checkpoint at: " + config_dict["log_dir"])

        # Save the last model #
        model.save(os.path.join(config_dict["log_dir"], "ppo_trained"))

        del model  # remove

    elif(config_dict['simulation_mode'] == 'evaluate'):
        print("===================================================")
        print("RL-based evaluation")
        print("===================================================")

        # Load trained agent #
        model = PPO.load(os.path.join(config_dict["log_dir"], "ppo_trained"))
        model.policy.set_training_mode(False)

        obs = env.reset()
        for x in range(1000):                                                # Run some steps for each env 
            action, _states = model.predict(obs, deterministic=True)         # Important: set deterministic to True to use the best learned policy (no exploration)

            obs, rewards, dones, info = env.step(action)

            # Render #
            if dart_env_dict['enable_dart_viewer'] and config_dict['env_key'] != 'iiwa_joint_vel':
                env.render()

    elif(config_dict['simulation_mode'] == 'evaluate_model_based' and config_dict['env_key'] != 'iiwa_joint_vel'):
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
                if dart_env_dict['use_inverse_kinematics']:                                                   # Generate an action for the current observation using a P-controller
                    action = np.reshape(env.env_method('action_by_p_control', control_kp, 2.0 * control_kp),
                                        (config_dict['num_envs'], env.action_space.shape[0]))
                else:                                                                                         # Random action
                    action = np.reshape(env.env_method('random_action'),
                                        (config_dict['num_envs'], env.action_space.shape[0]))

                obs, rewards, dones, info = env.step(action)                                                  # Play this action
                cum_reward += rewards

                # Render #
                if dart_env_dict['enable_dart_viewer']:
                    env.render()

                if dones.any():
                    episode_rewards.append(cum_reward)
                    break

        mean_reward = np.mean(episode_rewards)

        print("Mean reward: " + str(mean_reward))
    else:
        print("You have set an invalid simulation_mode or some other settings in the config.py are wrong - aborting")
