import os
from pathlib import Path

import numpy as np
import pandas as pd

# Import envs #
from simulator_vec_env import SimulatorVecEnv
from envs.iiwa_sample_joint_vel_env import IiwaJointVelEnv
from envs.iiwa_sample_env import IiwaSampleEnv

# Planar grasping
from envs_advanced.iiwa_numerical_planar_grasping_env import IiwaNumericalPlanarGraspingEnv
from envs_advanced.iiwa_end_to_end_planar_grasping_env import IiwaEndToEndPlanarGraspingEnv
from envs_advanced.iiwa_ruckig_planar_grasping_env import IiwaRuckigPlanarGraspingEnv

# Monitor
from stable_baselines3.common.vec_env import VecMonitor
from torch.utils.tensorboard import SummaryWriter

# Models #
from stable_baselines3 import PPO
from models.ruckig_planar_model import RuckigPlanarModel

# Set-up envs
from utils_advanced.helpers import get_env

def evaluation_mode(config_dict, dart_env_dict, gym_env_dict, hyper_dict, goal_dict, reward_dict, manual_actions_dict=None, randomization_dict=None, randomBoxesGenerator=None):
    """
        evaluate a single checkpoint or a whole directory that may include many runs and many checkpoints per run. Save results in .csv file

        each environment will call a dedicated evaluation function - adapt and extend to your task

        :param config_dict:          configuration dictionary
        :param dart_env_dict:        dart dictionary
        :param gym_env_dict:         env dictionary
        :param hyper_dict:           hyperparameters dict - models related
        :param goal_dict:            goal dict - target, or green box
        :param reard_dict:           reward dict - reward terms values and ranges
        :param manual_actions_dict:  manual action dict
        :param randomization_dict:   randomization dict
        :param randomBoxesGenerator: boxes generator to spawn boxes for planar grasping
    """

    env_key = config_dict['env_key']

    # Build env #
    env = get_env(config_dict, dart_env_dict, gym_env_dict, config_dict["log_dir"], reward_dict, goal_dict, manual_actions_dict, randomization_dict, randomBoxesGenerator)

    print("Using for evaluation the env: " + config_dict["env_key"])

    if env_key.find("planar") != -1: # Planar envs
        _evaluate_planar_grasping(env, config_dict, dart_env_dict, hyper_dict, randomBoxesGenerator)
    elif env_key == "iiwa_sample_dart_unity_env" or env_key == "iiwa_joint_vel":
        _evaluate_base_env(env, config_dict, dart_env_dict)
    else:
        raise Exception("This type of env has no available evaluation method. Create one in the evaluate.py first - aborting")

def _evaluate_base_env(env, config_dict, dart_env_dict):
    """
        evaluate iiwa_sample_joint_vel_env, or iiwa_joint_vel envs

        :param env:           gym env
        :param config_dict:   configuration dict
        :param dart_env_dict: dart dictionary
    """

    if(config_dict["simulation_mode"] == "evaluate"):
        run_name = config_dict["model_evaluation_type"].split("/")
        run_name = run_name[-2] + "_" + run_name[-1]                 # e.g. run_0/model_3125_0

        print("===================================================")
        print("(RL) Model evaluation: " + str(run_name))
        print("===================================================")

        # Load trained agent #
        model = PPO.load(os.path.join(config_dict["model_evaluation_type"]))
        model.policy.set_training_mode(False)

        obs = env.reset()
        for x in range(1000):                                        # Run some steps for each env 
            action, _states = model.predict(obs, deterministic=True) # Important: set deterministic to True to use the best learned policy (no exploration)

            obs, rewards, dones, info = env.step(action)

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

def _evaluate_planar_grasping(env, config_dict, dart_env_dict, hyper_dict, randomBoxesGenerator):
    """
        evaluate a saved model or many runs and many checkpoints per run. Print some stats and save pandas dfs (.csv) in an evaluation folder

        :param env:                  gym env
        :param config_dict:          configuration dictionary
        :param dart_env_dict:        dart dictionary
        :param hyper_dict:           hyperparameters dict. Models related: RUCKIG, and for PPO model, a checkpoint will be loaded (with saved hyperparams)
        :param randomBoxesGenerator: spawn boxes for evaluating planar grasping agents
    """

    ########################################################################
    # Df to save evaluation stats                                          #
    # Columns:                                                             #
    #        run: 0, 1, 2, .. -> folder with checkpoints                   #
    #        timestep: 0, ..., total_timestep -> many checkpoints          #
    # Note: evaluate the agent in the same boxes for each saved checkpoint #
    # e.g. run_0/model_3125_0, run_0/model_6250_1, etc                     #
    ########################################################################

    # Save all runs and all checkpoints per run #
    df = pd.DataFrame(columns=['evaluation_name', 'run', 'timestep', 'reward', 'success_ratio'])

    # Keep a separate file for the best model #
    df_best = pd.DataFrame(columns=['evaluation_name',
                                    'best_succ_ratio_model', 'best_succ_ratio', 'best_succ_ratio_reward', 
                                    'best_reward_model', 'best_reward', 'best_reward_succ_ratio'])

    # Set dataset to test mode #
    randomBoxesGenerator.mode = "test"

    ############################################
    # Keep best stats from all runs and models #
    ############################################
    best_succ_ratio = 0
    best_succ_ratio_model = None
    best_succ_ratio_reward = -np.inf # For best success ratio model save its corresponding reward value

    best_reward = -np.inf
    best_reward_model = None
    best_reward_succ_ratio = 0       # For best reward model save its corresponding success ratio value

    log_dir = config_dict["log_dir"]

    # Evaluate a single model #
    if (config_dict["model_evaluation_type"] != "all" or config_dict["env_key"] == 'iiwa_ruckig_planar_grasping_dart_unity_env'):

        # Create a dir to save the evaluation pandas dfs #
        Path(log_dir + "evaluation_single_dfs/").mkdir(parents=True, exist_ok=True)

        # Load model #
        if(config_dict["model"] == "PPO"):
            model = PPO.load(config_dict["model_evaluation_type"])
            model.policy.set_training_mode(False)

            run_name = config_dict["model_evaluation_type"].split("/")
            run_name = run_name[-2] + "_" + run_name[-1] # e.g. run_0/model_3125_0

        elif(config_dict["model"] == "RUCKIG"):
            model = RuckigPlanarModel(env, control_cycle=dart_env_dict["control_cycle"], hyper_dict=hyper_dict)
            run_name = "ruckig_model"

        print("===================================================")
        print("Model evaluation: " + str(run_name))
        print("===================================================")

        # Evaluate single model #
        mean_r, succ_ratio = _evaluate_planar_grapsing_single_model(env=env, model=model, log_dir=log_dir, run_name=run_name, model_id=0, 
                                                                    config_dict=config_dict, dart_env_dict=dart_env_dict, episodes_test=randomBoxesGenerator.val_size)

        # One model -> equivalent best stats #
        best_reward = mean_r
        best_reward_model = run_name
        best_reward_succ_ratio = succ_ratio

        best_succ_ratio = succ_ratio
        best_succ_ratio_model = run_name
        best_succ_ratio_reward = mean_r
        ######################################

        # Save dfs #
        df.loc[0] = [config_dict["evaluation_name"], 0, 0, mean_r, succ_ratio]
        df.to_csv(log_dir + "evaluation_single_dfs/" + "df_evaluation_" + run_name + ".csv", index=False)

        df_best.loc[0] = [config_dict["evaluation_name"],
                          best_succ_ratio_model, best_succ_ratio,  best_succ_ratio_reward, 
                          best_reward_model, best_reward, best_reward_succ_ratio
                         ]

        df_best.to_csv(log_dir + "evaluation_single_dfs/" + "df_evaluation_best_" + run_name + ".csv", index=False)

    else: # Evaluate many runs and their saved checkpoints 

        # Create a dir to save the evaluation pandas dfs #
        Path(log_dir + "evaluation_all_dfs/").mkdir(parents=True, exist_ok=True)

        # Scan logs folder and find how many runs exist #
        # logs/ -> run_0/, run_1/                       #
        log_dirs = [filename for filename in os.listdir(log_dir[:-1]) if filename.startswith("run_")]
        log_dirs = sorted(log_dirs, key=lambda x: int(x.split("_")[1])) 

        model_i = 0 # For run_0, run_1
        j = 0       # For pandas rows -> one df for all runs 

        for log_dir_name in log_dirs: # run_i 
            print("=================================")
            print("Model evaluation: " + str(model_i + 1))
            print("=================================")

            # Scan the checkpoints of this runs and sort them by the time step #
            models_names = [
                filename for filename in os.listdir(log_dir + log_dir_name + "/")
                if filename.startswith("model") and filename.endswith(".zip")
            ]

            models_names = sorted(models_names, key=lambda x: int(x.split("_")[1]))

            # First time step 0 - baseline for all runs - set the initial reward value from the config_advanced.py - adapt if the reward denifition or task changes #
            df.loc[j] = [config_dict["evaluation_name"], model_i + 1, 0, config_dict["reward_baseline"], 0.0]
            j += 1

            for i, m_name in enumerate(models_names): # Load and evaluate the current checkpoint of the run_i 
                print("Checkpoint number: " + str(i + 1))

                # Load saved model #
                model = PPO.load(log_dir + log_dir_name + "/" + m_name[:-4]) # Remove .zip
                model.policy.set_training_mode(False)

                # Evaluate model #
                mean_r, succ_ratio = _evaluate_planar_grapsing_single_model(env=env, model=model, log_dir=log_dir, run_name=log_dir_name, 
                                                                            model_id=m_name.split("_")[1], config_dict=config_dict, 
                                                                            dart_env_dict=dart_env_dict, episodes_test=randomBoxesGenerator.val_size)
                # Update best stats #
                if (mean_r > best_reward): # reward-based
                    best_reward = mean_r
                    best_reward_model = log_dir + log_dir_name + "/" + m_name # which model had the best reward
                    best_reward_succ_ratio = succ_ratio

                if (succ_ratio > best_succ_ratio): # success ratio-based
                    best_succ_ratio = succ_ratio
                    best_succ_ratio_model = log_dir + log_dir_name + "/" + m_name
                    best_succ_ratio_reward = mean_r

                # Save results for the current checkpoint #
                df.loc[j] = [config_dict["evaluation_name"], model_i + 1, m_name.split("_")[1], mean_r, succ_ratio]
                j += 1

            # All checkpoints were evaluated for the run_i. Go to the next run
            model_i += 1

        # Save pandas df of all checkpoints and runs #
        df.to_csv(log_dir + "evaluation_all_dfs/" + "df_evaluation_all" + ".csv", index=False)

        # Save pandas df of the best stats #
        df_best.loc[0] = [
            config_dict["evaluation_name"],
            best_succ_ratio_model, best_succ_ratio,  best_succ_ratio_reward, 
            best_reward_model, best_reward,  best_reward_succ_ratio
        ]
        df_best.to_csv(log_dir + "evaluation_all_dfs/" + "df_evaluation_best" + ".csv", index=False)

    # Print the best stats in console #
    print("best_reward: " + str(best_reward))
    print("best_reward_model: " + str(best_reward_model))
    print("best_reward_succ_ratio: " + str(best_reward_succ_ratio))

    print("best_succ_ratio: " + str(best_succ_ratio))
    print("best_succ_ratio_model: " + str(best_succ_ratio_model))
    print("best_succ_ratio_reward: " + str(best_succ_ratio_reward))

    # Revert dataset to train mode #
    randomBoxesGenerator.mode = "train"

def _evaluate_planar_grapsing_single_model(env, model, log_dir, run_name, model_id, config_dict, dart_env_dict, episodes_test):
    """
        evaluate a saved model - grasping planar environment. It writes metrics in a tensorboard summary file

        :param env:           gym env
        :param model:         saved model to evaluate
        :param log_dir:       log directory to save the tensorboard summary file
        :param run_name:      name of the run e.g. run_0
        :param model_id:      checkpoint id e.g. model_200 -> 200
        :param config_dict:   configuration dict
        :param dart_env_dict: dart dict
        :param episodes_test: how many evaluation targets (boxes) to test

        :return: mean_r, succ_ratio -> mean reward and success ratio for all the episodes (episodes_test)
    """

    # Save results to tensorboard summary file #
    writer = SummaryWriter(str(log_dir) + "/evaluation_tb_" + run_name)

    mean_reward = 0
    success_count = 0  # #successfull episodes
    total_episodes = 0 # total episodes 

    num_envs = config_dict["num_envs"]

    # Save total reward per episode per each env.  These detailed stats are not printed by default (see below) #
    total_rewards = np.zeros((num_envs, 5000))  # Assume max 5000 episodes played
    total_success = np.zeros((num_envs, 5000))  # Assume max 5000 episodes played
    num_episode = 0                             # All envs terminate at the same time step

    ##############################################################################################################
    # Calculate how many steps to play for the required "episodes_test" given the number of the envs,            #
    # and the 'max_time_step defined' per each episode                                                           #
    # Imporant: make sure your values result in a perfect division, else some target boxes will not be evaluated #
    ##############################################################################################################
    time_steps = int(episodes_test * dart_env_dict["max_time_step"] / num_envs)

    obs = env.reset()
    for _ in range(time_steps):

        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        total_rewards[:, num_episode] += rewards

        # Render #
        if dart_env_dict['enable_dart_viewer']:
            env.render()

        for i in range(num_envs):                                # Check if done
            if dones[i]:
                if(i == 0):                                      # Assume envs terminate at the same time step
                    num_episode += 1

                mean_reward += total_rewards[i][num_episode - 1] # Append total reward of this env in the cumulative sum, 'mean_rewards'

                if (info[i]["success"] == True):                 # Successful episode 
                    success_count += 1
                    total_success[i][num_episode - 1] = 1
                else:
                    total_success[i][num_episode - 1] = 0

                total_episodes += 1

    # Fix metrics #
    succ_ratio = (success_count * 100) / total_episodes
    mean_r = mean_reward / float(total_episodes)

    # Log metrics in tb summary file #
    writer.add_scalar('success_ratio', succ_ratio, model_id)
    writer.add_scalar('total_test_episodes', total_episodes, model_id)
    writer.add_scalar('mean_reward', mean_r, model_id)
    writer.flush()

    # Print some stats and metrics in the console #
    print("End of model evaluation with name: " + run_name + " and id: " + str(model_id))
    print("Number of targets that were tested: " + str(total_episodes))
    print("Success ratio: " + str(succ_ratio))
    print("Mean reward: " + str(mean_r))

    # Uncomment these lines for more detailed results (per episode) #
    #print("Total reward for 50 episodes for every env")
    #print(total_rewards[:, 0:50]) 
    #print("Success flag for 50 episodes for every env")
    #print(total_success[:, 0:50]) 

    print("")
    writer.close()

    return [mean_r, succ_ratio]
