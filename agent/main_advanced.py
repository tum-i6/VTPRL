import os
from pathlib import Path
import numpy as np

from config_advanced import ConfigAdvanced

# Model 
from stable_baselines3 import PPO

# Policy networks
from utils.policy_networks import PolicyNetworkVanillaReLU

from utils.helpers import set_seeds

# Set-up envs
from utils_advanced.helpers import get_env

from utils_advanced.boxes_generator import RandomBoxesGenerator
from utils_advanced.monitoring_agent import SaveOnBestTrainingRewardCallback

# Evaluate methods
from utils_advanced.evaluate import evaluation_mode

def train_mode(ppo_seed, agent, root, sim, gym, manip_env, observation, hyper_dict, goal_dict, reward_dict, manual_actions_dict=None, randomBoxesGenerator=None):
    """
        train the RL agent. Play some episodes, record metrics, and save checkpoints

        :param ppo_seed:             ppo seed
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

    # Check if other runs already exist in the log dir #
    run_num = len([file for file in os.listdir(agent["log_dir"]) if file.startswith("run_")]) # How many folders already exist
    agent["log_dir"] += "run_" + str(run_num) + "/"

    # Create new folder for logs #
    Path(agent["log_dir"]).mkdir(parents=True, exist_ok=True)

    # Build env #
    env = get_env(agent, root, sim, gym, manip_env, observation, agent["log_dir"], reward_dict,
                  goal_dict, manual_actions_dict, randomBoxesGenerator)

    print("Using for training the env: " + gym["env_key"]) 

    # Callback - save checkpoints and log best reward #
    callback_save_on_best = SaveOnBestTrainingRewardCallback(agent["check_freq"], save_model_freq=agent["save_model_freq"], 
                                                            log_dir=agent["log_dir"], total_timesteps=agent["total_timesteps"], 
                                                            num_envs=gym["num_envs"], best_mean=agent["best_mean_reward"]) 

    # Select policy network #
    if hyper_dict["policy_network"] is None:
        policy_kwargs = None
        print("Using the default SB3 policy")
    elif hyper_dict["policy_network"] == "PolicyNetworkVanillaReLU":
        policy_kwargs = PolicyNetworkVanillaReLU()
        print("Using custom policy with kwargs:\n")
        print(policy_kwargs)
    else:
        policy_kwargs = None
        print("This policy network is not supported. Using the default SB3 policy")

    # Define the RL model and its hyperparameters #
    model = PPO(policy=hyper_dict["policy"], env=env, seed=ppo_seed, learning_rate=hyper_dict["learning_rate"], ent_coef=hyper_dict["ent_coef"],
                vf_coef=hyper_dict["vf_coef"], max_grad_norm=hyper_dict["max_grad_norm"], gae_lambda=hyper_dict["gae_lambda"], n_epochs=hyper_dict["n_epochs"],
                n_steps=hyper_dict["n_steps"], batch_size=hyper_dict["batch_size"], gamma=hyper_dict["gamma"], clip_range=hyper_dict["clip_range"],
                tensorboard_log=agent["log_dir"], policy_kwargs=policy_kwargs, verbose=1)

    # Play some episodes #
    model.learn(total_timesteps=agent["total_timesteps"], callback=callback_save_on_best, reset_num_timesteps=True, tb_log_name=agent["tb_log_name"], log_interval=2)

    print("Training ended")
    # Save the last model #
    # model.save(os.path.join(config_dict["log_dir"], "ppo_trained"))

    del model  # remove


if __name__ == "__main__":

    main_config = ConfigAdvanced()

    # Structured config handles #
    agent = main_config.agent_dict
    root = main_config.root_dict
    sim = main_config.simulation_dict
    gym = main_config.gym_environment_dict
    manip_env = main_config.manipulator_environment_dict
    observation = main_config.observation_dict

    # Reward/goal/manual dicts (advanced)
    mg = gym['manipulator_gym_environment']
    reward_dict = mg['reward']
    goal_dict = mg['goal']
    manual_actions_dict = agent['manual_actions']
    hyper_dict = agent['hyper_dict']

    # Set global seeds and get a PPO seed #
    ppo_seed = set_seeds(sim["random_seed"]) 

    # Set-up the random boxes generator - for planar envs #
    if(goal_dict["goal_type"] == "box" and gym["env_key"].find("planar") != -1):
        randomBoxesGenerator = RandomBoxesGenerator(box_mode=goal_dict["box_mode"], box_samples=goal_dict["box_samples"], box_split=goal_dict["box_split"], box_save_val=goal_dict["box_save_val"], box_load_val=goal_dict["box_load_val"],
                                                    box_radius_val=goal_dict["box_radius_val"], box_min_distance_base=goal_dict["box_min_distance_base"],  box_max_distance_base=goal_dict["box_max_distance_base"], box_folder=goal_dict["box_folder"],
                                                    box_x_min=goal_dict["box_x_min"], box_x_max=goal_dict["box_x_max"], box_x_active=goal_dict["box_x_active"], box_z_min=goal_dict["box_z_min"], box_z_max=goal_dict["box_z_max"],
                                                    box_z_active=goal_dict["box_z_active"], box_ry_min=goal_dict["box_ry_min"], box_ry_max=goal_dict["box_ry_max"], box_ry_active=goal_dict["box_ry_active"], box_debug=goal_dict["box_debug"]
                                                    )
    else:
        randomBoxesGenerator = None

    # Train the RL-agent #
    if(agent['simulation_mode'] == 'train'):
        train_mode(ppo_seed, agent, root, sim, gym, manip_env, observation, hyper_dict, goal_dict, reward_dict,
                   manual_actions_dict, randomBoxesGenerator)

    # Evaluate agents - model-based or rl-based #
    elif(agent['simulation_mode'] == 'evaluate' or agent["simulation_mode"] == 'evaluate_model_based'):
        evaluation_mode(agent, root, sim, gym, manip_env, observation, hyper_dict, goal_dict, reward_dict,
                        manual_actions_dict, randomBoxesGenerator)
    else:
        print("You have set an invalid simulation_mode or some other settings in the config.py are wrong")
