from config import Config
from iiwa_sample_joint_vel_env import IiwaJointVelEnv
from simulator_vec_env import SimulatorVecEnv
from stable_baselines3 import PPO

import numpy as np

def get_env(config_dict, env_dict):
    env_key = config_dict['env_key']

    def create_env(id=0):
        # 'dart' should be always included in the env_key when there is a need to use a dart based environment
        if env_key == 'iiwa_joint_vel':
            env = IiwaJointVelEnv(max_ts=250, id=id, config=config_dict)
    num_envs = config_dict['num_envs']
    env = [create_env for i in range(num_envs)]
    env = SimulatorVecEnv(env, config_dict)
    return env


if __name__ == "__main__":

    config_dict = Config.get_config_dict()
    env_dict = Config.get_dart_env_dict()
    env = get_env(config_dict, env_dict)

    # check policy training and inference for joint velocity environment, use with

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save("ppo_trained")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_trained")

    obs = env.reset()
    for x in range(1000):
        action, _states = model.predict(obs, True)
        obs, rewards, dones, info = env.step(action)