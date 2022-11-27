from config import Config
from iiwa_sample_joint_vel_env import IiwaJointVelEnv
from iiwa_sample_env import IiwaSampleEnv
from simulator_vec_env import SimulatorVecEnv
from stable_baselines3 import PPO
import numpy as np


def get_env(config_dict, env_dict):
    env_key = config_dict['env_key']

    def create_env(id=0):
        # 'dart' should be always included in the env_key when there is a need to use a dart based environment
        if env_key == 'iiwa_joint_vel':
            env = IiwaJointVelEnv(max_ts=250, id=id, config=config_dict)
        elif env_key == 'iiwa_sample_dart_unity_env':
            env = IiwaSampleEnv(max_ts=env_dict['max_time_step'],
                                orientation_control=env_dict['orientation_control'],
                                use_ik=env_dict['use_inverse_kinematics'],
                                ik_by_sns=env_dict['linear_motion_conservation'],
                                enable_render=env_dict['enable_dart_viewer'],
                                state_type=config_dict['state'],
                                env_id=id)
        return env

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


    # check model-based controllers, update environment key in config.py, comment lines 40-51 and uncomment below code
    # to test dart-enabled environments

    # control_kp = 1.0 / env.observation_space.high[0]
    #
    # if config_dict['seed'] is not None:
    #     env.seed(config_dict['seed'])
    #
    # obs = env.reset()
    # episode_rewards = []
    # for _ in range(100):
    #     cum_reward = 0
    #     while True:
    #         if env_dict['use_inverse_kinematics']:
    #             action = np.reshape(env.env_method('action_by_p_control', control_kp, 2.0 * control_kp),
    #                                 (config_dict['num_envs'], env.action_space.shape[0]))
    #         else:
    #             action = np.reshape(env.env_method('random_action'),
    #                                 (config_dict['num_envs'], env.action_space.shape[0]))
    #         obs, rewards, dones, info = env.step(action)
    #         cum_reward += rewards
    #         if env_dict['enable_dart_viewer']:
    #             env.render()
    #         if dones.any():
    #             episode_rewards.append(cum_reward)
    #             break
    # mean_reward = np.mean(episode_rewards)
    #
    # print(mean_reward)

