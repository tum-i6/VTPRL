import numpy as np
import time

from config import Config

from envs_dart.iiwa_dart import IiwaDartEnv
from envs_dart.so100_dart import SO100DartEnv

if __name__ == '__main__':
    """
        Example method to run the base standalone DART environment.
        This is useful to quickly develop and check different model-based control policies.
    """
    env_dict = Config.get_dart_env_dict()
    config_dict = Config.get_config_dict()

    if config_dict['env_key'] == 'iiwa_sample_dart_unity_env':
        robot = IiwaDartEnv(max_ts=env_dict['max_time_step'], orientation_control=env_dict['orientation_control'],
                            use_ik=env_dict['use_inverse_kinematics'], ik_by_sns=env_dict['linear_motion_conservation'],
                            enable_render=env_dict['enable_dart_viewer'], task_monitor=env_dict['task_monitor'],
                            target_mode="random_joint_level", with_objects=env_dict['with_objects'])
    
    if config_dict['env_key'] == 'so100_sample_dart_unity_env':
        robot = SO100DartEnv(max_ts=env_dict['max_time_step'], orientation_control=env_dict['orientation_control'],
                             use_ik=env_dict['use_inverse_kinematics'], ik_by_sns=env_dict['linear_motion_conservation'],
                             enable_render=env_dict['enable_dart_viewer'], task_monitor=env_dict['task_monitor'],
                             target_mode="random_joint_level", with_objects=env_dict['with_objects'])
    
    else:
        raise Exception('* Undefined environment!')

    # robot.set_target(robot.create_target())

    pd_control = True and robot.dart_sim.use_ik
    control_kp = 1.0 / robot.observation_space.high[0]

    last_episode_time = time.time()

    robot.reset()
    if robot.dart_sim.enable_viewer:
        robot.render()
    time.sleep(1)

    while True:
        if pd_control:
            action = robot.action_by_pd_control(control_kp, 3.0 * control_kp)
        else:
            action = []
            for i in range(robot.action_space_dimension):
                action.append(np.random.rand() * 2. - 1.)

        state, reward, done, info = robot.step(action=action)

        if robot.dart_sim.enable_viewer:
            robot.render()

        if done:
            robot.reset()
            current_time = time.time()
            print(current_time - last_episode_time)
            last_episode_time = current_time
