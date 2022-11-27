from config import Config
from dart_envs.iiwa_dart import IiwaDartEnv

import numpy as np


if __name__ == '__main__':
    """
    Example method to run the base standalone DART environment.
    This is useful to quickly develop and check different model-based control policies.
    """

    env_dict = Config.get_dart_env_dict()

    iiwa = IiwaDartEnv(max_ts=env_dict['max_time_step'], orientation_control=env_dict['orientation_control'],
                       use_ik=env_dict['use_inverse_kinematics'], ik_by_sns=env_dict['linear_motion_conservation'],
                       enable_render=env_dict['enable_dart_viewer'])

    # iiwa.set_target(iiwa.create_target())

    pd_control = True and iiwa.USE_IK
    control_kp = 1.0 / iiwa.observation_space.high[0]

    import time
    last_episode_time = time.time()
    iiwa.render()
    iiwa.reset()
    time.sleep(1)

    while True:
        if pd_control:
            action = iiwa.action_by_pd_control(control_kp, 3.0 * control_kp)
        else:
            action = []
            for i in range(iiwa.action_space_dimension):
                action.append(np.random.rand() * 2. - 1.)

        state, reward, done, info = iiwa.step(action=action)

        iiwa.render()

        if done:
            iiwa.reset()
            current_time = time.time()
            print(current_time - last_episode_time)
            last_episode_time = current_time
