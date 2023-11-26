# modified from https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# to include simulating disturbed models and an analytical disturbance observer
# the environment is also changed to be episodic with maximum allowed time steps

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class PendulumEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, max_ts, disturbance=None):

        self.max_speed = 8.
        self.max_torque = 20.
        self.dt = 0.05
        self.g = 9.8
        self.m = 1.
        self.l = 1.
        self.ts = 0
        self.max_ts = max_ts
        self.disturbance = disturbance
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.dt_n = self.dt
        self.g_n = self.g
        self.m_n = self.m
        self.l_n = self.l

        self.obs = None
        self.obs_prev = None
        self.act_prev = None

        self.action_dist = 0.0
        self.obs_noise = np.zeros(3)  # TODO: finding robust model against obs noise
        if self.disturbance is not None:
            self._apply_disturbance()

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_disturbance(self, disturbance):
        self.disturbance = disturbance
        self._apply_disturbance()
        self.reset()

    def _apply_disturbance(self):
        for dist_type, _dist in self.disturbance.items():
            for dist_var, dist_val in _dist.items():
                if dist_type == 'parametric_disturbance':
                    if dist_var == 'pendulum_mass':
                        self.m *= dist_val
                    elif dist_var == 'pendulum_length':
                        self.l *= dist_val
                    elif dist_var == 'gravity':
                        self.g *= dist_val
                    elif dist_var == 'time_step':
                        self.dt *= dist_val
                    else:
                        raise Exception('* unknown parameter!')

                elif dist_type == 'action_disturbance':
                    if dist_var == '0':
                        self.action_dist = (dist_val - 1.0) * self.action_space.high[0]
                    else:
                        raise Exception('* unknown action!')

                elif dist_type == 'observation_noise':
                    obs_ind = int(dist_var)
                    if obs_ind in range(self.observation_space.shape[0]):
                        self.obs_noise[obs_ind] = (dist_val - 1.0) * self.observation_space.high[obs_ind]
                    else:
                        raise Exception('* unknown observation!')
                else:
                    raise Exception('* unknown disturbance type!')
        # print('param dist: ', self.m, self.l, self.g, self.dt)
        # print('action dist: ', self.action_dist)
        # print('obs noise: ', self.obs_noise)

    def step(self, u):
        self.ts += 1
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u_r = u[0]
        action_dist = self.np_random.uniform(low=0., high=np.abs(self.action_dist)) * np.sign(self.action_dist)
        u = u_r + action_dist
        u = np.clip(u, -self.max_torque, self.max_torque)
        costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .01 * (u_r ** 2)

        self.last_u = u_r  # for rendering

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])
        self.act_prev = u_r

        done = bool(self.ts > self.max_ts)

        return self._get_obs(), -costs, done, {}

    def reset(self):
        self.ts = 0
        self.obs = None
        self.act_prev = None
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        self.obs_prev = self.obs
        self.obs = np.array([np.cos(theta), np.sin(theta), thetadot])
        obs_noise = self.np_random.uniform(low=0., high=np.abs(self.obs_noise)) * np.sign(self.obs_noise)
        self.obs += obs_noise
        return self.obs

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "misc/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def estimate_disturbance(self):
        if self.act_prev is None:
            return 0.

        g, m, l, dt = self.g_n, self.m_n, self.l_n, self.dt_n
        costh, sinth, thdot = self.obs  # th := theta
        costh_prev, sinth_prev, thdot_prev = self.obs_prev

        act_prev_est = (m * l ** 2) / 3. * ((thdot - thdot_prev) / dt - 3 * g / (2 * l) * sinth_prev)

        return act_prev_est - self.act_prev

    def random_action(self):
        return self.action_space.sample()


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = PendulumEnv(200)
    check_env(env)
