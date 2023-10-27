# modified from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
# to include simulating disturbed models and an analytical disturbance observer
# the environment is also changed to be episodic with maximum allowed time steps
# the action space is also changed to be continuous Box space

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than max_ts.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, max_ts, disturbance=None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        self.ts = 0
        self.max_ts = max_ts
        self.disturbance = disturbance

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        # self.action_space = spaces.Discrete(2)
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.gravity_n = self.gravity
        self.masscart_n = self.masscart
        self.masspole_n = self.masspole
        self.total_mass_n = (self.masspole_n + self.masscart_n)
        self.length_n = self.length
        self.polemass_length_n = (self.masspole_n * self.length_n)
        self.force_mag_n = self.force_mag

        self.obs = None
        self.obs_prev = None
        self.act_prev = None

        self.action_dist = 0.0
        self.obs_noise = np.zeros(4)
        if self.disturbance is not None:
            self._apply_disturbance()

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

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
                    if dist_var == 'cart_mass':
                        self.masscart *= dist_val
                        self.total_mass = (self.masspole + self.masscart)
                    elif dist_var == 'pole_mass':
                        self.masspole *= dist_val
                        self.total_mass = (self.masspole + self.masscart)
                        self.polemass_length = (self.masspole * self.length)
                    elif dist_var == 'pole_length':
                        self.length *= dist_val
                        self.polemass_length = (self.masspole * self.length)
                    elif dist_var == 'gravity':
                        self.gravity *= dist_val
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
                        self.obs_noise[obs_ind] =\
                            (dist_val - 1.0) * self.observation_space.high[obs_ind - (obs_ind % 2)]  # should not be inf
                    else:
                        raise Exception('* unknown observation!')
                else:
                    raise Exception('* unknown disturbance type!')

    def step(self, action):
        self.ts += 1

        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        x, x_dot, theta, theta_dot = self.state
        self.act_prev = action
        action_dist = self.np_random.uniform(low=0., high=np.abs(self.action_dist)) * np.sign(self.action_dist)
        action = np.clip(action[0] + action_dist, -1., 1.)
        force = self.force_mag * action
        # force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) /\
                   (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        if self.ts > self.max_ts:
            reward = 100.0
            done = True

        return self._get_obs(), reward, done, {}

    def reset(self):
        self.ts = 0
        self.obs = None
        self.act_prev = None
        high = np.array([self.x_threshold / 2, self.x_threshold / 2,
                         self.theta_threshold_radians / 2, self.theta_threshold_radians / 2])
        self.state = self.np_random.uniform(low=-high, high=high, size=(4,))
        self.steps_beyond_done = None
        return self._get_obs()

    def _get_obs(self):
        self.obs_prev = self.obs
        self.obs = np.array(self.state)
        obs_noise = self.np_random.uniform(low=0., high=np.abs(self.obs_noise)) * np.sign(self.obs_noise)
        self.obs += obs_noise
        return self.obs

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def estimate_disturbance(self):
        if self.act_prev is None:
            return 0.

        # gravity = self.gravity_n
        # masscart = self.masscart_n
        # masspole = self.masspole_n
        total_mass = self.total_mass_n
        # length = self.length_n
        polemass_length = self.polemass_length_n
        force_mag = self.force_mag_n
        tau = self.tau

        x, x_dot, theta, theta_dot = self.obs
        x_prev, x_dot_prev, theta_prev, theta_dot_prev = self.obs_prev

        temp_prev = (x_dot - x_dot_prev) / tau +\
                    (polemass_length * math.cos(theta_prev) * ((theta_dot - theta_dot_prev) / tau)) / total_mass
        force_prev = total_mass * temp_prev - polemass_length * theta_dot_prev ** 2 * math.sin(theta_prev)
        act_prev_est = force_prev / force_mag

        return act_prev_est - self.act_prev

    def random_action(self):
        return self.action_space.sample()


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    env = CartPoleEnv(200)
    check_env(env)
