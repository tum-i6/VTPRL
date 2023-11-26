import numpy as np

from envs_other.cartpole import CartPoleEnv
from envs_other.nlinks_box2d import NLinksBox2DEnv
from envs_other.pendulum import PendulumEnv

from stable_baselines3.common.env_checker import check_env

if __name__ == '__main__':
    """
        Example method to run gym base environments
    """

    env_mode = "cartpole" # "nlinks_box2d", pendulum

    if(env_mode == "cartpole"):
        env = CartPoleEnv(200)
        check_env(env)

        while True:
            env.render()

            rnd_act = [np.random.uniform(-1, 1)]

            s, _, d, _ = env.step(action=rnd_act)
            if d:
                env.reset()

    elif(env_mode == "nlinks_box2d"):
        num_links = 10
        nlinks = NLinksBox2DEnv(max_ts=200, n_links=num_links)

        check_env(nlinks)

        # some silent initialization steps
        # for _ in range(100):
        #     nlinks.render()
        #     zero_act = np.zeros(num_links)
        #     nlinks.step(action=zero_act)

        while True:
            nlinks.render()
            # print("anchor", nlinks.anchor.position)
            # print("end_eff", nlinks.end_effector.position)
            # rnd_act = np.array([0.3, -0.3])
            # rnd_act = np.array([np.random.rand() * 2 - 1, np.random.rand() * 2 - 1])
            rnd_act = []
            for i in range(num_links):
                rnd_act.append(np.random.rand() * 2 - 1)

            s, _, d, _ = nlinks.step(action=rnd_act)
            # print("tx: {0:2.1f}; ty: {1:2.1f}".format(s[0], s[1]))
            # print("eex: {0:2.1f}; eey: {1:2.1f}".format(s[2], s[3]))
            if d:
                nlinks.reset()

    elif(env_mode == "pendulum"):
        env = PendulumEnv(200)
        check_env(env)
