
import gym
import numpy as np
from copy import deepcopy

def make_normalized_env(env):
    """ crate a new environment class with actions normalized to [-1,1] 
    clip rewards to [-1,1]
    """
    act_space = env.action_space
    obs_space = env.observation_space
    env_type = type(env)

    class NormalizedEnv(env_type):

        def __init__(self):
            # transfer properties
            self.__dict__.update(env.__dict__)

            # Action space
            h = act_space.high
            l = act_space.low
            sc = h - l
            self.act_k = sc / 2.
            self.act_b = (h + l) / 2.

            # Check and assign transformed spaces
            self.observation_space = obs_space
            self.action_space = gym.spaces.Box(-np.ones_like(act_space.high),
                                               np.ones_like(act_space.high))

            def assertEqual(a, b): assert np.all(a == b), "{} != {}".format(a, b)
            assertEqual(self._process_action(self.action_space.low), act_space.low)
            assertEqual(self._process_action(self.action_space.high), act_space.high)

        def _process_reward(self, reward):
            return np.clip(reward, -1., 1.)

        def _process_action(self, action):
            return self.act_k * action + self.act_b

        def reset(self):
            observation = env_type.reset(self)
            return deepcopy(observation)

        def step(self, action):
            action = self._process_action(action)
            observation2, r, done, info = env_type.step(self, action)
            reward = self._process_reward(r)

            return deepcopy(observation2), reward, done, info

        def render(self, **kwargs):
            env_type.render(self, **kwargs)

    nenv = NormalizedEnv()

    return nenv