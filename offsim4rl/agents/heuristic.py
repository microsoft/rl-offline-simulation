import gym
import numpy as np
from offsim4rl.agents.agent import Agent
from offsim4rl.data import ProbDistribution

class CartPolePDController(Agent):
    THETA_THRESHOLD = 2 * np.pi / 180

    def __init__(self, action_space: gym.Space, mode='theta-omega', eps=0):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("action_space must be gym.spaces.Discrete")

        self._prob_dist = np.ones(action_space.n) / action_space.n

        if mode not in ['theta', 'omega', 'theta-omega']:
            raise ValueError("mode should be one of ['theta', 'omega', 'theta-omega']")
        self.mode = mode

        if not (0 <= eps <= 1):
            raise ValueError("eps should be 0 <= eps <= 1")
        self.eps = eps

    @property
    def action_dist_type(self):
        return ProbDistribution.Discrete

    def begin_episode(self, observation):
        return self._get_act_distribution(observation)

    def commit_action(self, action):
        pass

    def step(self, reward, observation):
        return self._get_act_distribution(observation)

    def end_episode(self, reward, truncated=False):
        pass

    def _get_act_distribution(self, obs):
        pi = np.array(self._prob_dist)
        act = 0
        theta, w = obs[2:4]
        if self.mode == 'theta':
            act = 0 if theta < 0 else 1
        elif self.mode == 'omega':
            act = 0 if w < 0 else 1
        elif self.mode == 'theta-omega':
            if abs(theta) < CartPolePDController.THETA_THRESHOLD:
                act = 0 if w < 0 else 1
            else:
                act = 0 if theta < 0 else 1

        pi[act] = 1 - self.eps / 2
        pi[1 - act] = self.eps / 2
        return pi
