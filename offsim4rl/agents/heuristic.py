import gym
import numpy as np
from offsim4rl.agents.agent import Agent
from offsim4rl.data import ProbDistribution

class CartPolePDController(Agent):
    '''
        Implementation based on https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f 
    '''
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

    def _theta_policy(self, obs):
        theta = obs[2]
        return 0 if theta < 0 else 1

    def _omega_policy(self, obs):
        w = obs[3]
        return 0 if w < 0 else 1

    def _theta_omega_policy(self, obs):
        theta, w = obs[2:4]
        if abs(theta) < 0.03:
            return 0 if w < 0 else 1
        else:
            return 0 if theta < 0 else 1

    def _get_act_distribution(self, obs):
        pi = np.array(self._prob_dist)
        act = 0
        if self.mode == 'theta':
            act = self._theta_policy(obs)
        elif self.mode == 'omega':
            act = self._omega_policy(obs)
        elif self.mode == 'theta-omega':
            act = self._theta_omega_policy(obs)

        pi[act] = 1 - self.eps
        pi[1 - act] = self.eps
        return pi
