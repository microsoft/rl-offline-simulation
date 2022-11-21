import gym
import numpy as np
from offsim4rl.agents.agent import Agent
from offsim4rl.data import ProbDistribution

class DiscreteRandom(Agent):
    def __init__(self, action_space: gym.Space):
        if not isinstance(action_space, gym.spaces.Discrete):
            raise ValueError("action_space must be gym.spaces.Discrete")
        self._prob_dist = np.ones(action_space.n) / action_space.n
    
    @property
    def action_dist_type(self):
        return ProbDistribution.Discrete

    def begin_episode(self, observation):
        return np.array(self._prob_dist)

    def commit_action(self, action):
        pass

    def step(self, reward, observation):
        return np.array(self._prob_dist)

    def end_episode(self, reward, truncated=False):
        pass
