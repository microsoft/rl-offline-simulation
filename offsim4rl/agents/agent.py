import abc

import gym


class Agent(abc.ABC):
    @abc.abstractproperty
    def action_dist_type(self):
        pass

    @abc.abstractmethod
    def begin_episode(self, observation):
        pass

    @abc.abstractmethod
    def commit_action(self, action):
        pass

    @abc.abstractmethod
    def step(self, reward, next_observation):
        pass

    @abc.abstractmethod
    def end_episode(self, reward):
        pass


class AdaptiveAgent(Agent):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model, model_kwargs=dict(), seed=0):
        self.action_space = action_space
        self.observation_space = observation_space
        self.model = model(observation_space, action_space, **model_kwargs)
        self.model_kwargs = model_kwargs
        self.seed = seed

    def begin_episode(self, observation):
        pass

    def commit_action(self, action):
        pass

    def end_episode(self, reward):
        pass

    def step(self, reward, next_observation):
        pass

    def adapt(self, transitions=None):
        pass
