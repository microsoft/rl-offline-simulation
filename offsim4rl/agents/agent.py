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
    def step(self, prev_reward, observation):
        pass

    @abc.abstractmethod
    def end_episode(self, reward):
        pass
