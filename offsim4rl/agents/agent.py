import abc

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
    def step(self, reward, observation):
        pass

    @abc.abstractmethod
    def end_episode(self, reward):
        pass
