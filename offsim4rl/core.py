import gym
import abc
from torch.distributions import Distribution
from typing import Tuple

class RevealedRandomnessEnv(gym.Env, abc.ABC):
    @abc.abstractmethod
    def step_dist(self, action_dist: Distribution) -> Tuple[object, object, float, bool, dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts a distribution over actions and returns a tuple (action, observation, reward, done, info).

        Args:
            action_dist (object): a distribution over actions, provided by the agent

        Returns:
            action (object): an action sampled from the provided distribution
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        action = action_dist.sample()
        next_obs, reward, done, info = self._env.step(action.numpy())
        return action, next_obs, reward, done, info


class ActionSamplingEnv(RevealedRandomnessEnv):
    def __init__(self, env: gym.Env):
        self._env = env

    def step_dist(self, action_dist):
        action = action_dist.sample()
        next_obs, reward, done, info = self._env.step(action.numpy())
        assert 'action' not in info, f'Wrapper environment already contains action in the info object - do you want to override it?'
        info['action'] = action
        return next_obs, reward, done, info

    # The rest of the properties and methods are simple wrappers:
    @property
    def metadata(self):
        return self._env.metadata
    
    @property
    def reward_range(self):
        return self._env.reward_range
    
    @property
    def spec(self):
        return self._env.spec

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space
    
    def step(self, action):
        return self._env.step(action)

    def reset(self):
        return self._env.reset()

    def render(self, mode='human'):
        self._env.render(mode=mode)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        # TODO: should we set torch.random seed, since we use action_dist.sample()?
        return self._env.seed(seed=seed)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    def __str__(self):
        return self._env.__str__()

    def __enter__(self):
        return self._env.__enter__()

    def __exit__(self, *args):
        return self._env.__exit__(*args)
