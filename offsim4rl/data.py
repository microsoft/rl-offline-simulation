import abc
import collections.abc
from typing import Iterator, Dict, Union
import gym
import numpy as np
import enum

import numpy as np
from tqdm import tqdm
import h5py

from offsim4rl.core import ActionSamplingEnv

class ProbDistribution(enum.Enum):
    
    """ No probabilities are provided. """
    NoProbability = 0

    """ Only probability of the logged action (single float). """
    # TODO: is this needed?
    LoggedActionOnly = 1

    """ Probability Mass Function, mapping each discrete action to a probability (a vector of floats). """
    Discrete = 2

    # TODO: Normal, Multivariate, etc.


class OfflineDataset(collections.abc.Iterable):
    @abc.abstractproperty
    def observation_space(self) -> gym.Space:
        pass

    @abc.abstractproperty
    def action_space(self) -> gym.Space:
        pass

    def contains_action_dist(self) -> bool:
        return self.action_dist_type() > ProbDistribution.LoggedActionOnly

    @abc.abstractproperty
    def action_dist_type(self) -> ProbDistribution:
        pass

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Dict[str, Union[np.ndarray, dict]]]:
        # Note: singular form since we're iterating example by example.
        yield {
            "observation": ...,
            "action": ...,
            "action_dist": ...,
            "reward": ...,
            "terminal": ...,
            "info": ...
        }

class InMemoryDataset(OfflineDataset):
    def __init__(self, observations: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminals: np.ndarray, infos: list):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals
        self.infos = infos

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(low=0, high=1, shape=self.observations.shape[1:])

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(self.actions.shape[1])

    @property
    def action_dist_type(self) -> ProbDistribution:
        return ProbDistribution.Discrete

    def __iter__(self) -> Iterator[Dict[str, Union[np.ndarray, dict]]]:
        for i in range(self.observations.shape[0]):
            yield {
                "observation": self.observations[i],
                "action": self.actions[i],
                "action_dist": self.actions[i],
                "reward": self.rewards[i],
                "terminal": self.terminals[i],
                "info": self.infos[i]
            }


class EnvironmentRecorder(ActionSamplingEnv):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reset_buffer()
    
    def reset(self):
        self._obs = self._env.reset()
        return self._obs
    
    def step(self, action_dist):
        _obs = self._obs
        action, next_obs, reward, done, info = super().step(action_dist)
        self.append_buffer((_obs, action, reward, next_obs, done, action_dist, info))
        self._obs = next_obs
        return action, next_obs, reward, done, info
    
    def reset_buffer(self):
        self._buffer = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'infos/action_probs': [],
       }

    def append_buffer(self, tup):
        S, A, R, S_, done, action_dist, info = tup
        self._buffer['observations'].append(S)
        self._buffer['actions'].append(A)
        self._buffer['rewards'].append(R)
        self._buffer['next_observations'].append(S_)
        self._buffer['terminals'].append(done)
        self._buffer['infos/action_probs'].append(np.array(action_dist.probs))

    def npify(self, buffer):
        buffer_np = {}
        for k in buffer:
            if k == 'terminals':
                dtype = np.bool_
            else:
                dtype = np.float32

            buffer_np[k] = np.array(buffer[k], dtype=dtype)
        return buffer_np
    
    def save(self, save_path):
        dataset = h5py.File(save_path, 'w')
        buffer_np = self.npify(self._buffer)
        for k in buffer_np:
            dataset.create_dataset(k, data=buffer_np[k], compression='gzip')
