import abc
from typing import Dict, Union
import gym
import numpy as np
import enum

import numpy as np
import h5py

from offsim4rl.core import ActionSamplingEnv

class ProbDistribution(enum.Enum):
    """ Type of probability distribution used describe action. """

    """ No probabilities are provided. """
    NoProbability = 0

    """ Only probability of the logged action (single float). """
    # TODO: do we need this?
    LoggedActionOnly = 1

    """ Probability Mass Function, mapping each discrete action to a probability (a vector of floats). """
    Discrete = 2

    """ torch.distributions.Distribution object (provides most flexibility). """
    TorchDistribution = 3


# TODO: consider if we should inherit this from collections.abc.Iterable
#  and add a method like:
# @abc.abstractmethod
# def __iter__(self) -> Iterator[Dict[str, Union[np.ndarray, dict]]]:
#     # Note: singular form since we're iterating example by example.
#     yield {
#         "observation": ...,
#         "action": ...,
#         "action_dist": ...,
#         "reward": ...,
#         "terminal": ...,
#         "info": ...
#     }
class OfflineDataset:
    @abc.abstractproperty
    def observation_space(self) -> gym.Space:
        pass

    @abc.abstractproperty
    def action_space(self) -> gym.Space:
        pass

    @abc.abstractproperty
    def action_dist_type(self) -> ProbDistribution:
        pass

    @abc.abstractproperty
    def num_examples(self) -> int:
        pass

    @abc.abstractproperty
    def experience(self) -> Dict[str, Union[np.ndarray, dict]]:
        pass


class EnvironmentRecorder(ActionSamplingEnv):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reset_buffer()
    
    def reset(self):
        self._obs = self._env.reset()
        return self._obs
    
    def step_dist(self, action_dist):
        _obs = self._obs
        next_obs, reward, done, info = super().step_dist(action_dist)
        action = info['action']
        self.append_buffer((_obs, action, reward, next_obs, done, action_dist, info))
        self._obs = next_obs
        return next_obs, reward, done, info
    
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
        # TODO: right now we buffer experience in memory and then save everything at once.
        #   We should use h5py's resize functionality to append smaller buffers.
        file = h5py.File(save_path, 'w')
        buffer_np = self.npify(self._buffer)
        for k in buffer_np:
            file.create_dataset(k, data=buffer_np[k], compression='gzip')
