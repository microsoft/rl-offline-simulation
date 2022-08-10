import abc
from typing import Dict, Union
import gym
import numpy as np
import enum
import logging
from collections import namedtuple

import numpy as np
import h5py
import pickle

from offsim4rl.core import RevealedRandomnessEnv

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


Transition = namedtuple(
    'Transition',
    ['episode_id', 'step', 'observation', 'action', 'action_distribution', 'reward', 'next_observation', 'terminal', 'info'])


class OfflineDataset:
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            action_dist_type: ProbDistribution,
            **experience: Dict[str, object]):
        """
        Required keys in the experience dict:
          observations: np.ndarray,
          actions: np.ndarray,
          rewards: np.ndarray,
          next_observations: np.ndarray,
          terminals: np.ndarray
        
        Optional keys:
          episode_ids: np.ndarray,
          steps: np.ndarray,
          action_distributions: np.ndarray (TODO: can this be also a list of torch.distributions.Distribution objects?)
          infos: list of dicts (?)
        """

        self._validate_experience(experience)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dist_type = action_dist_type

        self.experience = experience
    
    def iterate_row_tuples(self):
        for i in range(self.experience['observations'].shape[0]):
            yield Transition(
                self.experience['episode_ids'][i] if 'episode_ids' in self.experience else None,
                self.experience['steps'][i] if 'steps' in self.experience else 0,
                self.experience['observations'][i],
                self.experience['actions'][i],
                self.experience['action_distributions'][i] if 'action_distributions' in self.experience else None,
                self.experience['rewards'][i],
                self.experience['next_observations'][i],
                self.experience['terminals'][i],
                self.experience['info'][i] if 'info' in self.experience else {})

    @classmethod
    def load_hdf5(cls, path, group_name=None):
        return HDF5Dataset(path, group_name)
    
    def save_hdf5(self, path, group_name=None):
        with h5py.File(path, "w") as fout:
            group = fout.create_group(group_name) if group_name else fout

            self._serialize_attr(group, 'observation_space', self.observation_space)
            self._serialize_attr(group, 'action_space', self.action_space)
            self._serialize_attr(group, 'action_dist_type', self.action_dist_type)

            for k in self.experience:
                group.create_dataset(k, data=self.experience[k], compression='gzip')
    
    @staticmethod
    def _serialize_attr(group, attr_name, attr_value):
        group.attrs.create(attr_name, np.void(pickle.dumps(attr_value)))

    @staticmethod
    def _validate_experience(experience):
        required_keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
        for k in required_keys:
            if k not in experience:
                raise ValueError(f'Missing required key {k} in experience')
        
        if experience['observations'].shape != experience['next_observations'].shape:
            raise ValueError(f'Shapes in observations and next_observations do not match')

        all_keys = required_keys + ['action_distributions', 'episode_ids', 'steps']
        for k in all_keys:
            if k in experience and experience[k].shape[0] != experience['observations'].shape[0]:
                raise ValueError(f'Length of {k} does not match length of observations')
        
        if 'steps' not in experience:
            logging.warning('Missing steps in experience. Algorithms may need to assume all states can be initial states...')

        if 'episode_ids' not in experience:
            logging.warning('Missing episode_ids in experience. Some algorithms may not be compatible with this dataset.')

class HDF5Dataset(OfflineDataset):
    def __init__(self, path, group_name=None):
        self._fin = h5py.File(path, 'r')
        group = self._fin.get(group_name, default=self._fin) if group_name else self._fin

        obs_space = self._deserialize_attr(group, 'observation_space')
        action_space = self._deserialize_attr(group, 'action_space')
        action_dist_type = self._deserialize_attr(group, 'action_dist_type', ProbDistribution.NoProbability)

        super().__init__(
            observation_space=obs_space,
            action_space=action_space,
            action_dist_type=action_dist_type,
            **{k: group[k] for k in group})

    def close(self):
        self._fin.close()

    def __del__(self):
        self.close()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    @staticmethod
    def _deserialize_attr(group, attr_name, default=None):
        bytes_str = group.attrs.get(attr_name, default=None)
        return pickle.loads(bytes_str.tobytes()) if bytes_str is not None else default
