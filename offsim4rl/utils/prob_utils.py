import gym
import numpy as np

import torch
import torch.distributions as dist
from typing import Union

from offsim4rl.data import ProbDistribution

def get_uniform_dist(action_space: gym.Space, action_dist_type: ProbDistribution=ProbDistribution.Discrete):
    if isinstance(action_space, gym.spaces.Discrete):
        if action_dist_type == ProbDistribution.Discrete:
            return np.full((action_space.n,), fill_value=1.0 / action_space.n)
        elif action_dist_type == ProbDistribution.TorchDistribution:
            return dist.Categorical(probs=torch.ones(action_space.n) / action_space.n)
        else:
            raise NotImplementedError(f'Unsupported action_dist_type: {action_dist_type} for action_space: {action_space}')
    else:
        raise NotImplementedError(f'Unsupported action_space: {action_space}')

def sample_dist(prob_dist: Union[np.ndarray, dist.Distribution]):
    if isinstance(prob_dist, np.ndarray):
        return np.random.choice(len(prob_dist), p=prob_dist)
    elif isinstance(prob_dist, dist.Distribution):
        return prob_dist.sample().numpy()
    else:
        raise NotImplementedError(f'Unsupported prob_dist: {prob_dist}')
