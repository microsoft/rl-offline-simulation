import gym
import numpy as np

import torch
import torch.distributions as dist

def get_uniform_dist(action_space: gym.Space):
    if isinstance(action_space, gym.spaces.Discrete):
        return dist.Categorical(probs=torch.ones(action_space.n) / action_space.n)
    else:
        raise NotImplementedError()
