import gym
import numpy as np
import torch
import pytest
from torch.distributions import Categorical
from offsim4rl.core import RevealedRandomnessEnvWrapper

def test_revealed_randomness_env_numpy():
    env = gym.make('CartPole-v0')
    env = RevealedRandomnessEnvWrapper(env)
    assert hasattr(env, 'step_dist')
    obs = env.reset()
    assert type(obs) == np.ndarray
    action, next_obs, reward, terminated, timeout, info = env.step_dist(np.array([0.5, 0.5]))
    assert type(action) == np.int64
    assert type(next_obs) == np.ndarray
    assert type(reward) == float
    assert type(terminated) == bool
    assert type(timeout) == bool
    assert type(info) == dict

def test_revealed_randomness_env_torch():
    env = gym.make('CartPole-v0')
    env = RevealedRandomnessEnvWrapper(env)
    assert hasattr(env, 'step_dist')
    obs = env.reset()
    assert type(obs) == np.ndarray
    action, next_obs, reward, terminated, timeout, info = env.step_dist(Categorical(torch.tensor([0.5, 0.5])))
    assert type(action) == torch.Tensor
    assert type(next_obs) == np.ndarray
    assert type(reward) == float
    assert type(terminated) == bool
    assert type(timeout) == bool
    assert type(info) == dict

def test_revealed_randomness_env_other():
    env = gym.make('CartPole-v0')
    env = RevealedRandomnessEnvWrapper(env)
    assert hasattr(env, 'step_dist')
    obs = env.reset()
    assert type(obs) == np.ndarray
    with pytest.raises(ValueError):
        env.step_dist(0)
