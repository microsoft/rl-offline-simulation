# These tests verify that currently used version of the gym package is compatible with the offsim4rl package.
import gym
import numpy as np

def test_terminated():
    env = gym.make('CartPole-v1', new_step_api=True)
    env.reset()
    step_result = env.step(0)
    assert len(step_result) == 5, f'Expected new gym API returning 5 elements, but got {len(step_result)}. Are you using gym package >=0.25.0?'
    obs, reward, terminated, truncated, info = step_result
    assert type(obs) == np.ndarray
    assert type(reward) == float
    assert type(terminated) == bool
    assert type(truncated) == bool
    assert type(info) == dict
