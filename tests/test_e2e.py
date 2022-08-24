# End-to-end tests
import gym
import numpy as np
import pytest

import torch
from torch.utils.data import random_split

from offsim4rl.agents.discrete_random import DiscreteRandom
from offsim4rl.data import OfflineDataset, ProbDistribution, SAS_Dataset
from offsim4rl.evaluators.per_state_rejection import PerStateRejectionSampling
from offsim4rl.utils.dataset_utils import record_dataset_in_memory
from offsim4rl.utils.prob_utils import get_uniform_dist
from offsim4rl.encoders.homer import HOMEREncoder


def test_cartpole_simulation():
    env = gym.make('CartPole-v1', new_step_api=True)
    agent = DiscreteRandom(env.action_space)
    dataset = record_dataset_in_memory(
        env,
        agent,
        num_samples=1000,
        seed=98052)

    full_dataset = SAS_Dataset(dataset.experience['observations'], dataset.experience['actions'], dataset.experience['next_observations'])

    train_dataset, val_dataset = random_split(
        full_dataset,
        [len(full_dataset) // 2, len(full_dataset) // 2],
        generator=torch.Generator().manual_seed(42)
    )

    num_states = 10
    homer_encoder = HOMEREncoder(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        latent_size=num_states,
        hidden_size=16)

    homer_encoder.train(train_dataset, val_dataset, num_epochs=2)
    
    psrs = PerStateRejectionSampling(dataset, num_states=num_states, encoder=homer_encoder, new_step_api=True)

    simulated_steps = 0
    obs = psrs.reset()
    while obs is not None:
        action, obs, reward, terminated, truncated, info = psrs.step_dist(get_uniform_dist(env.action_space))
        simulated_steps += 1

        eps_done = terminated or truncated
        if eps_done:
            obs = psrs.reset()

    print(f'Simulated {simulated_steps} steps')
    assert simulated_steps > 900
