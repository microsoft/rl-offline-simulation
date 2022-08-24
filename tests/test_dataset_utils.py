import os
import gym
import numpy as np
import pytest
from offsim4rl.utils.dataset_utils import record_dataset_in_memory
from offsim4rl.agents.discrete_random import DiscreteRandom
from offsim4rl.data import OfflineDataset

TEST_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '.test_output')

def test_record_dataset_new_step_api():
    _test_record_dataset(new_step_api=True)

def test_record_dataset_old_step_api():
    _test_record_dataset(new_step_api=False)

def test_record_dataset_incompatible_step_api():
    env = gym.make('CartPole-v1', new_step_api=False)
    agent = DiscreteRandom(env.action_space)
    with pytest.raises(ValueError):
        # Default is new_step_api=True
        dataset = record_dataset_in_memory(
            env,
            agent,
            num_samples=10,
            seed=98052)

def _test_record_dataset(new_step_api):
    env = gym.make('CartPole-v1', new_step_api=new_step_api)
    agent = DiscreteRandom(env.action_space)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    path = os.path.join(TEST_OUTPUT_DIR, 'cartpole.hdf5')
    
    if os.path.exists(path):
        os.remove(path)
    assert not os.path.exists(path)

    dataset = record_dataset_in_memory(
        env,
        agent,
        num_samples=1000,
        seed=98052,
        new_step_api=new_step_api)
    dataset.save_hdf5(path)

    # Check that the file was created
    assert os.path.exists(path)

    dataset2 = OfflineDataset.load_hdf5(path)
    
    assert dataset.action_space == dataset2.action_space
    assert dataset.observation_space == dataset2.observation_space
    assert dataset.action_dist_type == dataset2.action_dist_type
    assert dataset.experience.keys() == dataset2.experience.keys()
    for k in dataset.experience.keys():
        assert np.all(np.isclose(dataset.experience[k], dataset2.experience[k])), f'Not close: {k}'