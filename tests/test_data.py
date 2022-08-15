import gym
import numpy as np
import pytest
import os
from offsim4rl.data import OfflineDataset, ProbDistribution

TEST_OUTPUT_DIR = '.test_output'
HDF5_FILE_PATH = os.path.join(TEST_OUTPUT_DIR, 'dataset.hdf5')


def test_save_and_load_offline_dataset_discrete_action_dist():
    dataset = OfflineDataset(
        observation_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Discrete(4),
        action_dist_type=ProbDistribution.Discrete,
        observations=np.array([[0.0], [0.5]], dtype=np.float32),
        actions=np.array([0, 1], dtype=np.int64),
        action_distributions=np.full((2, 4), fill_value=0.25, dtype=np.float32),
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        next_observations=np.array([[0.5], [0.0]], dtype=np.float32),
        terminals=np.array([False, True], dtype=bool))
    
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    dataset.save_hdf5(HDF5_FILE_PATH)
    dataset2 = OfflineDataset.load_hdf5(HDF5_FILE_PATH)

    assert gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32) == dataset2.observation_space
    assert gym.spaces.Discrete(4) == dataset2.action_space
    assert ProbDistribution.Discrete == dataset2.action_dist_type

    assert np.array_equal(np.array([[0.0], [0.5]], dtype=np.float32), dataset2.experience['observations'])
    assert np.array_equal(np.array([0, 1], dtype=np.int64), dataset2.experience['actions'])
    assert np.array_equal(np.full((2, 4), fill_value=0.25, dtype=np.float32), dataset2.experience['action_distributions'])
    assert np.array_equal(np.array([0.0, 1.0], dtype=np.float32), dataset2.experience['rewards'])
    assert np.array_equal(np.array([[0.5], [0.0]], dtype=np.float32), dataset2.experience['next_observations'])
    assert np.array_equal(np.array([False, True], dtype=bool), dataset2.experience['terminals'])


def test_save_and_load_offline_dataset_no_action_dist():
    dataset = OfflineDataset(
        observation_space=gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        action_space=gym.spaces.Discrete(4),
        action_dist_type=ProbDistribution.NoProbability,
        observations=np.array([[0.0], [0.5]], dtype=np.float32),
        actions=np.array([0, 1], dtype=np.int64),
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        next_observations=np.array([[0.5], [0.0]], dtype=np.float32),
        terminals=np.array([False, True], dtype=bool))
    
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    dataset.save_hdf5(HDF5_FILE_PATH)
    dataset2 = OfflineDataset.load_hdf5(HDF5_FILE_PATH)

    assert gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32) == dataset2.observation_space
    assert gym.spaces.Discrete(4) == dataset2.action_space
    assert ProbDistribution.NoProbability == dataset2.action_dist_type

    assert np.array_equal(np.array([[0.0], [0.5]], dtype=np.float32), dataset2.experience['observations'])
    assert np.array_equal(np.array([0, 1], dtype=np.int64), dataset2.experience['actions'])
    assert np.array_equal(np.array([0.0, 1.0], dtype=np.float32), dataset2.experience['rewards'])
    assert np.array_equal(np.array([[0.5], [0.0]], dtype=np.float32), dataset2.experience['next_observations'])
    assert np.array_equal(np.array([False, True], dtype=bool), dataset2.experience['terminals'])

# TODO: tests with continuous distributions.