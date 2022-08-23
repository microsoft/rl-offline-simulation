import gym
import h5py
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict

from offsim4rl.utils.prob_utils import sample_dist
from offsim4rl.agents.agent import Agent
from offsim4rl.data import OfflineDataset


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def load_h5_dataset(h5path):
    dataset = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                dataset[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                dataset[k] = dataset_file[k][()]
    return dataset


def record_dataset_in_memory(
        env: gym.Env,
        agent: Agent,
        num_samples,
        seed=None,
        worker_id=0,
        workers_num=1,
        new_step_api=True):
    
    def _append_buffer(buffer, **kwargs):
        for k,v in kwargs.items():
            buffer[k].append(v)

    def _npify(buffer):
        for k in buffer:
            buffer[k] = np.array(buffer[k])

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed=seed)

    episode_id = worker_id
    obs = env.reset()

    buffer = defaultdict(lambda: [])
    t = 0
    reward = None
    for _ in range(num_samples):
        action_dist = agent.begin_episode(obs) if t == 0 else agent.step(reward, obs)

        action = sample_dist(action_dist)
        step_result = env.step(action)

        if len(step_result) == 4:
            # Old gym API. Discouraged, since it may incorrectly mark regular states as terminal states, due to episode truncation.
            if new_step_api:                
                raise ValueError("new_step_api is enabled, but the environment seems return just 'done' instead of 'terminated' and 'truncated'. Use different environment or set new_step_api to False.")
            next_obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            # New gym API. Recommended for collecting data for offline simulation.
            next_obs, reward, terminated, truncated, info = env.step(action)

        numpy_infos = {}
        for k,v in info.items():
            # is v numerical?
            if isinstance(v, (int, float, np.ndarray)):
                numpy_infos[f"infos/{k}"] = v

        _append_buffer(
            buffer,
            episode_ids=episode_id,
            steps=t,
            observations=obs,
            actions=action,
            action_distributions=action_dist,
            rewards=reward,
            next_observations=next_obs,
            terminals=terminated,
            **numpy_infos)

        if terminated or truncated:
            if terminated:
                agent.end_episode(reward)
            obs = env.reset()

            t = 0
            episode_id += workers_num
        else:
            obs = next_obs
            t = t + 1

    _npify(buffer)
    dataset = OfflineDataset(
        observation_space=env.observation_space,
        action_space=env.action_space,
        action_dist_type=agent.action_dist_type,
        **buffer)
    return dataset
