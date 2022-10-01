import gym
import h5py
import numpy as np
import random
import logging
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from collections import defaultdict

from offsim4rl.utils.prob_utils import sample_dist
from offsim4rl.agents.agent import Agent
from offsim4rl.data import OfflineDataset
from offsim4rl.evaluators.per_state_rejection import PerStateRejectionSampling


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
        new_step_api=True,
        include_infos=True,
        env_time_limit=500):

    def _append_buffer(buffer, **kwargs):
        for k, v in kwargs.items():
            buffer[k].append(v)

    def _npify(buffer):
        for k in buffer:
            buffer[k] = np.array(buffer[k])

    episode_id = worker_id
    obs = env.reset(seed=seed)

    buffer = defaultdict(lambda: [])
    t = 0
    reward = None

    info_keys = None
    info_supported_types = (int, float, np.ndarray)

    for _ in range(num_samples):
        action_dist = agent.begin_episode(obs) if t == 0 else agent.step(reward, obs)

        if isinstance(env, PerStateRejectionSampling):
            action, next_obs, reward, terminated, truncated, info = env.step_dist(action_dist)
            if next_obs is None or action is None:
                break
        else:
            action = sample_dist(action_dist)
            agent.commit_action(action)
            step_result = env.step(action)

            if len(step_result) == 4:
                # Old gym API. Discouraged, since it may incorrectly mark regular states 
                # terminal states, due to episode truncation.
                if new_step_api:
                    raise ValueError("new_step_api is enabled, but the environment seems return just 'done' instead of 'terminated' and 'truncated'. Use different environment or set new_step_api to False.")
                next_obs, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                # New gym API. Recommended for collecting data for offline simulation.
                next_obs, reward, terminated, truncated, info = step_result

        numpy_infos = {}

        if include_infos:
            if info_keys is None:
                # To make sure the vectors of the same length and are well-aligned with the rest of the experience,
                # only info keys that occur in the first step are being recorded.
                info_keys = [k for k, v in info.items() if isinstance(v, info_supported_types)]

            for k in info_keys:
                if k not in info:
                    numpy_infos[f"infos/{k}"] = None
                    logging.debug(f"Info key {k} is not present in the info dict. It will be recorded as None.")
                elif isinstance(info[k], info_supported_types):
                    numpy_infos[f"infos/{k}"] = info[k]
                else:
                    logging.warning(f"Example's info {k} is not of a supported type. Skipping.")

            remaining_keys = [k for k in info.keys() if k not in info_keys]
            logging.debug(f"Skipping the following keys in the example's info dict: {remaining_keys}")

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

        if t >= env_time_limit:
            truncated = True

        if terminated or truncated:
            agent.end_episode(reward, truncated=truncated)
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
