# Collect data for continuous observation grid world using random policy
import argparse
from collections import defaultdict
import os

import gym
import gzip
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

from offsim4rl.envs import gridworld


def reset_buffer():
    return {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': [],
        'terminals': [],
        'infos/probs': [],
   }

def append_buffer(buffer, data):
    S, A, R, S_, done, p, info = data
    buffer['observations'].append(S)
    buffer['actions'].append(A)
    buffer['rewards'].append(R)
    buffer['next_observations'].append(S_)
    buffer['terminals'].append(done)
    buffer['infos/probs'].append(p)

def npify(buffer):
    for k in buffer:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        buffer[k] = np.array(buffer[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=int(1e6))
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(seed=args.seed)

    env_name = 'MyGridNaviCoords-v1'
    env = gym.make(env_name, seed=0)
    env.reset_task(np.array([4, 4]))
    pi = defaultdict(lambda : np.ones(5) / 5)  # random policy

    S = env.reset()
    done = False

    buffer = reset_buffer()
    t = 0
    for _ in range(args.num_samples):
        p = pi.get(tuple(S))
        A = rng.choice(env.nA, p=p)
        S_, R, done, info = env.step(A)
        append_buffer(buffer, (S, A, R, S_, done, p, info))

        if done:
            E = env.reset()
            done = False
            t = 0
        else:
            S = S_
            t = t + 1

    fname = f'{args.output_dir}/{env_name}_random.h5'
    dataset = h5py.File(fname, 'w')
    npify(buffer)
    for k in buffer:
        dataset.create_dataset(k, data=buffer[k], compression='gzip')

    states_visited = buffer['observations']

    fig, ax = plt.subplots(figsize=(4,4))
    plt.plot(np.array(states_visited)[:, 0], np.array(states_visited)[:, 1], alpha=0.5, lw=0, marker='.', mew=0, markersize=3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(f'{args.output_dir}/{env_name}-state_visitation.pdf')

if __name__ == "__main__":
    main()
