# Collect data for cartpole useing ppo agent
import argparse
from collections import defaultdict
import os

import gym
import gzip
import h5py
import matplotlib.pyplot as plt
import numpy as np
import random

from spinup.algos.pytorch.ppo import core
from spinup.algos.pytorch.ppo.ppo import ppo
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.output_dir)

    ppo(
        lambda : gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs
    )

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # rng = np.random.default_rng(seed=args.seed)

    # env = gym.make('CartPole-v1', new_step_api=True)
    # agent = DiscreteRandom(env.action_space)
    # os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
    # path = os.path.join(TEST_OUTPUT_DIR, 'cartpole.hdf5')

    # dataset = record_dataset_in_memory(
    #     env,
    #     agent,
    #     num_samples=1000,
    #     seed=98052,
    #     new_step_api=new_step_api)
    # dataset.save_hdf5(path)


if __name__ == "__main__":
    main()
