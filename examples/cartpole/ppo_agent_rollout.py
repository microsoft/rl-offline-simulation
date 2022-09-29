# Collect data for cartpole useing ppo agent
import argparse
import os

import gym
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_fork, proc_id
from spinup.utils.run_utils import setup_logger_kwargs

from offsim4rl.agents.ppo import PPOAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--num_iter', type=int, default=50000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.output_dir)
    logger = EpochLogger(**logger_kwargs)

    env = gym.make(args.env, new_step_api=True)
    agent = PPOAgent(
        env.observation_space,
        env.action_space,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        logger=logger,
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
    )

    mpi_fork(args.cpu)  # run parallel code with mpi
    num_interactions = args.num_iter
    obs = env.reset(seed=args.seed)
    a = agent.begin_episode(obs)
    for t in range(num_interactions):
        obs, r, terminated, truncated, _ = env.step(a)
        a = agent.step(r, obs)

        if terminated or truncated:
            agent.end_episode(r, truncated=truncated)
            obs = env.reset(seed=args.seed)
            a = agent.begin_episode(obs)


if __name__ == "__main__":
    main()
