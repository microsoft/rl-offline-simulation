# Collect data for cartpole useing ppo agent
import argparse
import os

import gym
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_fork
from spinup.utils.run_utils import setup_logger_kwargs

from offsim4rl.agents.heuristic import CartPolePDController
from offsim4rl.utils.dataset_utils import record_dataset_in_memory
from offsim4rl.utils.prob_utils import sample_dist

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(args.env, new_step_api=True)
    agent = CartPolePDController(
        env.action_space,
        mode=args.mode,
        eps=args.eps
    )

    logger_kwargs = setup_logger_kwargs(args.exp_name + f'_eps_{args.eps}', args.seed, data_dir=args.output_dir)

    mpi_fork(args.cpu)  # run parallel code with mpi
    for s in range(100, 200):
        output_path = os.path.join(args.output_dir, f'cartpole_pd_controller_eps_{args.eps}_{args.mode}_seed_{s}_steps_{args.num_iter}.hdf5')
        record_dataset(agent, env, num_interactions=args.num_iter, seed=s, output_path=output_path)
    # rollout(agent, env, args.num_iter, args.seed, EpochLogger(**logger_kwargs))

def rollout(agent, env, num_interactions, seed=None, logger=None):
    obs = env.reset(seed=seed)
    action_dist = agent.begin_episode(obs)
    ep_len = 0
    ep_ret = 0
    logger.store(EpRet=0, EpLen=0)
    for t in range(num_interactions):
        ep_len += 1
        a = sample_dist(action_dist)
        agent.commit_action(a)
        obs, r, terminated, truncated, _ = env.step(a)
        action_dist = agent.step(r, obs)
        ep_ret += r

        if terminated or truncated:
            a = sample_dist(action_dist)
            agent.commit_action(a)
            agent.end_episode(r, truncated=truncated)
            obs = env.reset()
            action_dist = agent.begin_episode(obs)
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            ep_len = 0
            ep_ret = 0

        if t % 1000 == 0:
            logger.log_tabular('Epoch', t // 1000)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.dump_tabular()

def record_dataset(agent, env, num_interactions=100, seed=None, output_path='.'):
    dataset = record_dataset_in_memory(
        env,
        agent,
        num_samples=num_interactions,
        seed=seed,
        new_step_api=True)

    dataset.save_hdf5(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_iter', type=int, default=10000)
    parser.add_argument('--mode', type=str, default='theta')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--exp_name', type=str, default='carpole')
    parser.add_argument('--eps', type=float, default=0.05)


    args = parser.parse_args()
    main(args)
