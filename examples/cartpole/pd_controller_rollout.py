# Collect data for cartpole useing ppo agent
import argparse
import os

import gym
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_fork
from spinup.utils.run_utils import setup_logger_kwargs

from offsim4rl.agents.heuristic import CartPolePDController
from offsim4rl.data import OfflineDataset
from offsim4rl.utils.dataset_utils import record_dataset_in_memory
from offsim4rl.utils.prob_utils import sample_dist
from offsim4rl.utils.vis_utils import CartPoleVisUtils

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'cartpole_pd_controller_{args.mode}_seed_{args.seed}_steps_{args.num_iter}.hdf5')

    env = gym.make(args.env, new_step_api=True)
    agent = CartPolePDController(
        env.action_space,
        mode=args.mode
    )

    # mpi_fork(args.cpu)  # run parallel code with mpi
    # rollout(agent, env, num_interactions=100, seed=args.seed)
    record_dataset(agent, env, num_interactions=args.num_iter, seed=args.seed, output_path=output_path)
    dataset = OfflineDataset.load_hdf5(output_path)
    CartPoleVisUtils.replay(dataset, record_clip=True, output_dir=os.path.join(args.output_dir, 'clips'))

def rollout(agent, env, num_interactions, seed=None):
    obs = env.reset(seed=seed)
    action_dist = agent.begin_episode(obs)
    for t in range(num_interactions):
        a = sample_dist(action_dist)
        obs, r, terminated, truncated, _ = env.step(a)
        env.render()
        action_dist = agent.step(r, obs)

        if terminated or truncated:
            agent.commit_action(a)
            agent.end_episode(r, truncated=truncated)
            obs = env.reset(seed=seed)
            action_dist = agent.begin_episode(obs)

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

    args = parser.parse_args()
    main(args)
