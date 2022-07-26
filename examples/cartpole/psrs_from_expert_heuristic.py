# Collect data for cartpole useing ppo agent
import argparse
import os
import time

import gym
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tools import mpi_fork
from spinup.utils.run_utils import setup_logger_kwargs

from offsim4rl.agents.ppo import PPOAgentRevealed
from offsim4rl.data import OfflineDataset, HDF5Dataset
from offsim4rl.encoders.heuristic import CartpoleBoxEncoder
from offsim4rl.evaluators.per_state_rejection import PerStateRejectionSampling
from offsim4rl.utils.vis_utils import plot_metric_from_spinup_progress


def concatenate_files(input_dir, output_dir, prefix):
    files_to_concatenate = []
    for file in os.listdir(input_dir):
        if file.startswith(prefix):
            files_to_concatenate.append(os.path.join(input_dir, file))

    dataset_name = f'{prefix}_concat.hdf5'
    HDF5Dataset.concatenate(
        files_to_concatenate,
        os.path.join(output_dir, dataset_name)
    )

    return dataset_name


def main(args):
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.output_dir)
    logger = EpochLogger(**logger_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)

    env = gym.make(args.env, new_step_api=True)
    agent = PPOAgentRevealed(
        env.observation_space,
        env.action_space,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        logger=logger,
        gamma=args.gamma,
        seed=args.seed,
        steps_per_epoch=args.steps,
    )

    # mpi_fork(args.cpu)  # run parallel code with mpi

    dataset_name = concatenate_files(os.path.join(args.input_dir, 'ppo'), args.output_dir, args.prefix)

    dataset = OfflineDataset.load_hdf5(os.path.join(args.input_dir, dataset_name))
    box_encoder = CartpoleBoxEncoder()
    start_time = time.time()
    psrs = PerStateRejectionSampling(dataset, num_states=box_encoder.N_BOXES, encoder=box_encoder, new_step_api=True)

    obs = psrs.reset()
    reward = None
    simulated_steps = 0
    steps_in_episode = 0
    while obs is not None:
        action_dist = agent.begin_episode(obs) if steps_in_episode == 0 else agent.step(reward, obs)
        action, obs, reward, terminated, truncated, info = psrs.step_dist(action_dist)
        if action is None:
            break
        agent.commit_action(action)

        simulated_steps += 1
        steps_in_episode += 1

        if steps_in_episode >= 500:
            truncated = True

        if terminated or truncated:
            print(f'End of episode. Steps in episode: {steps_in_episode}')
            agent.end_episode(reward, truncated=truncated)
            obs = psrs.reset()
            steps_in_episode = 0

    print('Simulation took: {0} minutes.'.format(str((time.time() - start_time) / 60)))

    plot_metric_from_spinup_progress(
        progress_file_path=os.path.join(logger_kwargs['output_dir'], 'progress.txt'),
        metric_name='EpRet',
        output_dir=logger_kwargs['output_dir']
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--num_iter', type=int, default=50000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--prefix', type=str, default='cartpole')

    args = parser.parse_args()
    main(args)
