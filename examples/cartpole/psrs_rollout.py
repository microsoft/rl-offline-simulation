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
from offsim4rl.utils.dataset_utils import record_dataset_in_memory
from offsim4rl.utils.vis_utils import plot_metric_from_spinup_progress


def concatenate_files(input_dir, output_dir, prefix):
    files_to_concatenate = []
    for file in os.listdir(input_dir):
        if file.startswith(prefix):
            files_to_concatenate.append(os.path.join(input_dir, file))

    dataset_name = f'{prefix}_concat_{time.time()}.hdf5'
    HDF5Dataset.concatenate(
        files_to_concatenate,
        os.path.join(output_dir, dataset_name)
    )

    return dataset_name


def main(args):
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.output_dir)
    logger = EpochLogger(**logger_kwargs)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'cartpole_psrs_seed_{args.seed}.hdf5')

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

    mpi_fork(args.cpu)  # run parallel code with mpi

    concat_dataset = concatenate_files(os.path.join(args.output_dir, args.dataset_name), os.path.join(args.output_dir, args.dataset_name), args.prefix)

    dataset = OfflineDataset.load_hdf5(os.path.join(args.output_dir, args.dataset_name, concat_dataset))
    print("Loaded concat dataset")
    box_encoder = CartpoleBoxEncoder()
    start_time = time.time()
    print(f"Starting PSRS intialization at: {start_time}")
    psrs = PerStateRejectionSampling(dataset, num_states=box_encoder.N_BOXES, encoder=box_encoder, new_step_api=True)
    print(f"Finished PSRS intialization. It took: {time.time() - start_time} seconds")

    dataset = record_dataset_in_memory(
        env,
        agent,
        num_samples=args.num_iter,
        seed=args.seed,
        new_step_api=True)

    dataset.save_hdf5(output_path)

    print('Simulation took: {0} minutes.'.format(str((time.time() - start_time) / 60)))

    # plot_metric_from_spinup_progress(
    #     progress_file_path=os.path.join(logger_kwargs['output_dir'], 'progress.txt'),
    #     metric_name='EpRet',
    #     output_dir=logger_kwargs['output_dir']
    # )


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
    parser.add_argument('--dataset_name', type=str, default='ppo')
    parser.add_argument('--prefix', type=str, default='cartpole')

    args = parser.parse_args()
    main(args)
