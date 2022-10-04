# Collect data for cartpole useing ppo agent
import argparse
import os

from offsim4rl.agents.ppo import PPOAgentRevealed
from offsim4rl.data import OfflineDataset, HDF5Dataset
from offsim4rl.encoders.heuristic import CartpoleBoxEncoder
from offsim4rl.evaluators.per_state_rejection import PerStateRejectionSampling
from offsim4rl.utils.vis_utils import plot_metric_from_spinup_progress


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    files_to_concatenate = []
    for file in os.listdir(args.input_dir):
        if file.startswith(args.prefix):
            files_to_concatenate.append(os.path.join(args.input_dir, file))

    HDF5Dataset.concatenate(
        files_to_concatenate,
        os.path.join(args.output_dir, f'{args.prefix}_concat.hdf5')
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='cartpole')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_dir', type=str, default='./inputs')

    args = parser.parse_args()
    main(args)
