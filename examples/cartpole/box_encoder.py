import argparse
import os

import gym
import pandas as pd

from offsim4rl.data import OfflineDataset
from offsim4rl.encoders.heuristic import CartpoleBoxEncoder
from offsim4rl.utils.vis_utils import CartPoleVisUtils

def main(args):
    seed = 0
    num_iter = 20000

    input_path = os.path.join(args.input_dir, f'cartpole_seed_{seed}_steps_{num_iter}.hdf5')
    dataset = OfflineDataset.load_hdf5(input_path)
    visited_states = dataset.experience['observations']
    CartPoleVisUtils.plot_visited_states(visited_states, f'{args.output_dir}/state_visitation.png')

    df_output = pd.DataFrame([], columns=['x', 'y', 'i'])
    df_output['x'] = visited_states[:, 0]
    df_output['y'] = visited_states[:, 2]
    df_output['i'] = CartpoleBoxEncoder().encode(visited_states)
    CartPoleVisUtils.plot_latent_state(df_output, output_path=f'{args.output_dir}/latent_state.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./outputs')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()
    main(args)
