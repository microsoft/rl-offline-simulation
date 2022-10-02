# Collect data for cartpole useing ppo agent
import argparse
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from offsim4rl.data import OfflineDataset
from offsim4rl.encoders.heuristic import CartpoleBoxEncoder
from offsim4rl.utils.vis_utils import CartPoleVisUtils


def main(args):
    dataset_names = [
        # 'cartpole_ppo_seed_197_steps_50000.hdf5',
        # 'cartpole_psrs_seed_9.hdf5',
        # 'cartpole_pd_controller_eps_0.2_theta-omega_seed_0_steps_20000.hdf5',
        # 'cartpole_pd_controller_eps_0.2_theta_seed_0_steps_20000.hdf5',
        # 'cartpole_pd_controller_eps_0.2_omega_seed_0_steps_20000.hdf5',
        # 'cartpole_pd_controller_eps_0.05_theta-omega_seed_199_steps_50000.hdf5',
        # 'cartpole_pd_controller_omega_seed_0_steps_20000.hdf5',
        # 'cartpole_pd_controller_theta_seed_0_steps_20000.hdf5',
        # 'cartpole_pd_controller_theta-omega_seed_0_steps_20000.hdf5'
        'cartpole_psrs_seed_2_theta-omega.hdf5',
        'cartpole_psrs_seed_3_theta-omega.hdf5',
    ]

    for dataset_name in dataset_names:
        dataset = OfflineDataset.load_hdf5(os.path.join(args.input_dir, dataset_name))
        obs = dataset.experience['observations']
        obs_encodings = CartpoleBoxEncoder().encode(dataset.experience['observations'])
        actions = dataset.experience['actions']

        # plt.plot(obs[:3000, 0])
        # plt.show()

        # df = pd.DataFrame(zip(obs_encodings, actions), columns=['obs', 'act'])
        # df.hist()
        # sns.pairplot(df, plot_kws={"s": 3})

        # OBS HIST
        bins = np.arange(-5, 180, 5)  # fixed bin size
        plt.xlim([min(obs_encodings) - 5, max(obs_encodings) + 5])
        plt.hist(obs_encodings, bins=bins, alpha=0.5, label=dataset_name)

        # OBS-ACT HIST
        # plt.clf()
        # plt.hist2d(actions, obs_encodings, cmap=plt.cm.nipy_spectral)
        # plt.xlabel('action')
        # plt.ylabel('box')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig(os.path.join(args.output_dir, 'action_obs_data_coverage.png'))

        CartPoleVisUtils.replay(dataset, record_clip=True, total_num_steps=2000, output_dir=os.path.join(args.output_dir, f'clips_{dataset_name}'))

    plt.title('Cartpole data coverage in box encodings')
    plt.xlabel('box')
    plt.ylabel('count')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'data_coverage.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--dataset_name', type=str, default='cartpole')

    args = parser.parse_args()
    main(args)
