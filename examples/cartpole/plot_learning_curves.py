# Collect data for cartpole useing ppo agent
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from offsim4rl.utils.vis_utils import plot_metric_from_spinup_progress


def main(args):
    # LEARNING CURVES CARTPOLE ##
    df_pd_real = pd.read_csv(os.path.join(args.input_dir, 'progress-CartPole-v1_ppo-revealed-0.1.txt'), sep='\t')
    df_ppo = pd.read_csv(os.path.join(args.input_dir, 'progress-CartPole-PSRS-ppo-0.3.txt'), sep='\t')
    df_pd_control = pd.read_csv(os.path.join(args.input_dir, 'progress-CartPole-PSRS-ppo-pd-controller.txt'), sep='\t')
    df_ppo_box = pd.read_csv(os.path.join(args.input_dir, 'progress-CartPole-v1_ppo_box_encoder-0.1.txt'), sep='\t')

    dfs = [
        (df_ppo, 'tab:blue', 'PPO'),
        # (df_pd_control, 'tab:green', 'PD Controller'),
        (df_pd_real, 'k', 'Real'),
        # (df_ppo_box, 'tab:red', 'PPO-Box')
    ]
    metric_name = 'EpRet'
    for df, color, label in dfs:
        plt.plot(df['Average' + metric_name], color=color, label=label)
        plt.fill_between(df['Epoch'], df['Average' + metric_name] - df['Std' + metric_name], df['Average' + metric_name] + df['Std' + metric_name], alpha=0.2, color=color)

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'comparison_{metric_name}.png'))

    # plot_metric_from_spinup_progress(os.path.join(args.input_dir, 'progress-CartPole-PSRS-ppo-0.3.txt'), 'EpRet', args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()
    main(args)
