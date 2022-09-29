# Collect data for cartpole useing ppo agent
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd

from offsim4rl.utils.vis_utils import plot_metric_from_spinup_progress


def main(args):
    # df_ppo = pd.read_csv(os.path.join(args.input_dir, 'progress-CartPole-PSRS-ppo.txt'), sep='\t')
    # df_pd_control = pd.read_csv(os.path.join(args.input_dir, 'progress-CartPole-PSRS-ppo-pd-controller.txt'), sep='\t')
    df_pd_real = pd.read_csv(os.path.join(args.input_dir, 'progress-CartPole-v1_ppo-revealed.txt'), sep='\t')
    df_ppo_box = pd.read_csv(os.path.join(args.input_dir, 'CartPole-v1_ppo_box_encoder.txt'), sep='\t')

    metric_name = 'EpRet'
    # for df, color, label in zip([df_ppo, df_pd_control, df_pd_real], ['tab:blue', 'tab:green', 'k'], ['PPO', 'PD Controller', 'Real']):
    for df, color, label in zip([df_pd_real, df_ppo_box], ['tab:blue', 'tab:green'], ['PPO', 'PPO-box']):
        plt.plot(df['Average' + metric_name], color=color, label=label)
        plt.fill_between(df['Epoch'], df['Average' + metric_name] - df['Std' + metric_name], df['Average' + metric_name] + df['Std' + metric_name], alpha=0.2, color=color)

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'comparison_{metric_name}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()
    main(args)
