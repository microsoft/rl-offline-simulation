# Collect data for cartpole useing ppo agent
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from offsim4rl.utils.vis_utils import plot_metric_from_spinup_progress


def plot_aggregated_performance_curves(metric_name, input_dir, output_dir):
    dfs_apa = [pd.read_csv(os.path.join(input_dir, 'CartPole-v1_apa', f'CartPole-v1_apa_s{seed}', 'progress.txt'), sep='\t') for seed in [0, 1, 2, 3, 4, 42]]
    df_apa = pd.concat(dfs_apa).groupby('Epoch').mean().reset_index()

    dfs_ppo = [pd.read_csv(os.path.join(input_dir, 'CartPole-v1_ppo', f'CartPole-v1_ppo_s{seed}', 'progress.txt'), sep='\t') for seed in [0, 1, 2, 3, 4, 42]]
    df_ppo = pd.concat(dfs_ppo).groupby('Epoch').mean().reset_index()


    # dfs_psrs_ppo = [pd.read_csv(os.path.join(input_dir, 'CartPole-PSRS-ppo_20x50k', f'CartPole-PSRS-ppo_s{seed}', 'progress.txt'), sep='\t') for seed in [20, 21, 22, 23, 24]]
    # dfs_psrs_ppo = [pd.read_csv(os.path.join(input_dir, 'CartPole-PSRS-ppo_50x20k', f'CartPole-PSRS-ppo_s{seed}', 'progress.txt'), sep='\t') for seed in [5, 8, 9]]
    # dfs_psrs_ppo = [pd.read_csv(os.path.join(input_dir, 'CartPole-PSRS-ppo_100x20k', f'CartPole-PSRS-ppo_s{seed}', 'progress.txt'), sep='\t') for seed in [0,1,2,3,4,11,12,13,14]]

    # df_psrs_ppo = pd.concat(dfs_psrs_ppo).groupby('Epoch').mean().reset_index()

    # dfs_psrs_pd = [pd.read_csv(os.path.join(input_dir, 'CartPole-PSRS-ppo-pd-controller', f'CartPole-PSRS-ppo-pd-controller_s{seed}', 'progress.txt'), sep='\t') for seed in [5, 6, 7, 8, 9, 10]]
    # df_psrs_pd = pd.concat(dfs_psrs_pd).groupby('Epoch').mean().reset_index()

    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    plt.plot(df_apa[metric_name], color='tab:orange', ls='--', label='APA')
    plt.fill_between(df_apa['Epoch'], df_apa[metric_name.replace('Average', 'Min')], df_apa[metric_name.replace('Average', 'Max')], alpha=0.2, color='tab:orange')

    plt.plot(df_ppo[metric_name], color='k', ls='--', label='PPO')
    plt.fill_between(df_ppo['Epoch'], df_ppo[metric_name.replace('Average', 'Min')], df_ppo[metric_name.replace('Average', 'Max')], alpha=0.2, color='k')

    # plt.plot(df_psrs_ppo[metric_name], lw=1.5, alpha=0.9, label='PSRS-ppo', c='tab:orange')
    # # plt.fill_between(df_psrs_ppo['Epoch'], df_psrs_ppo[metric_name.replace('Average', 'Min')], df_psrs_ppo[metric_name.replace('Average', 'Max')], alpha=0.2, color='tab:orange')
    # plt.scatter(len(df_psrs_ppo[metric_name])-1, df_psrs_ppo[metric_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)

    # plt.plot(df_psrs_pd[metric_name], lw=1.5, alpha=0.9, label='PSRS-pd-controller', c='tab:green')
    # # plt.fill_between(df_psrs_pd['Epoch'], df_psrs_pd[metric_name.replace('Average', 'Min')], df_psrs_pd[metric_name.replace('Average', 'Max')], alpha=0.2, color='tab:green')
    # plt.scatter(len(df_psrs_pd[metric_name])-1, df_psrs_pd[metric_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)

    plt.xlabel('Epoch')
    plt.ylabel('Learning Performance')
    plt.xlim(0, 50)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'cartpole_learning_performance.png'), bbox_inches='tight')
    plt.show()

def plot_learning_curves(metric_name, input_dir, output_dir):
    # LEARNING CURVES CARTPOLE ##
    df_ppo = pd.read_csv(os.path.join(input_dir, 'CartPole-v1_ppo_s42', 'progress.txt'), sep='\t')
    df_apa = pd.read_csv(os.path.join(input_dir, 'CartPole-v1_apa_s42', 'progress.txt'), sep='\t')

    # df_pd_real = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-v1_ppo-revealed-0.1.txt'), sep='\t')
    # df_ppo = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-PSRS-ppo-0.5.txt'), sep='\t')

    # df_pd_theta_omega_real_0 = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-v1_pd_controller_theta-omega_eps_0.2.txt'), sep='\t')
    # df_pd_theta_omega_real_1 = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-v1_pd_controller_theta-omega_eps_0.05.txt'), sep='\t')
    # df_pd_theta_omega_real_2 = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-v1_pd_controller_theta-omega_eps_0.1.txt'), sep='\t')

    # df_pd_omega_real_0 = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-v1_pd_controller_omega_eps_0.05.txt'), sep='\t')
    # df_pd_theta_real_0 = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-v1_pd_controller_theta_eps_0.05.txt'), sep='\t')

    # df_pd_control_0 = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-PSRS-ppo-pd-controller-theta-omega-eps-0.2-20x50k-0.1.txt'), sep='\t')

    # df_ppo_box = pd.read_csv(os.path.join(input_dir, 'progress-CartPole-v1_ppo_box_encoder-0.1.txt'), sep='\t')

    dfs = [
        (df_ppo, 'k', 'PPO'),
        (df_apa, 'tab:red', 'APA'),
        # (df_pd_real, 'k', 'Real'),
        # (df_ppo, 'tab:red', 'PPO'),
        # (df_pd_control_0, 'tab:green', 'PD Controller-theta-omega-20x50k-1'),
        # (df_pd_theta_omega_real_0, 'tab:orange', 'PD Controller-theta-omega Real eps 0.2'),
        # (df_pd_theta_omega_real_2, 'tab:green', 'PD Controller-theta-omega Real eps 0.1'),
        # (df_pd_theta_omega_real_1, 'tab:blue', 'PD Controller-theta-omega Real eps 0.05'),
        # (df_pd_omega_real_0, 'tab:green', 'PD Controller-omega Real eps 0.05'),
        # (df_pd_theta_real_0, 'tab:orange', 'PD Controller-theta Real eps 0.05'),

        # (df_ppo_box, 'tab:red', 'PPO-Box')
    ]

    for df, color, label in dfs:
        plt.plot(df['Average' + metric_name], color=color, label=label)
        plt.fill_between(df['Epoch'], df['Average' + metric_name] - df['Std' + metric_name], df['Average' + metric_name] + df['Std' + metric_name], alpha=0.2, color=color)

    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'comparison_{metric_name}.png'))


def main(args):
    plot_aggregated_performance_curves('AverageEpRet', args.input_dir, args.output_dir)
    # plot_learning_curves('EpRet', args.input_dir, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()
    main(args)
