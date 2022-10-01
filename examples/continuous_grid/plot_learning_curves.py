# Collect data for cartpole useing ppo agent
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from offsim4rl.utils.vis_utils import plot_metric_from_spinup_progress


def main(args):
    ## LEARNING CURVES GRID##
    # metric_name = 'EpRet'
    # for seed in range(10):
    #     plt.clf()
    #     df = pd.read_csv(os.path.join(args.input_dir, f'seed={seed}', 'progress.txt'), sep='\t')
    #     plt.plot(df['Average' + metric_name], color='k')
    #     plt.fill_between(df['Epoch'], df['Average' + metric_name] - df['Std' + metric_name], df['Average' + metric_name] + df['Std' + metric_name], alpha=0.2, color='k')
    #     plt.xlabel('Epoch')
    #     plt.ylabel(metric_name)
    #     plt.legend()
    #     plt.savefig(os.path.join(args.output_dir, f'seed_{seed}_{metric_name}.png'))

    # dfs_real = [pd.read_csv(os.path.join(args.input_dir, f'grid_true/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    # df_real = dfs_real[0][:30]

    # dfs_homer50 = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_homer50/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    # df_homer50 = dfs_homer50[0][:30]

    # dfs_psrs_oracle = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_oracle/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    # df_psrs_oracle = dfs_psrs_oracle[0][:30]

    # dfs_baseline_obs_only = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_baseline-obs-only/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    # df_baseline_obs_only = dfs_baseline_obs_only[0][:30]

    # dfs_baseline_act_only = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_baseline-act-only/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    # df_baseline_act_only = dfs_baseline_act_only[0][:30]

    # dfs_baseline_random = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_baseline-random/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    # df_baseline_random = dfs_baseline_random[0][:30]

    # fig, ax = plt.subplots(figsize=(3.6, 3.6))

    # col_name = 'AverageEpRet'

    # plt.plot(df_real[col_name], color='k', label='Real')
    # plt.fill_between(df_real['Epoch'], df_real[col_name] - df_real[col_name.replace('Average', 'Std')], df_real[col_name] + df_real[col_name.replace('Average', 'Std')], alpha=0.2, color='k')

    # plt.plot(df_psrs_oracle[col_name], lw=1.5, alpha=0.9, label='PSRS-oracle', c='tab:blue')
    # # plt.scatter(len(df_psrs_oracle[col_name])-1, df_psrs_oracle[col_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)
    # plt.fill_between(df_psrs_oracle['Epoch'], df_psrs_oracle[col_name] - df_psrs_oracle[col_name.replace('Average', 'Std')], df_psrs_oracle[col_name] + df_psrs_oracle[col_name.replace('Average', 'Std')], alpha=0.2, color='tab:blue')

    # plt.plot(df_homer50[col_name], lw=1.5, alpha=0.9, label='PSRS-encoder', c='tab:green')
    # # plt.scatter(len(df_homer50[col_name])-1, df_homer50[col_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)
    # plt.fill_between(df_homer50['Epoch'], df_homer50[col_name] - df_homer50[col_name.replace('Average', 'Std')], df_homer50[col_name] + df_homer50[col_name.replace('Average', 'Std')], alpha=0.2, color='tab:green')

    # # plt.plot(df_baseline_obs_only[col_name], lw=1.5, alpha=0.9, label='PSRS-obs-only', c='tab:orange')
    # # # plt.scatter(len(df_baseline_obs_only[col_name])-1, df_baseline_obs_only[col_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)
    # # plt.fill_between(df_baseline_obs_only['Epoch'], df_baseline_obs_only[col_name] - df_baseline_obs_only[col_name.replace('Average', 'Std')], df_baseline_obs_only[col_name] + df_baseline_obs_only[col_name.replace('Average', 'Std')], alpha=0.2, color='tab:orange')

    # # plt.plot(df_baseline_act_only[col_name], lw=1.5, alpha=0.9, label='PSRS-act-only', c='tab:purple')
    # # # plt.scatter(len(df_baseline_act_only[col_name])-1, df_baseline_act_only[col_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)
    # # plt.fill_between(df_baseline_act_only['Epoch'], df_baseline_act_only[col_name] - df_baseline_act_only[col_name.replace('Average', 'Std')], df_baseline_act_only[col_name] + df_baseline_act_only[col_name.replace('Average', 'Std')], alpha=0.2, color='tab:purple')

    # # plt.plot(df_baseline_random[col_name], lw=1.5, alpha=1, label='PSRS-random', c='tab:red')
    # # # plt.scatter(len(df_baseline_random[col_name])-1, df_baseline_random[col_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)
    # # plt.fill_between(df_baseline_random['Epoch'], df_baseline_random[col_name] - df_baseline_random[col_name.replace('Average', 'Std')], df_baseline_random[col_name] + df_baseline_random[col_name.replace('Average', 'Std')], alpha=0.2, color='tab:red')

    # plt.xlabel('Training Epoch')
    # plt.ylabel('Learning Curves')
    # plt.xlim(0, 15)
    # plt.legend(loc='lower right')
    # plt.savefig(os.path.join(args.output_dir, 'grid_ppo_learning_curves_FINAL.png'), bbox_inches='tight')
    # plt.show()

    # PERFORMANCE CHARTS GRID##
    dfs_real = [pd.read_csv(os.path.join(args.input_dir, f'grid_true/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    df_real = pd.concat(dfs_real).groupby('Epoch').mean().reset_index()

    dfs_homer50 = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_homer50/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    df_homer50 = pd.concat(dfs_homer50).groupby('Epoch').mean().reset_index()

    dfs_psrs_oracle = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_oracle/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    df_psrs_oracle = pd.concat(dfs_psrs_oracle).groupby('Epoch').mean().reset_index()

    dfs_baseline_obs_only = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_baseline-obs-only/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    df_baseline_obs_only = pd.concat(dfs_baseline_obs_only).groupby('Epoch').mean().reset_index()

    dfs_baseline_act_only = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_baseline-act-only/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    df_baseline_act_only = pd.concat(dfs_baseline_act_only).groupby('Epoch').mean().reset_index()

    dfs_baseline_random = [pd.read_csv(os.path.join(args.input_dir, f'grid_psrs_baseline-random/seed={seed}/progress.txt'), sep='\t') for seed in range(10)]
    df_baseline_random = pd.concat(dfs_baseline_random).groupby('Epoch').mean().reset_index()

    fig, ax = plt.subplots(figsize=(3.6, 3.6))

    col_name = 'AverageEpRet'

    plt.plot(df_real[col_name], color='k', ls='--', label='Real')
    plt.fill_between(df_real['Epoch'], df_real[col_name.replace('Average', 'Min')], df_real[col_name.replace('Average', 'Max')], alpha=0.2, color='k')

    plt.plot(df_psrs_oracle[col_name], lw=1.5, alpha=0.9, label='PSRS-oracle', c='tab:gray')
    plt.scatter(len(df_psrs_oracle[col_name])-1, df_psrs_oracle[col_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)

    plt.plot(df_homer50[col_name], lw=1.5, alpha=0.9, label='PSRS-encoder', c='tab:green')
    plt.scatter(len(df_homer50[col_name])-1, df_homer50[col_name].iloc[-1], marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)

    plt.plot(df_baseline_obs_only[col_name], lw=1.5, alpha=0.9, label='PSRS-obs-only', c='tab:orange')
    plt.scatter(len(df_baseline_obs_only[col_name])-1, df_baseline_obs_only[col_name].iloc[-1],
                marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)

    plt.plot(df_baseline_act_only[col_name], lw=1.5, alpha=0.9, label='PSRS-act-only', c='tab:purple')
    plt.scatter(len(df_baseline_act_only[col_name])-1, df_baseline_act_only[col_name].iloc[-1],
                marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)

    plt.plot(df_baseline_random[col_name], lw=1.5, alpha=1, label='PSRS-random', c='tab:red')
    plt.scatter(len(df_baseline_random[col_name])-1, df_baseline_random[col_name].iloc[-1],
                marker='o', color=plt.gca().lines[-1].get_color(), alpha=1, zorder=11)

    plt.xlabel('Epoch')
    plt.ylabel('Learning Performance')
    plt.xlim(0,50)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(args.output_dir, 'grid_ppo_FINAL.pdf'), bbox_inches='tight')
    plt.show()

    T = int(np.median([len(df) for df in dfs_homer50]))

    for name_sim, df_sim, dfs_sim in zip(
        ['Real', 'oracle', 'HOMER50', 'obs', 'act', 'random'],
        [df_real, df_psrs_oracle, df_homer50, df_baseline_obs_only, df_baseline_act_only, df_baseline_random], 
        [dfs_real, dfs_psrs_oracle, dfs_homer50, dfs_baseline_obs_only, dfs_baseline_act_only, dfs_baseline_random], 
    ):
        print(
            name_sim, 
            int(np.median([len(df) for df in dfs_sim])),
            '{:.3f}'.format(
                np.sqrt(np.mean(np.square(
                    df_sim[col_name][:T] - df_real[col_name][:T])))
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./inputs')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    args = parser.parse_args()
    main(args)
