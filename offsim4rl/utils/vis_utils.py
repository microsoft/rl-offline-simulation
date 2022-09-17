import matplotlib.pyplot as plt
import numpy as np

def plot_latent_state_color_map(df_output, output_path='latent_state.png'):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.scatter(df_output['x'], df_output['y'], c=df_output['i'], cmap='nipy_spectral', marker='.', lw=0, s=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(output_path)


class CartPoleVisUtils():
    @staticmethod
    def plot_visited_states(visited_states, output_path='state_visitation.png'):
        plt.clf()
        fig, ax = plt.subplots(figsize=(6, 5))
        plt.plot(np.array(visited_states)[:, 0], np.array(visited_states)[:, 2], alpha=0.25, lw=0, marker='.', mew=0, markersize=3)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.25, 0.25)
        plt.axhline(-0.2095, c='gray')
        plt.axhline(0.2095, c='gray')
        plt.axvline(-2.4, c='gray')
        plt.axvline(2.4, c='gray')
        plt.xlabel('Cart Position')
        plt.ylabel('Pole Angle')
        plt.savefig(output_path)

    @staticmethod
    def plot_latent_state(df_output, output_path='latent_state.png'):
        plt.clf()
        plt.scatter(df_output['x'], df_output['y'], c=df_output['i'], cmap='nipy_spectral', marker='.', lw=0, s=1)
        plt.xlim(-2.5, 2.5)
        plt.ylim(-0.25, 0.25)
        plt.xlabel('Cart Position')
        plt.ylabel('Pole Angle')
        plt.savefig(output_path)
