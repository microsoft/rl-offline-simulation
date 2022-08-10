import matplotlib.pyplot as plt
import numpy as np

def plot_latent_state_color_map(latent_state_dataset, output_path='latent_state.png'):
    colors = [plt.cm.get_cmap('jet', 100)(i) for i in range(100)]
    np.random.default_rng(seed=0).shuffle(colors)
    colors = [plt.cm.tab20(i) for i in range(20)] + ['k', 'r', 'g', 'b', 'y'] + colors
    fig, ax = plt.subplots(figsize=(4, 4))
    for ix, x in latent_state_dataset:
        plt.plot(x[0], x[1], color=colors[ix], alpha=1, lw=0, marker='.', mew=0, markersize=5)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(output_path)
