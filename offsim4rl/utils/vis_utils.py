import matplotlib.pyplot as plt
import numpy as np

def plot_latent_state_color_map(df_output, output_path='latent_state.png'):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.scatter(df_output['x'], df_output['y'], c=df_output['i'], cmap='nipy_spectral', marker='.', lw=0, s=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(output_path)
