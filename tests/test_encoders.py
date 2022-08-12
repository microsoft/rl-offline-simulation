
import os
import torch

from offsim4rl.encoders.homer import HOMEREncoder

from torch.utils.data import Dataset, DataLoader, random_split
from offsim4rl.utils.dataset_utils import load_h5_dataset
# from offsim4rl.utils.vis_utils import plot_latent_state_color_map
from offsim4rl.data import SAS_Dataset

import torch.nn.functional as F
def encode_observations(model, observations):
    log_prob = F.log_softmax(model.obs_encoder(observations), dim=1)
    _, argmax_indices = log_prob.max(dim=1)
    return argmax_indices

# def test_homer_encoder_inference():
homer_encoder = HOMEREncoder(
    latent_size=25,
    hidden_size=64,
    model_path=os.path.join('models', 'encoders', 'encoder_model.pt')
)

obs = torch.tensor([0.5053, 0.3705])
i = homer_encoder.encode(obs)

### plot using actual generated data
# buffer = load_h5_dataset(os.path.join('outputs', 'MyGridNaviCoords-v1_random.h5'))
# full_dataset = SAS_Dataset(buffer['observations'], buffer['actions'], buffer['next_observations'], )
# train_dataset, val_dataset = random_split(
#     full_dataset,
#     [len(full_dataset) // 2, len(full_dataset) // 2],
#     generator=torch.Generator().manual_seed(42)
# )
# plot_dataset = val_dataset[:][0]

### plot using meshgrid
import numpy as np
x, y = np.meshgrid(np.arange(0, 1, 0.002), np.arange(0, 1, 0.002))
plot_dataset = torch.tensor(np.stack([x,y]).reshape((2, -1)).T, device=homer_encoder.device).float()

with torch.no_grad():
    output = encode_observations(homer_encoder.model, plot_dataset).detach().cpu()

import pandas as pd
df_output = []
for i, x in zip(output, plot_dataset):
    df_output.append((i, *x))
df_output = pd.DataFrame(df_output, columns=['i', 'x', 'y'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(4,4))
plt.scatter(df_output['x'], df_output['y'], c=df_output['i'], cmap='nipy_spectral', marker='.', lw=0, s=1)
plt.xlim(0,1)
plt.ylim(0,1)
plt.savefig('./outputs/latent_state.png', dpi=300)

# plot_latent_state_color_map(output, os.path.join('./', 'latent_state.png'))
