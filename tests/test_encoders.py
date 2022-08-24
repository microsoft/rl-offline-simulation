
import os
import numpy as np
import torch

from offsim4rl.encoders.homer import HOMEREncoder

from torch.utils.data import Dataset, DataLoader, random_split

def test_homer_encoder_inference():
    homer_encoder = HOMEREncoder(
<<<<<<< HEAD
        observation_dim=2,
        action_dim=5,
=======
        obs_dim=2, action_dim=5,
>>>>>>> main
        latent_size=25,
        hidden_size=64,
        model_path=os.path.join('models', 'encoders', 'encoder_model.pt'),
    )
    x, y = np.meshgrid(np.arange(0, 1, 0.002), np.arange(0, 1, 0.002))
    obs = torch.tensor(np.stack([x, y]).reshape((2, -1)).T, device=homer_encoder.device).float()
    emb = homer_encoder.encode(obs)

    assert emb.shape[0] == obs.shape[0]
