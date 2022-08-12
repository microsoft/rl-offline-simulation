
import os
import torch

from offsim4rl.encoders.homer import HOMEREncoder

def test_homer_encoder_inference():
    homer_encoder = HOMEREncoder(
        latent_size=25,
        hidden_size=64,
        model_path=os.path.join('models', 'encoders', 'encoder_model.pt')
    )

    obs = torch.tensor([0.5053, 0.3705])
    i = homer_encoder.encode(obs)

