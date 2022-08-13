import argparse
import os
from datetime import datetime
import torch
from torch.utils.data import random_split

import random
import numpy as np
import pandas as pd

from offsim4rl.utils.dataset_utils import load_h5_dataset
from offsim4rl.data import SAS_Dataset
from offsim4rl.encoders.homer import HOMEREncoder
from offsim4rl.utils.vis_utils import plot_latent_state_color_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None)  # for debugging
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--latent_size', type=int, default=25)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--temperature_decay', type=bool, default=False)
    parser.add_argument('--input_dir', type=str, default='outputs')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_dir = f"./trial={datetime.now().isoformat(timespec='minutes').replace('-','').replace(':','')}," + \
                f"encoder_model=both,seed={args.seed}," + \
                f"dZ={args.latent_size},dH={args.hidden_size},lr={args.lr},weight_decay={args.weight_decay}/"
    os.makedirs(os.path.join(args.output_dir, model_dir, 'vis'), exist_ok=True)

    buffer = load_h5_dataset(os.path.join(args.input_dir, 'MyGridNaviCoords-v1_random.h5'))
    full_dataset = SAS_Dataset(buffer['observations'], buffer['actions'], buffer['next_observations'])

    # TRAINING
    if args.num_samples is not None:  # Limit sample size for debugging
        full_dataset = torch.utils.data.Subset(full_dataset, range(args.num_samples))

    train_dataset, val_dataset = random_split(
        full_dataset,
        [len(full_dataset) // 2, len(full_dataset) // 2],
        generator=torch.Generator().manual_seed(42)
    )

    homer_encoder = HOMEREncoder(
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        log_dir=os.path.join(args.output_dir, model_dir),
    )

    homer_encoder.train(
        train_dataset,
        val_dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        temperature_decay=args.temperature_decay,
    )

    # INFERENCE
    x, y = np.meshgrid(np.arange(0, 1, 0.002), np.arange(0, 1, 0.002))
    obs = torch.tensor(np.stack([x, y]).reshape((2, -1)).T, device=homer_encoder.device).float()

    emb = homer_encoder.encode(obs).detach().cpu()
    df_output = []
    for i, x in zip(emb, obs):
        df_output.append((i, *x))

    df_output = pd.DataFrame(df_output, columns=['i', 'x', 'y'])

    plot_latent_state_color_map(df_output, os.path.join(args.output_dir, model_dir, 'vis', 'latent_state.png'))
