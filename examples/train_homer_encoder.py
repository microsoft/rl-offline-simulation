import argparse
import copy
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from joblib import Parallel, delayed
import joblib
import copy

from datetime import datetime


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--latent_size', type=int, default=25)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    args = parser.parse_args([])
