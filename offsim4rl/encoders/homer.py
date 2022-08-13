import argparse
import copy
from datetime import datetime
import logging
import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from offsim4rl.encoders.models import EncoderModel
from offsim4rl.utils.tb_utils import TensorboardWriter
from offsim4rl.utils.dataset_utils import load_h5_dataset
from offsim4rl.utils.vis_utils import plot_latent_state_color_map
from offsim4rl.data import SAS_Dataset


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s', level=logging.INFO)

class HOMEREncoder():
    def __init__(self, latent_size, hidden_size, model_path=None, log_dir='./logs'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EncoderModel(2, 5, latent_size, hidden_size).to(self.device)
        if model_path:
            self.model.load(model_path)
            self.model.eval()

        self.tb_writer = TensorboardWriter(log_dir=log_dir)
        self.log_dir = log_dir

    def train(
        self,
        train_dataset,
        val_dataset,
        optimizer=None,
        lr=1e-3,
        weight_decay=0.0,
        loss_fn=None,
        num_epochs=1000,
        batch_size=64,
        patience_threshold=50,
        temperature_decay=False,
        model_dir=None,
        model_name='encoder_model.pt',
    ):
        # gumbel softmax temperature decay schedule
        tau0 = 1.0    # initial temperature
        TAU_ANNEAL_RATE = 0.005
        TAU_MIN = 0.5

        if not self.model.training:
            self.model.train()

        if optimizer is None:
            optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, weight_decay=weight_decay)

        if loss_fn is None:
            loss_fn = HOMEREncoder._calc_loss

        best_val_loss, best_epoch, train_loss = 0.69, -1, 0.69
        best_model = copy.deepcopy(self.model)
        num_train_examples, num_val_examples = 0, 0
        patience_counter = 0

        val_set_errors, past_entropy = [], []

        train_loader_real = DataLoader(train_dataset, batch_size, shuffle=True)
        train_loader_impo = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader_real = DataLoader(val_dataset, batch_size, shuffle=True)
        val_loader_impo = DataLoader(val_dataset, batch_size, shuffle=True)

        for epoch_ in range(1, num_epochs + 1):
            train_loss, mean_entropy, num_train_examples = 0.0, 0.0, 0
            for train_batch in zip(train_loader_real, train_loader_impo):
                if temperature_decay:
                    tau = max(TAU_MIN, tau0 * np.exp(-TAU_ANNEAL_RATE*epoch_))
                else:
                    tau = tau0

                loss, info_dict = loss_fn(self.model, train_batch, tau)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 40)
                optimizer.step()

                for key in info_dict:
                    self.tb_writer.log_scalar(key, info_dict[key])

                batch_size = len(train_batch)
                train_loss = train_loss + float(info_dict["classification_loss"]) * batch_size
                # mean_entropy = mean_entropy + float(info_dict["mean_entropy"]) * batch_size
                num_train_examples = num_train_examples + batch_size

            train_loss = train_loss / float(max(1, num_train_examples))
            # mean_entropy = mean_entropy / float(max(1, num_train_examples))

            # Evaluate on test batches
            val_loss = 0
            num_val_examples = 0
            for val_batch in zip(val_loader_real, val_loader_impo):
                _, info_dict = loss_fn(self.model, val_batch, discretized=True)
                batch_size = len(val_batch)
                val_loss = val_loss + float(info_dict["classification_loss"]) * batch_size
                num_val_examples = num_val_examples + batch_size

            val_loss = val_loss / float(max(1, num_val_examples))
            logging.info("Epoch %r" % epoch_)
            logging.info("Train Loss is %r" % round(train_loss, 3))
            logging.info("Val   Loss is %r" % round(val_loss, 3))

            self.tb_writer.log_scalar('train_loss', train_loss)
            self.tb_writer.log_scalar('val_loss', val_loss)

            val_set_errors.append(val_loss)
            # past_entropy.append(mean_entropy)

            if val_loss < best_val_loss:
                patience_counter = 0
                best_val_loss = val_loss
                best_epoch = epoch_
                best_model.load_state_dict(self.model.state_dict())
            else:
                # Check patience condition
                patience_counter += 1  # number of max_epoch since last increase
                if val_loss > 0.8:  # diverged
                    break

                if patience_counter == patience_threshold:
                    print("Patience Condition Triggered: No improvement for %r epochs" % patience_counter)
                    break

            if epoch_ % 10 == 0:
                best_model.save(os.path.join(model_dir, f"epoch_{epoch_}_" + model_name))
            
            if epoch_ % 10 == 0:
                x, y = np.meshgrid(np.arange(0, 1, 0.002), np.arange(0, 1, 0.002))
                obs = torch.tensor(np.stack([x, y]).reshape((2, -1)).T, device=self.device).float()

                with torch.no_grad():
                    emb = self.encode(obs).detach().cpu()
                
                df_output = []
                for i, x in zip(emb, obs):
                    df_output.append((i, *x))
                df_output = pd.DataFrame(df_output, columns=['i', 'x', 'y'])

                plot_latent_state_color_map(df_output, os.path.join(self.log_dir, 'vis', f'epoch_{epoch_}_latent_state.png'))

            if model_dir is None:
                model_dir = os.path.join(self.log_dir, 'models')

            os.makedirs(model_dir, exist_ok=True)
            best_model.save(os.path.join(model_dir, model_name))
            self.model.eval()

        logging.info("(Discretized: %r), Train/Test = %d/%d, Best Tune Loss %r at max_epoch %r, Train Loss after %r epochs is %r " % (
            False,
            num_train_examples,
            num_val_examples,
            round(best_val_loss, 2),
            best_epoch,
            epoch_,
            round(train_loss, 2))
        )

    def encode(self, observations):
        if not self.model or self.model.training:
            raise ValueError("Model not initialized. Either train a new model for the encoder or load an existing one.")

        with torch.no_grad():
            log_prob = F.log_softmax(self.model.obs_encoder(observations), dim=1)
            _, argmax_indices = log_prob.max(dim=1)

        return argmax_indices

    @staticmethod
    def _calc_loss(model, batch, temperature=1.0, discretized=False):
        (obs, a, next_obs_real), (_, _, next_obs_impo) = batch

        # Compute loss
        log_probs_real, _ = model.gen_log_prob(prev_observations=obs, actions=a, observations=next_obs_real, temperature=temperature, discretized=discretized)
        log_probs_impo, _ = model.gen_log_prob(prev_observations=obs, actions=a, observations=next_obs_impo, temperature=temperature, discretized=discretized)
        classification_loss = (F.nll_loss(log_probs_real, torch.ones(len(obs)).long().to(obs.device)) + F.nll_loss(log_probs_impo, torch.zeros(len(obs)).long().to(obs.device))) / 2

        info_dict = dict()
        info_dict["classification_loss"] = classification_loss

        return classification_loss, info_dict


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--latent_size', type=int, default=25)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()

    model_dir = f"./trial={datetime.now().isoformat(timespec='minutes').replace('-','').replace(':','')}," + \
                f"encoder_model=both,seed={args.seed}," + \
                f"dZ={args.latent_size},dH={args.hidden_size},lr={args.lr},weight_decay={args.weight_decay}/"

    homer_encoder = HOMEREncoder(
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        model_path=os.path.join(args.output_dir, 'encoder_model.pt'),
        log_dir=os.path.join(args.output_dir, 'logs'),
    )

    x, y = np.meshgrid(np.arange(0, 1, 0.002), np.arange(0, 1, 0.002))
    obs = torch.tensor(np.stack([x, y]).reshape((2, -1)).T, device=homer_encoder.device).float()

    emb = homer_encoder.encode(obs).detach().cpu()
    df_output = []
    for i, x in zip(emb, obs):
        df_output.append((i, *x))

    df_output = pd.DataFrame(df_output, columns=['i', 'x', 'y'])

    plot_latent_state_color_map(df_output, os.path.join(args.output_dir, 'latent_state.png'))
