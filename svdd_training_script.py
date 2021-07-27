import os
import numpy as np
import random
import copy
from collections import defaultdict

import matplotlib.pyplot as plt

plt.rc("font", size=8)

import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader

from dataloader import CleanSineDataset
from autoencoder_models import RNNEncoder, RNNDecoder, Seq2SeqAttn
from svdd_models import ReccurentSVDD
from data_utils import pad_collate

# comment out warnings if you are testing it out
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="RNNAutoEncoder")

parser.add_argument(
    "--save-freq",
    type=int,
    default=5,
    help="every x epochs save weights",
)
parser.add_argument(
    "--batch",
    type=int,
    default=20,
    help="train batch size",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="initial lr",
)
parser.add_argument(
    "--exp-name",
    default="testing_nn_hq",
    help="Experiment name",
)
args = parser.parse_args()


import wandb

os.environ["WANDB_NAME"] = args.exp_name
wandb.init(project="sub_seq")
wandb.config.update(args)
wandb.config.update({"dataset": "ae_sine_dataset"})


# fix random seeds
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

clean_dataset = CleanSineDataset()


clean_dataloader = DataLoader(
    clean_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=0,
    drop_last=True,
    collate_fn=pad_collate,
)


def calc_loss(batch_embeddings, model, metrics):
    # single epoch using the soft-boundary version from the paper

    dist = (batch_embeddings - model.center) ** 2
    dist = torch.sum(dist)

    scores = dist - model.R ** 2
    loss = model.R ** 2 + (1 / model.nu) * torch.mean(
        torch.max(torch.zeros_like(scores), scores)
    )

    model.R.data = torch.tensor(model.get_radius(dist), device=model.device())

    copy_rad = copy.deepcopy(model.R.data)
    metrics["SVDD_loss"] += loss.data.cpu().numpy() * batch_embeddings.size(0)
    metrics["Radius"] += copy_rad.cpu().numpy() * batch_embeddings.size(0)

    return loss


def print_metrics(metrics, epoch_samples, epoch):
    outputs = []
    outputs.append("{}:".format(str(epoch)))
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        wandb.log({k: metrics[k] / epoch_samples})
    print("{}".format(", ".join(outputs)))


def train_model(model, optimizer, scheduler, num_epochs=25):

    (_, anchor_batch, anchor_lens) = next(iter(clean_dataloader))
    anchor_batch = anchor_batch.to(device)
    model.init_center_c(anchor_batch, anchor_lens)
    print(model.center)
    center = copy.deepcopy(model.center).cpu().numpy()
    np.save("svdd_sin_center_.npy", center)

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        metrics = defaultdict(float)
        epoch_samples = 0

        for (_, clean, clean_lens) in tqdm(clean_dataloader):
            clean = clean.to(device)

            # forward
            optimizer.zero_grad()

            batch_embeddings = model.encoded_embedding(clean, clean_lens)

            loss = calc_loss(batch_embeddings, model, metrics)

            loss.backward()
            optimizer.step()

            epoch_samples += clean.size(0)

        print_metrics(metrics, epoch_samples, epoch)
        # deep copy the model
        if epoch % args.save_freq == 0:
            print("saving model")
            best_model_wts = copy.deepcopy(model.state_dict())
            weight_name = "svdd_sin_weights_" + str(epoch) + ".pt"
            torch.save(best_model_wts, weight_name)
            radius = copy.deepcopy(model.R.data)
            radius = radius.cpu().numpy()
            np.save("svdd_sin_radius_"+ str(epoch) + ".npy", radius)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


e = RNNEncoder(input_dim=1, bidirectional=True).to(device)
d = RNNDecoder(
    input_dim=(e.input_size + e.hidden_size * 2),
    hidden_size=e.hidden_size,
    bidirectional=True,
).to(device)

ae_model = Seq2SeqAttn(encoder=e, decoder=d).to(device)
ae_model.load_state_dict(torch.load("sin_ae_weights_30.pt"))

model = ReccurentSVDD(base_encoder=ae_model.encoder)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = None
train_model(model, optimizer, scheduler, num_epochs=150)