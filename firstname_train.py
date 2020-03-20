import argparse
import matplotlib.pyplot as plt
import pandas as pd
import pyro
import torch
from pyro.infer import SVI, Trace_ELBO, TraceGraph_ELBO  # , ReweightedWakeSleep
from pyro.optim import Adam
from torch.utils.data import DataLoader

import util.config as config
from const import *
from firstname import NameParser
from util.dataset import NameDataset


def plot_losses(losses, folder: str = "result", filename: str = None):
    x = list(range(len(losses)))
    theta_losses = list(map(lambda loss: loss[0], losses))
    phi_losses = list(map(lambda loss: loss[1], losses))
    plt.plot(x, theta_losses, 'r--', label="WakeSleep Theta Loss")
    plt.plot(x, phi_losses, 'b--', label="WakeSleep Phi Loss")
    plt.title("Losses")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()


# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='UNNAMED_SESSION', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--peak_probs', help="Peakedness of the likelihood from 0 to 1", nargs='?', default=0.99,
                    type=float)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.001, type=float)
parser.add_argument('--batch_size', help='Size of the batch training on', nargs='?', default=128, type=int)
parser.add_argument('--num_epochs', help='Number of epochs', nargs='?', default=250, type=int)
parser.add_argument('--num_label', help='Number of labelled data to use', nargs='?', default=100000, type=int)
parser.add_argument('--num_no_label', help='Number of unlabelled data to use', nargs='?', default=100000, type=int)
parser.add_argument('--num_particle', help='Number of particles to evaluate for loss', nargs='?', default=10, type=int)
parser.add_argument('--alpha', help='Higher this is, more strength the supervision has', nargs='?', default=100,
                    type=float)
parser.add_argument('--loss', help='1: TRACE_ELBO 2: TraceGraph_ELBO 3: RWS 4: TVO', nargs='?', default=2, type=int)
parser.add_argument('--insomnia', help='From 0. to 1.', nargs='?', default=1., type=float)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=False, type=bool)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
SESSION_NAME = args.name
ALPHA = {"alpha": args.alpha}
NUM_PARTICLE = args.num_particle
to_save = {
    'session_name': SESSION_NAME,
    'hidden_size': args.hidden_size,
    'peak_probs': args.peak_probs,
    'batch_size': args.batch_size,
    'num_epochs': args.num_epochs,
    'learning_rate': args.lr,
    'num_label': args.num_label,
    'num_no_label': args.num_no_label,
    'num_particle': args.num_particle,
    'loss': args.loss,
    'insomnia': args.insomnia
}
config.save_json(f'config/{SESSION_NAME}.json', to_save)

# Pyro validation support in case of errors
pyro.enable_validation(True)

# Read CSV containing all data labelled
df = pd.read_csv("dataset/FB_FN.csv")

# Split data to supervised and unsupervised dataframes
if args.num_label > len(df) or args.num_no_label > len(df):
    raise Exception(f"Number of datapoint to use is greater than the size of the dataset.")
supervised_df = df.sample(n=args.num_label).fillna("")
unsupervised_df = df.sample(n=args.num_no_label).fillna("")

# Convert panda dataframes to PyTorch dataset
supervised_ds = NameDataset(supervised_df, "name", ALL_PRINTABLES, max_name_length=MAX_NAME_LENGTH)
unsupervised_ds = NameDataset(unsupervised_df, "name", ALL_PRINTABLES, max_name_length=MAX_NAME_LENGTH)

# Convert dataset to dataloader
supervised_dataloader = DataLoader(supervised_ds, batch_size=args.batch_size, shuffle=True) if len(
    supervised_ds) > 0 else []
unsupervised_dataloader = DataLoader(unsupervised_ds, batch_size=args.batch_size, shuffle=True) if len(
    unsupervised_ds) > 0 else []

print(f"Labelled Training Data Size: {len(supervised_ds)}")
print(f"Unlabelled Training Data Size: {len(unsupervised_ds)}")

# Set neural net parameters based on command args
name_parser = NameParser(hidden_sz=args.hidden_size, num_layers=2, peak_prob=args.peak_probs)

# If arg is continue_training then load weights
if args.continue_training:
    name_parser.load_checkpoint(filename=f"{SESSION_NAME}.pth.tar")

# Determine loss
loss = ReweightedWakeSleep(num_particles=NUM_PARTICLE, vectorize_particles=False, insomnia=args.insomnia)

# Set parameters for SVI to instantiate list for logging loss
optimizer = Adam({'lr': args.lr}, {"clip_norm": 5.0})
svi_loss = SVI(name_parser.model, name_parser.guide, optimizer, loss=loss)

"""
from pyro import poutine


def semisupervised_loss(model, guide, *args, **kwargs):
    alpha = ALPHA['alpha']
    batch_size = len(args[0])

    guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
    model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)

    model_sup_particle = model_trace.log_prob_sum(lambda name, site: site['type'] == 'sample' and name.find('sup') == 0)
    model_unsup_particle = model_trace.log_prob_sum(
        lambda name, site: site['type'] == 'sample' and name.find('unsup') == 0)
    guide_particle = guide_trace.log_prob_sum()
    elbo = (model_unsup_particle - guide_particle) / batch_size + alpha * model_sup_particle / batch_size
    return -elbo


def semisupervised_loss_and_grads(model, guide, *args, **kwargs):
    alpha = ALPHA['alpha']
    print(alpha)
    batch_size = len(args[0])
    num_particle = NUM_PARTICLE

    elbo_u, elbo_s = 0., 0.
    guide_traces, model_traces = [], []
    for _ in range(num_particle):
        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        model_trace = poutine.trace(poutine.replay(model, trace=guide_trace)).get_trace(*args, **kwargs)
        guide_traces.append(guide_trace)
        model_traces.append(model_trace)

        model_sup_particle = model_trace.log_prob_sum(
            lambda name, site: site['type'] == 'sample' and name.find('sup') == 0)
        model_unsup_particle = model_trace.log_prob_sum(
            lambda name, site: site['type'] == 'sample' and name.find('unsup') == 0)
        guide_particle = guide_trace.log_prob_sum()
        elbo_u += (model_unsup_particle - guide_particle)
        elbo_s += model_sup_particle

    # Must multiply N/M <AKA len(unsupervised_dataloader)> to unsupervised ELBO
    # where N: data size and M: batch size
    batch_constant = len(unsupervised_dataloader)
    elbo = batch_constant / num_particle * (elbo_u + alpha * elbo_s)

    surrogate_theta_particle = 0.
    surrogate_phi_particle = 0.
    for model_trace, guide_trace in zip(model_traces, guide_traces):
        guide_particle = guide_trace.log_prob_sum()

        # Compute theta gradient expectation
        model_z_sup_particle = model_trace.log_prob_sum(
            lambda name, site: site['type'] == 'sample' and name.find('sup_first_name') == 0)
        model_z_unsup_particle = model_trace.log_prob_sum(
            lambda name, site: site['type'] == 'sample' and name.find('unsup_first_name') == 0)
        surrogate_theta_particle += model_z_unsup_particle + (alpha * model_z_sup_particle)

        # Compute phi gradient expectation
        surrogate_phi_particle += (elbo_u - 1).detach() * guide_particle

    # Scale the gradient functions by N/M and num_particle
    surrogate_theta_particle = -batch_constant / num_particle * surrogate_theta_particle
    surrogate_phi_particle = -batch_constant / num_particle * surrogate_phi_particle

    # Backprop on theta and phi gradient function
    surrogate_theta_particle.backward(retain_graph=True)
    surrogate_phi_particle.backward(retain_graph=True)
    return -elbo


# Set parameters for SVI to instantiate list for logging loss
optimizer = Adam({'lr': args.lr}, {"clip_norm": 5.0})
svi_loss = SVI(name_parser.model, name_parser.guide, optimizer, loss=semisupervised_loss, loss_and_grads=semisupervised_loss_and_grads)
"""


def train(model, svi, supervised_dataloader: DataLoader, unsupervised_dataloader: DataLoader, num_epochs: int):
    # Run epochs and record average loss
    batch_losses = []

    for e in range(num_epochs):
        print(f"=== Epoch {e} ===")
        sup_data = iter(supervised_dataloader)
        unsup_data = iter(unsupervised_dataloader)
        batch_length = min(len(supervised_dataloader), len(unsupervised_dataloader))
        for i in range(batch_length):
            # X_s, Z_s = next(sup_data)
            # LETS CHEAT!!!!!!!!
            X_s = next(sup_data)
            Z_s = {"first_name": X_s}
            # !!!!!!!!!!!!!!!!!!

            X_u = next(unsup_data)
            curr_loss = svi.step(X_u, X_s, Z_s)
            norm_loss = (curr_loss[0] / len(X_u), curr_loss[1] / len(X_u))
            batch_losses.append(norm_loss)
            if (i % 2) == 0:
                print(f"Epoch {e} Batch {i}/{batch_length} Theta/Phi Loss: {norm_loss}")
                print(f"generated: {model.generate()}")
                print(f"inferred: travis => {model.infer(['travis'])}")
            if (i % 10) == 0:
                plot_losses(batch_losses, folder="result", filename=f"{SESSION_NAME}.png")
                model.save_checkpoint(folder="nn_model", filename=f"{SESSION_NAME}.pth.tar")


train(name_parser, svi_loss, supervised_dataloader, unsupervised_dataloader, args.num_epochs)
