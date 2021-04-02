import argparse
from datetime import datetime
import pdb
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim

from classifiers import TextClassificationLSTM

from dataset import get_datasets, padded_collate, PadSequence
from torch.utils.data import DataLoader

# General teuxdeuxs
# TODO: Look into tensorboard logging.

def train_one_epoch(model, data_loader, optimizer, device):
    pass



def train(seed,
          device,
          batch_size,
          embedding_dim,
          hidden_dim,
          num_layers,
          bidirectional,
          dropout,
          batch_first,
          epochs):
    #TODO: set seed
    torch.manual_seed(seed)

    # set starting time of full training pipeline
    start_time = datetime.now()

    # set device
    #TODO: device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    # # define hyperparameters
    # batch_size = 1
    # # NB: vocab_size can only be defined after dataset is created
    # embedding_dim = 16
    # hidden_dim = 32
    # # NB: num_classes can more dynamically be defined after dataset is created
    # num_layers = 1
    # bidirectional = False
    # dropout = 0
    # device = device
    # batch_first = True

    # Load data and create dataset instance
    dataset = get_datasets()

    # get vocab size number of classes
    vocab_size = len(dataset.vocab)
    num_classes = dataset.num_classes

    # create dataloader with pre-specified batch size
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=PadSequence())

    # initialize model
    model = TextClassificationLSTM(batch_size = batch_size,
                                   vocab_size = vocab_size,
                                   embedding_dim = embedding_dim,
                                   hidden_dim = hidden_dim,
                                   num_classes = num_classes,
                                   num_layers = num_layers,
                                   bidirectional = bidirectional,
                                   dropout = dropout,
                                   device = device,
                                   batch_first = batch_first)

    # Print model architecture and trainable parameters
    print("MODEL ARCHITECTURE:")
    print(model)

    # count trainable parameters
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'The model has {trainable_params} trainable parameters.')

    # set up optimizer and loss criterion
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss() # combines LogSoftmax and NLL

    for epoch in tqdm(range(500), desc="Outer loop over epochs."):

        # set model to training mode. NB: in the actual training loop later on, this
        # statement goes at the beginning of each epoch.
        model.train()

        for batch_index, (input, target, lengths) in enumerate(data_loader):

            # Reset gradients for next iteration
            model.zero_grad()

            model_out = model(text = input, text_lengths = lengths)

            loss = criterion(model_out, target)
            loss.backward()

            optimizer.step()

            print(loss.item())


def parse_arguments(args = None):
    parser = argparse.ArgumentParser(description="Train discriminator neural text classifiers.")

    parser.add_argument("--seed", type=int, default=2021, help="Seed for reproducibility")
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")
    parser.add_argument('--batch_size', type=int, default=1, help="Number of datapoints to simultaneously process.")  # TODO: set to reasonable default after batching problem fixed.
    parser.add_argument('--embedding_dim', type=int, default=16, help="Dimensionality of embedding.")
    parser.add_argument('--hidden_dim', type=int, default=32, help="Size in LSTM hidden layer.")
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers in LSTM')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Create a bidirectional LSTM. LSTM will be unidirectional if omitted.')
    parser.add_argument('--dropout', type=float, default=0, help='Probability of applying dropout in final LSTM layer.')
    parser.add_argument('--batch_first', action='store_true', help='Assume batch size is first dimension of data.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of passes through entire dataset during training.')

    # Parse command line arguments
    args = parser.parse_args()

    return args

def test():
    pass

def hp_search():
    pass

if __name__ == "__main__":

    # Parse and print command line arguments for model configurations
    args = parse_arguments()

    print(f"Configuration: {args}")

    # Train model
    train(**vars(args))
