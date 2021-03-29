# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import argparse
import os
from datetime import datetime
import time
from pdb import set_trace
import tqdm

import numpy as np

import torch
import torch.optim as optim

from classifiers import TextClassificationLSTM

from dataset import get_datasets
from torch.utils.data import DataLoader

def train_one_epoch(model, data_loader, optimizer, device):



def train():
    #TODO: set seed
    torch.manual_seed(2021)

    # set starting time of full training pipeline
    start_time = datetime.now()

    # set device
    #TODO: device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    # define hyperparameters
    batch_size = 1
    # NB: vocab_size can only be defined after dataset is created
    embedding_dim = 16
    hidden_dim = 32
    # NB: num_classes can more dynamically be defined after dataset is created
    num_layers = 1
    bidirectional = False
    dropout = 0
    device = 'cpu'
    batch_first = True

    # Load data and create dataset instance
    dataset = get_datasets()

    # get vocab size number of classes
    vocab_size = len(dataset.vocab)
    num_classes = dataset.num_classes

    # create dataloader with pre-specified batch size
    data_loader = DataLoader(dataset, batch_size = batch_size)

    # initialize model
    model = TextClassificationLSTM(batch_size = batch_size,
                                   vocab_size = vocab_size,
                                   embedding_dim = embedding_dim,
                                   hidden_dim = hidden_dim,
                                   num_classes = num_classes,
                                   num_layers = num_layers,
                                   bidirectional = False,
                                   dropout = dropout,
                                   device = device,
                                   batch_first = True)

    # Print model architecture and trainable parameters
    print("MODEL ARCHITECTURE:")
    print(model)

    # count trainable parameters
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'The model has {trainable_params} trainable parameters.')

    # set up optimizer and loss criterion
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss() # combines LogSoftmax and NLL

    for epoch in range(500):

        # set model to training mode. NB: in the actual training loop later on, this
        # statement goes at the beginning of each epoch.
        model.train()

        for batch_index, (input, target) in enumerate(data_loader):

            text_lengths = [input.shape[1]]

            # Reset gradients for next iteration
            model.zero_grad()

            model_out = model(text = input, text_lengths = text_lengths)

            loss = criterion(model_out, target)
            loss.backward()

            optimizer.step()

            print(loss.item())


def parse_arguments(args = None):
    parser = argparse.ArgumentParser()

    #TODO

if __name__ == "__main__":

    # Parse and print command line arguments for model configurations
    # args = parse_arguments()
    # print(args)
    # args = vars(args)

    # Train model
    # train(**args, args = args)
    train()
