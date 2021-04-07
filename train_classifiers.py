import argparse
from datetime import datetime
from pdb import set_trace
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim

from classifiers import TextClassificationLSTM

from dataset import get_datasets, padded_collate, PadSequence
from torch.utils.data import DataLoader

# General teuxdeuxs
# TODO: .to(device) everything
# TODO: Look into tensorboard logging.
# TODO: Try WordTokenizer
# Complete training/testing loop:

def train_one_epoch(model,
                    data_loader,
                    criterion,
                    optimizer,
                    device,
                    start_iteration,
                    clip_grad,
                    max_norm,
                    log_interval):

    # set model to train mode
    model.train()

    # print("\n Starting to train next epoch ... \n")

    for iteration, (batch_inputs, batch_labels, batch_lengths) in tqdm(enumerate(data_loader, start=start_iteration)):

        # Reset gradients for next iteration
        model.zero_grad()

        # move everything to device
        batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                    batch_lengths.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward pass through model
        log_probs = model(batch_inputs, batch_lengths)  # log_probs shape: (batch_size, num_classes)

        # Evaluate loss, gradients, and update network parameters
        loss = criterion(log_probs, batch_labels)
        loss.backward()

        # Apply gradient clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=max_norm)

        optimizer.step()

        predictions = torch.argmax(log_probs, dim=1, keepdim=True)
        correct = predictions.eq(batch_labels.view_as(predictions)).sum().item()
        accuracy = correct / log_probs.size(0)

        # print(loss.item())
        #
        # print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
        #                    Examples/Sec = {:.2f}, "
        #       "Accuracy = {:.2f}, Loss = {:.3f}".format(
        #     datetime.now().strftime("%Y-%m-%d %H:%M"), step,
        #     config.train_steps, config.batch_size, examples_per_second,
        #     accuracy, loss
        # ))

        if iteration % log_interval == 0:

            print(
                "\n [{}] Iteration {} | Batch size = {} |"
                "Average loss = {:.6f} | Accuracy = {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), iteration,
                    batch_labels.size(0), loss.item(), accuracy
                )
            )

    return iteration



def train(seed,
          device,
          batch_size,
          embedding_dim,
          hidden_dim,
          num_layers,
          bidirectional,
          dropout,
          batch_first,
          epochs,
          clip_grad,
          max_norm,
          train_frac,
          log_interval):

    #TODO: set seed
    torch.manual_seed(seed)

    # set starting time of full training pipeline
    start_time = datetime.now()

    # set device
    device = torch.device(device)
    print(f"Device: {device}")

    print("Starting data preprocessing ... ")
    data_prep_start = datetime.now()

    # Load data and create dataset instance
    dataset = get_datasets()

    # Train, val, and test splits
    train_size = int(train_frac * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # TODO: adapt the get_datasets() functions s.t. the dataset splits are instances of BlogDataset
    # TODO initialize the LSTM with the vocab of the training set alone, so you can see how the model handles unknown tokens during testing

    # get vocab size number of classes
    vocab_size = len(dataset.vocab)
    num_classes = dataset.num_classes

    # create dataloaders with pre-specified batch size
    # data_loader = DataLoader(dataset=dataset,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          collate_fn=PadSequence())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=PadSequence())

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=PadSequence())

    print(f'Data preprocessing finished. Data prep took {datetime.now() - data_prep_start}.')

    # initialize model
    print("Initializing model ...")
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

    # model to device
    model.to(device)

    # Print model architecture and trainable parameters
    print("MODEL ARCHITECTURE:")
    print(model)

    # count trainable parameters
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'The model has {trainable_params} trainable parameters.')

    # set up optimizer and loss criterion
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()  # combines LogSoftmax and NLL

    # initialize iterations at zero
    iterations = 0

    for epoch in tqdm(range(epochs)):

        epoch_start_time = datetime.now()
        try:
            # set model to training mode. NB: in the actual training loop later on, this
            # statement goes at the beginning of each epoch.
            model.train()
            iterations = train_one_epoch(model=model, data_loader=train_loader, criterion=criterion,
                                         optimizer=optimizer, device=device, start_iteration=iterations,
                                         clip_grad=clip_grad, max_norm=max_norm, log_interval=log_interval)

        except KeyboardInterrupt:
            print("Manually stopped current epoch")
            __import__('pdb').set_trace()

        print(f"###############################################################")
        print(f"Epoch {epoch} finished, validation loss: val_loss, ppl: ppl")
        print(f"###############################################################")
        print("Current epoch training took {}".format(datetime.now() - epoch_start_time))

        test_loss, test_accuracy = evaluate_performance(model=model,
                                                        data_loader=test_loader,
                                                        device=device,
                                                        criterion=criterion)


        # for batch_index, (input, target, lengths) in enumerate(data_loader):
        #
        #     # Reset gradients for next iteration
        #     model.zero_grad()
        #
        #     # zero the parameter gradients
        #     # optimizer.zero_grad()
        #
        #     # Forward pass through model
        #     model_out = model(text = input, text_lengths = lengths)
        #
        #     # Evaluate loss, gradients, and update network parametrers
        #     loss = criterion(model_out, target)
        #     loss.backward()
        #
        #     # Apply gradient clipping
        #     if clip_grad:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(),
        #                                        max_norm=max_norm)
        #
        #     optimizer.step()
        #
        #     print(loss.item())

def evaluate_performance(model, data_loader, device, criterion):

    # set model to evaluation mode
    model.eval()

    # initialize loss and number of correct predictions
    test_loss = 0
    total_correct = 0

    with torch.no_grad():
        for iteration, (batch_inputs, batch_labels, batch_lengths) in enumerate(data_loader):

            # move everything to device
            batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                        batch_lengths.to(device)

            # forward pass through model
            log_probs = model(batch_inputs, batch_lengths)  # log_probs shape: (batch_size, num_classes)

            # compute and sum up batch loss and correct predictions
            test_loss += criterion(log_probs, batch_labels)

            predictions = torch.argmax(log_probs, dim=1, keepdim=True)
            total_correct += predictions.eq(batch_labels.view_as(predictions)).sum().item()

        # average losses and accuracy
        test_loss /= len(data_loader.dataset)
        accuracy = total_correct / len(data_loader.dataset)

        print(
            "Performance on test set: "
            "Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})".format(
                test_loss, total_correct, len(data_loader.dataset), accuracy
            )
        )

        return test_loss, accuracy



def hp_search():
    pass

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
    parser.add_argument('--clip_grad', action='store_true',
                        help = 'Apply gradient clipping. Set to True if included in command line arguments.')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--train_frac', type=float, default=0.8,
                        help='Fraction of full dataset to separate for training.')
    # parser.add_argument('--test_frac', type=float, default=0.15,
    #                     help='Fraction of full dataset to separate for testing.')
    parser.add_argument('--log_interval', type=int, default=5, help="Number of iterations between printing metrics.")
    # parser.add_argument('--padding_index', type=int, default=0,
    #                     help="Pos. int. value to use as padding when collating input batches.")

    # Parse command line arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # Parse and print command line arguments for model configurations
    args = parse_arguments()

    print(f"Configuration: {args}")

    # Train model
    train(**vars(args))
