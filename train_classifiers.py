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

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter # for logging


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
                    log_interval,
                    losses,
                    accs):

    # set model to train mode
    model.train()

    # print("\n Starting to train next epoch ... \n")

    for iteration, (batch_inputs, batch_labels, batch_lengths) in tqdm(enumerate(data_loader, start=start_iteration)):




        # move everything to device
        batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                    batch_lengths.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward pass through model
        log_probs = model(batch_inputs, batch_lengths)  # log_probs shape: (batch_size, num_classes)

        # Evaluate loss, gradients, and update network parameters
        loss = criterion(log_probs, batch_labels)

        writer.add_scalar("Loss/train", loss, iteration)
        # Reset gradients for next iteration
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=max_norm)

        optimizer.step()

        predictions = torch.argmax(log_probs, dim=1, keepdim=True)
        correct = predictions.eq(batch_labels.view_as(predictions)).sum().item()
        accuracy = correct / log_probs.size(0)
        writer.add_scalar("Loss/train", loss, iteration)

        # print(loss.item())
        #
        # print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
        #                    Examples/Sec = {:.2f}, "
        #       "Accuracy = {:.2f}, Loss = {:.3f}".format(
        #     datetime.now().strftime("%Y-%m-%d %H:%M"), step,
        #     config.train_steps, config.batch_size, examples_per_second,
        #     accuracy, loss
        # ))

        losses.append(loss.item())
        accs.append(accuracy)
        writer.add_scalar('Accuracy/train', accuracy, iteration)


        if iteration % log_interval == 0:

            print(
                "\n [{}] Iteration {} | Batch size = {} |"
                "Average loss = {:.6f} | Accuracy = {:.4f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), iteration,
                    batch_labels.size(0), loss.item(), accuracy
                )
            )

    return iteration, losses, accs



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
          val_frac,
          test_frac,
          subset_size,
          log_interval):

    # set seed for reproducibility on cpu or gpu based on availability
    torch.manual_seed(seed) if device == 'cpu' else torch.cuda.manual_seed(seed)

    # set starting time of full training pipeline
    start_time = datetime.now()

    # set device
    device = torch.device(device)
    print(f"Device: {device}")

    print("Starting data preprocessing ... ")
    data_prep_start = datetime.now()

    # Load data and create dataset instances
    train_dataset, val_dataset, test_dataset = get_datasets(subset_size=subset_size,
                                                            train_frac=train_frac,
                                                            val_frac=val_frac,
                                                            test_frac=test_frac,
                                                            seed=seed)

    # Train, val, and test splits
    # train_size = int(train_frac * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # TODO: adapt the get_datasets() functions s.t. the dataset splits are instances of BlogDataset
    # TODO initialize the LSTM with the vocab of the training set alone, so you can see how the model handles unknown tokens during testing

    # get vocab size number of classes
    vocab_size = train_dataset.vocab_size
    num_classes = train_dataset.num_classes

    # create dataloaders with pre-specified batch size
    # data_loader = DataLoader(dataset=dataset,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          collate_fn=PadSequence())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=PadSequence())

    val_loader = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=PadSequence())

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              collate_fn=PadSequence())

    print(f'Data preprocessing finished. Data prep took {datetime.now() - data_prep_start}.')

    print('######### DATA STATS ###############')
    print(f'Number of classes: {num_classes}')
    print(f'Vocabulary size: {vocab_size}')
    print(f'Training set size: {train_dataset.__len__()}')
    print(f'Validation set size: {val_dataset.__len__()}')
    print(f'Test set size: {test_dataset.__len__()}')
    print(81 * '#')

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

    # values for model selection
    best_val_loss = torch.tensor(np.inf, device=device)
    best_epoch = None
    best_model = None

    # metrics for losses
    train_losses = []
    train_accs = []

    for epoch in tqdm(range(epochs)):

        epoch_start_time = datetime.now()
        try:
            # set model to training mode. NB: in the actual training loop later on, this
            # statement goes at the beginning of each epoch.
            model.train()
            iterations, train_losses, train_accs = train_one_epoch(model=model, data_loader=train_loader, criterion=criterion,
                                         optimizer=optimizer, device=device, start_iteration=iterations,
                                         clip_grad=clip_grad, max_norm=max_norm, log_interval=log_interval,
                                         losses=train_losses, accs=train_accs)

        except KeyboardInterrupt:
            print("Manually stopped current epoch")
            __import__('pdb').set_trace()

        print("Current epoch training took {}".format(datetime.now() - epoch_start_time))

        val_loss, val_accuracy = evaluate_performance(model=model,
                                                        data_loader=val_loader,
                                                        device=device,
                                                        criterion=criterion)

        print(f"#######################################################################")
        print(f"Epoch {epoch + 1} finished, validation loss: {val_loss}, val acc: {val_accuracy}")
        print(f"#######################################################################")

        # update best performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
            best_epoch = epoch + 1

    print(f"#######################################################################")
    print(f"Done training and validating. Best model from epoch {best_epoch}:")
    print(best_model)
    print(f"#######################################################################")

    print("Starting testing...")
    _, _ = evaluate_performance(model=best_model, data_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion,
                                                  set='test')

    plot_performance(losses=train_losses, accs=train_accs)




def evaluate_performance(model, data_loader, device, criterion, set='validation'):

    # set model to evaluation mode
    model.eval()

    # initialize loss and number of correct predictions
    set_loss = 0
    total_correct = 0

    with torch.no_grad():
        for iteration, (batch_inputs, batch_labels, batch_lengths) in enumerate(data_loader):

            # move everything to device
            batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                        batch_lengths.to(device)

            # forward pass through model
            log_probs = model(batch_inputs, batch_lengths)  # log_probs shape: (batch_size, num_classes)

            # compute and sum up batch loss and correct predictions
            set_loss += criterion(log_probs, batch_labels)

            predictions = torch.argmax(log_probs, dim=1, keepdim=True)
            total_correct += predictions.eq(batch_labels.view_as(predictions)).sum().item()

        # average losses and accuracy
        set_loss /= len(data_loader.dataset)
        accuracy = total_correct / len(data_loader.dataset)

        print(
            "Performance on " + set + " set: "
            "Average loss: {:.4f}, Accuracy: {}/{} ({:.4f})".format(
                set_loss, total_correct, len(data_loader.dataset), accuracy
            )
        )

        return set_loss, accuracy


def plot_performance(losses, accs, show=False, save=False):
    # saving destiation
    FIGDIR = './figures/'
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 12))
    fig.suptitle(f"LSTM accuracy and loss for default settings.")

    # accs_run, loss_run, steps_run = train(config, seed=seed)
    # accs_runs.append(accs_run)
    # loss_runs.append(loss_run)
    # steps_runs.append(steps_run)
    #
    # accs_means = np.mean(accs_runs, axis=0)
    # accs_stds = np.std(accs_runs, axis=0)
    # ci = 1.96 * accs_stds / np.sqrt(num_runs)
    #
    # loss_means = np.mean(loss_runs, axis=0)
    # loss_stds = np.std(loss_runs, axis=0)
    # ci_loss = 1.96 * loss_stds / np.sqrt(num_runs)

    steps = np.arange(len(losses))

    ax1.plot(steps, accs, label="Average accuracy")
    # ax1.fill_between(steps_runs[0], (accs_means - ci), (accs_means + ci), alpha=0.3)
    ax1.set_title("Accuracy and Loss for character prediction.")
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("steps")

    ax2.plot(steps, losses, label="Average CE Loss")
    # ax2.fill_between(steps_runs[0], (loss_means - ci_loss),
    #                  (loss_means + ci_loss), alpha=0.3)
    ax2.set_title("CE Loss for various sequence lengths")
    ax2.set_ylabel("loss")
    ax2.set_xlabel("steps")

    ax1.legend()
    ax2.legend()

    if save:
        plt.savefig(f"{FIGDIR}lstm_blog.png",
                    bbox_inches='tight')
    if show:
        plt.show()


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
    parser.add_argument('--train_frac', type=float, default=0.7,
                        help='Fraction of full dataset to separate for training.')
    parser.add_argument('--val_frac', type=float, default=0.1,
                        help='Fraction of full dataset to separate for training.')
    parser.add_argument('--test_frac', type=float, default=0.2,
                        help='Fraction of full dataset to separate for testing.')
    parser.add_argument('--subset_size', type=int, default=None,
                        help='Number of datapoints to take as subset. If None, full dataset is taken.')
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

    # Create detailed experiment tag for tensorboard summary writer

    cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
    log_dir = f'runs/blog_lstm_emb_{args.embedding_dim}_hid_{args.hidden_dim}_l_{args.num_layers}_' \
              f'bd_{args.bidirectional}_drop_{args.dropout}_bs_{args.batch_size}_epochs_{args.epochs}_' \
              f'subset_{args.subset_size}_train_{args.train_frac}_val_{args.val_frac}_test_{args.test_frac}_' \
              f'clip_{args.clip_grad}_maxnorm_{args.max_norm}_seed_{args.seed}_device_{args.device}_' \
              f'{cur_datetime}'

    writer = SummaryWriter(log_dir)

    # Train model
    train(**vars(args))

    writer.close()
