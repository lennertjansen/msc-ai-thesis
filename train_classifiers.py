import argparse
import pdb
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

from copy import deepcopy
import shutil
from pathlib import Path
import os, glob

import pandas as pd


# General teuxdeuxs


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
                    accs,
                    writer,
                    disable_bars):

    # set model to train mode
    model.train()

    # print("\n Starting to train next epoch ... \n")

    for iteration, (batch_inputs, batch_labels, batch_lengths) in tqdm(enumerate(data_loader, start=start_iteration),
                                                                       disable=disable_bars):




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
        # writer.add_scalar("Loss/train", loss, iteration)

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
          data,
          mode,
          device,
          batch_size,
          embedding_dim,
          hidden_dim,
          num_layers,
          bidirectional,
          dropout,
          batch_first,
          epochs,
          lr,
          clip_grad,
          max_norm,
          early_stopping_patience,
          train_frac,
          val_frac,
          test_frac,
          subset_size,
          log_interval,
          writer=None,
          train_dataset=None,
          val_dataset=None,
          test_dataset=None):


    if mode =='train' or mode == 'test':
        # set seed for reproducibility on cpu or gpu based on availability
        torch.manual_seed(seed) if device == 'cpu' else torch.cuda.manual_seed(seed)

        data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv' if data == 'bnc' else 'data/blogs_kaggle/blogtext.csv'

        # set starting time of full training pipeline
        start_time = datetime.now()

        # set device
        device = torch.device(device)
        print(f"Device: {device}")

        print("Starting data preprocessing ... ")
        data_prep_start = datetime.now()

        # Load data and create dataset instances
        train_dataset, val_dataset, test_dataset = get_datasets(subset_size=subset_size,
                                                                file_path=data_path,
                                                                train_frac=train_frac,
                                                                val_frac=val_frac,
                                                                test_frac=test_frac,
                                                                seed=seed,
                                                                data=data)

        # Train, val, and test splits
        # train_size = int(train_frac * len(dataset))
        # test_size = len(dataset) - train_size
        # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


        # # get vocab size number of classes
        # vocab_size = train_dataset.vocab_size
        # num_classes = train_dataset.num_classes

        # create dataloaders with pre-specified batch size
        # data_loader = DataLoader(dataset=dataset,
        #                          batch_size=batch_size,
        #                          shuffle=True,
        #                          collate_fn=PadSequence())


        print(f'Data preprocessing finished. Data prep took {datetime.now() - data_prep_start}.')

        print('######### DATA STATS ###############')
        print(f'Number of classes: {train_dataset.num_classes}')
        print(f'Vocabulary size: {train_dataset.vocab_size}')
        print(f'Training set size: {train_dataset.__len__()}')
        print(f'Validation set size: {val_dataset.__len__()}')
        print(f'Test set size: {test_dataset.__len__()}')
        print(81 * '#')

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

    if mode == 'train' or mode == 'val':

        # initialize model
        print("Initializing model ...")
        model = TextClassificationLSTM(batch_size = batch_size,
                                       vocab_size = train_dataset.vocab_size,
                                       embedding_dim = embedding_dim,
                                       hidden_dim = hidden_dim,
                                       num_classes = train_dataset.num_classes,
                                       num_layers = num_layers,
                                       bidirectional = bidirectional,
                                       dropout = dropout,
                                       device = device,
                                       batch_first = batch_first)
    elif mode == 'test':
        model, _, _, _, _ = load_saved_model(model_class=TextClassificationLSTM, optimizer_class=optim.Adam, lr=lr,
                                             device=device, batch_size=batch_size, vocab_size=train_dataset.vocab_size,
                                             embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                             num_classes=train_dataset.num_classes, num_layers=num_layers,
                                             bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)


    # model to device
    model.to(device)

    # Print model architecture and trainable parameters
    print("MODEL ARCHITECTURE:")
    print(model)

    if data == 'bnc':
        n_samples = [train_dataset.df['age_cat'].value_counts()[0],
                    train_dataset.df['age_cat'].value_counts()[1]]
        normed_weights = [1 - (x / sum(n_samples)) for x in n_samples]
        weights = [1, 4]
        normed_weights = torch.FloatTensor(weights).to(device)

        criterion = torch.nn.CrossEntropyLoss(weight=normed_weights)  # combines LogSoftmax and NLL
    else:
        criterion = torch.nn.CrossEntropyLoss()  # combines LogSoftmax and NLL

    if mode == 'train' or mode == 'val':

        # count trainable parameters
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print(f'The model has {trainable_params} trainable parameters.')

        # set up optimizer and loss criterion
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        # initialize iterations at zero
        iterations = 0

        # values for model selection
        best_val_loss = torch.tensor(np.inf, device=device)
        best_val_accuracy = torch.tensor(-np.inf, device=device)
        best_epoch = None
        best_model = None

        # Initialize patience for early stopping
        patience = 0

        # metrics for losses
        train_losses = []
        train_accs = []

        # disable tqdm progress bars in train and train_one_epoch if in validation mode
        disable_bars = mode == 'val'

        for epoch in tqdm(range(epochs), disable=disable_bars):

            epoch_start_time = datetime.now()
            try:
                # set model to training mode. NB: in the actual training loop later on, this
                # statement goes at the beginning of each epoch.
                model.train()
                iterations, train_losses, train_accs = train_one_epoch(model=model, data_loader=train_loader,
                                                                       criterion=criterion,
                                                                       optimizer=optimizer, device=device,
                                                                       start_iteration=iterations,
                                                                       clip_grad=clip_grad, max_norm=max_norm,
                                                                       log_interval=log_interval,
                                                                       losses=train_losses, accs=train_accs, writer=writer,
                                                                       disable_bars=disable_bars)

            except KeyboardInterrupt:
                print("Manually stopped current epoch")
                __import__('pdb').set_trace()

            print("Current epoch training took {}".format(datetime.now() - epoch_start_time))

            val_loss, val_accuracy = evaluate_performance(model=model,
                                                          data_loader=val_loader,
                                                          device=device,
                                                          criterion=criterion,
                                                          writer=writer,
                                                          iteration=iterations)

            print(f"#######################################################################")
            print(f"Epoch {epoch + 1} finished, validation loss: {val_loss}, val acc: {val_accuracy}")
            print(f"#######################################################################")

            # # update best performance
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     best_val_accuracy = val_accuracy
            #     best_model = model
            #     best_epoch = epoch + 1

            # update best performance
            if val_accuracy > best_val_accuracy:
                best_val_loss = val_loss
                best_val_accuracy = val_accuracy
                best_model = deepcopy(model)
                best_optimizer = deepcopy(optimizer)
                best_epoch = epoch + 1
                patience = 0
            else:
                patience +=1
                if patience >= early_stopping_patience:
                    print("EARLY STOPPING")
                    break

        print(f"#######################################################################")
        print(f"Done training and validating. Best model from epoch {best_epoch}:")
        print(best_model)
        print(f"#######################################################################")

    if mode == 'val':
        return best_val_loss, best_val_accuracy, best_model, best_epoch, best_optimizer
    elif mode == 'train':
        print("Starting testing...")
        _, _ = evaluate_performance(model=best_model, data_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion,
                                                  set='test')
    elif mode == 'test':
        print("Starting testing...")
        _, _ = evaluate_performance(model=model, data_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion,
                                                  set='test')

    # plot_performance(losses=train_losses, accs=train_accs)





def evaluate_performance(model, data_loader, device, criterion, writer=None, iteration=0, set='validation'):

    # set model to evaluation mode
    model.eval()

    # initialize loss and number of correct predictions
    set_loss = 0
    total_correct = 0

    with torch.no_grad():
        for iteration, (batch_inputs, batch_labels, batch_lengths) in tqdm(enumerate(data_loader)):

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

        if set == 'validation':
            writer.add_scalar('Accuracy/val', accuracy, iteration)
            writer.add_scalar('Loss/val', set_loss, iteration)

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


def hp_search(seed,
              data,
              mode,
              device,
              batch_size,
              embedding_dim,
              hidden_dim,
              num_layers,
              bidirectional,
              dropout,
              batch_first,
              epochs,
              lr,
              clip_grad,
              max_norm,
              early_stopping_patience,
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

    data_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv' if data == 'bnc' else 'data/blogs_kaggle/blogtext.csv'

    print("Starting data preprocessing ... ")
    data_prep_start = datetime.now()

    # Load data and create dataset instances
    train_dataset, val_dataset, test_dataset = get_datasets(subset_size=subset_size,
                                                            file_path=data_path,
                                                            train_frac=train_frac,
                                                            val_frac=val_frac,
                                                            test_frac=test_frac,
                                                            seed=seed,
                                                            data=data)

    print(100*"{}")
    print('BASELINES//VALUE COUNTS')
    print('Train')
    print(train_dataset.df['age_cat'].value_counts(normalize=True))
    print('Validation')
    print(val_dataset.df['age_cat'].value_counts(normalize=True))
    print(100 * "{}")

    # Train, val, and test splits
    # train_size = int(train_frac * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # get vocab size number of classes
    # vocab_size = train_dataset.vocab_size
    # num_classes = train_dataset.num_classes

    # create dataloaders with pre-specified batch size
    # data_loader = DataLoader(dataset=dataset,
    #                          batch_size=batch_size,
    #                          shuffle=True,
    #                          collate_fn=PadSequence())



    print(f'Data preprocessing finished. Data prep took {datetime.now() - data_prep_start}.')

    print('######### DATA STATS ###############')
    print(f'Number of classes: {train_dataset.num_classes}')
    print(f'Vocabulary size: {train_dataset.vocab_size}')
    print(f'Training set size: {train_dataset.__len__()}')
    print(f'Validation set size: {val_dataset.__len__()}')
    print(f'Test set size: {test_dataset.__len__()}')
    print(81 * '#')

    # Set hyperparameters for grid search*
    # seeds = [0, 1, 2]
    lrs = [1e-5, 1e-3, 1e-1]
    embedding_dims = [64, 256, 512]
    hidden_dims = [128, 512, 1024]
    nums_layers = [1, 2]
    bidirectionals = [False, True]


    # set holders for best performance metrics and corresponding hyperparameters
    best_metrics = {'loss' : float("inf"),
                    'acc' : float('-inf')}
    best_hps = {'lr' : None,
                'embedding_dim' : None,
                'hidden_dim' : None,
                'num_layers': None,
                'bidirectional' : None}

    best_model = None # TODO: what's the appropriate type for this?

    #TODO: add tqdm's and print statements to these loops for progress monitoring

    best_file_name = None
    best_epoch = None

    # For keeping track of metrics for all configs
    keys = ['lr', 'emb_dim', 'hid_dim', 'n_layers', 'bd', 'val_acc', 'val_loss']
    df = pd.DataFrame(columns=keys)

    best_model_updates = -1

    for lr_ in tqdm(lrs, position=0, leave=True, desc='Learning rates'):
        for emb_dim in tqdm(embedding_dims, position=0, leave=True, desc='Embedding dims'):
            for hid_dim in tqdm(hidden_dims, position=0, leave=True, desc='Hidden dims'):
                # skip if hidden size not larger than embedding dim
                if not hid_dim > emb_dim:
                    continue

                for n_layers in tqdm(nums_layers, position=0, leave=True, desc='No. layers'):
                    for bd in tqdm(bidirectionals, position=0, leave=True, desc='Bidirectional'):

                        print(f"Current config: lr: {lr_} | emb: {emb_dim} | hid_dim: {hid_dim} | n_layers: {n_layers} "
                              f"| bd: {bd} | ")

                        # Create detailed experiment tag for tensorboard summary writer
                        cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
                        log_dir = f'runs/hp_search/{data}/'
                        file_name = f'lstm_emb_{emb_dim}_hid_{hid_dim}_l_{n_layers}_' \
                                    f'bd_{bd}_drop_{dropout}_bs_{batch_size}_epochs_{epochs}_' \
                                    f'lr_{lr_}_subset_{subset_size}_train_{train_frac}_val_{val_frac}_' \
                                    f'test_{test_frac}_clip_{clip_grad}_maxnorm_{max_norm}' \
                                    f'es_{early_stopping_patience}_seed_{seed}_device_{device}_dt_{cur_datetime}'

                        # create summary writer instance for logging
                        log_path = log_dir+file_name
                        writer = SummaryWriter(log_path)

                        # train model (in val mode)
                        loss, acc, model, epoch, optimizer = train(mode=mode, data=data, seed=seed, device=device,
                                                                   batch_size=batch_size, embedding_dim=emb_dim,
                                                                   hidden_dim=hid_dim, num_layers=n_layers,
                                                                   bidirectional=bd, dropout=dropout,
                                                                   batch_first=batch_first, epochs=epochs,
                                                                   lr=lr_, clip_grad=clip_grad, max_norm=max_norm,
                                                                   early_stopping_patience=early_stopping_patience,
                                                                   train_frac=train_frac, val_frac=val_frac,
                                                                   test_frac=test_frac, subset_size=subset_size,
                                                                   log_interval=log_interval, writer=writer,
                                                                   train_dataset=train_dataset, val_dataset=val_dataset,
                                                                   test_dataset=test_dataset
                                                                   )

                        # close tensorboard summary writer
                        writer.close()


                        # Update metric logging dataframe
                        df.loc[0 if pd.isnull(df.index.max()) else df.index.max() + 1] = [lr_] + [emb_dim] + [hid_dim] \
                                                                                         + [n_layers] + [bd] + [acc] + \
                                                                                         [loss.item()]

                        # Save metric logging dataframe to csv
                        df.to_csv(
                            f'output/{data}_lstm_hp_search_metrics.csv',
                            index=False
                        )

                        # update best ...
                        if acc > best_metrics['acc']:

                            best_model_updates +=1

                            # ... metrics
                            best_metrics['acc'] = acc
                            best_metrics['loss'] = loss

                            best_epoch = epoch

                            # ... hyperparams
                            best_hps['lr'] = lr_
                            best_hps['embedding_dim'] = emb_dim
                            best_hps['hidden_dim'] = hid_dim
                            best_hps['num_layers'] = n_layers
                            best_hps['bidirectional'] = bd

                            # ... model
                            best_model = deepcopy(model)

                            # ... optimizer
                            best_optimizer = deepcopy(optimizer)

                            # filename
                            best_file_name = file_name

                            # Delete previous current best model checkpoint file
                            for filename in glob.glob(f"models/{data}/lstm/cur_best_*"):
                                os.remove(filename)

                                # save current best model checkpoint
                            # Save best model checkpoint
                            model_dir = f'models/{data}/lstm/'
                            Path(model_dir).mkdir(parents=True, exist_ok=True)
                            model_path = model_dir + 'cur_best_' + best_file_name + '.pt'

                            torch.save({
                                'epoch': best_epoch,
                                'model_state_dict': best_model.state_dict(),
                                'optimizer_state_dict': best_optimizer.state_dict(),
                                'loss': best_metrics['loss'],
                                'acc': best_metrics['acc']
                            }, model_path)

                            print("New current best model found.")
                            print(f'Current best hyperparameters: {best_hps}')
                            print(f'Current best model: {best_model}')
                            print(f'Current best metrics: {best_metrics}')

    # # Save metric logging dataframe to csv
    # df.to_csv(
    #     'output/blog_lstm_hp_search_metrics.csv',
    #     index=False
    # )

    # Save best model checkpoint
    model_dir = f'models/{data}/lstm/'
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = model_dir + 'best_' + best_file_name + '.pt'

    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': best_optimizer.state_dict(),
        'loss': best_metrics['loss'],
        'acc': best_metrics['acc']
    }, model_path)

    print("Finished hyperparameter search.")
    print(f'Best hyperparameters: {best_hps}')
    print(f'Best model: {best_model}')
    print(f'Best metrics: {best_metrics}')

    # Delete equivalent cur_best file
    for filename in glob.glob("models/blog/lstm/cur_best_*"):
        os.remove(filename)

    print(f"Best model updates: {best_model_updates}")



def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def load_saved_model(model_class, optimizer_class, lr, device, batch_size, vocab_size, embedding_dim, hidden_dim,
                     num_classes, num_layers, bidirectional, dropout, batch_first):

    checkpoint_path = 'models/blog/lstm/best_blog_lstm_emb_128_hid_256_l_2_bd_True_drop_0_bs_64_epochs_5_lr_0.001_' \
                      'subset_None_train_0.75_val_0.15_test_0.1_clip_False_maxnorm_10.0es_2_seed_2021_' \
                      'device_cuda_dt_13_May_2021_16_25_34.pt'

    # initialize model instance
    model = model_class(batch_size=batch_size, vocab_size=vocab_size, embedding_dim=embedding_dim,
                        hidden_dim=hidden_dim, num_classes=num_classes, num_layers=num_layers,
                        bidirectional=bidirectional, dropout=dropout, device=device, batch_first=batch_first)

    # model to device
    model.to(device)

    # initialize optimizer
    optimizer = optimizer_class(params=model.parameters(), lr=lr)

    # load checkpoint
    checkpoint = torch.load(checkpoint_path)

    # {
    #     'epoch': best_epoch,
    #     'model_state_dict': best_model.state_dict(),
    #     'optimizer_state_dict': best_optimizer.state_dict(),
    #     'loss': best_metrics['loss'],
    #     'acc': best_metrics['acc']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    acc = checkpoint['acc']

    # set_trace()

    return model, optimizer, epoch, loss, acc


def parse_arguments(args = None):
    parser = argparse.ArgumentParser(description="Train discriminator neural text classifiers.")

    parser.add_argument(
        '--data', type=str, choices=['blog', 'bnc'], default='blog',
        help='Choose dataset to work with. Either blog corpus or BNC.'
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'val', 'test'], default='train',
        help='Set script to training, development/validation, or test mode.'
    )
    parser.add_argument(
        "--seed", type=int, default=2021, help="Seed for reproducibility"
    )
    parser.add_argument(
        '--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
        help="Device to run the model on."
    )
    parser.add_argument(
        '--batch_size', type=int, default=1, help="Number of datapoints to simultaneously process."
    )  # TODO: set to reasonable default after batching problem fixed.
    parser.add_argument(
        '--embedding_dim', type=int, default=16, help="Dimensionality of embedding."
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=32, help="Size in LSTM hidden layer."
    )
    parser.add_argument(
        '--num_layers', type=int, default=1, help='Number of hidden layers in LSTM'
    )
    parser.add_argument(
        '--bidirectional', action='store_true',
        help='Create a bidirectional LSTM. LSTM will be unidirectional if omitted.'
    )
    parser.add_argument(
        '--dropout', type=float, default=0, help='Probability of applying dropout in final LSTM layer.'
    )
    parser.add_argument(
        '--batch_first', action='store_true', help='Assume batch size is first dimension of data.'
    )
    parser.add_argument(
        '--epochs', type=int, default=1, help='Number of passes through entire dataset during training.'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3, help='Adam optimizer learning rate.'
    )
    parser.add_argument(
        '--clip_grad', action='store_true',
        help = 'Apply gradient clipping. Set to True if included in command line arguments.'
    )
    parser.add_argument(
        '--max_norm', type=float, default=10.0
    )
    parser.add_argument(
        '-es', '--early_stopping_patience', type=int, default=3, help="Early stopping patience. Default: 3"
    )
    parser.add_argument(
        '--train_frac', type=float, default=0.75, help='Fraction of full dataset to separate for training.'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.15, help='Fraction of full dataset to separate for training.'
    )
    parser.add_argument(
        '--test_frac', type=float, default=0.10, help='Fraction of full dataset to separate for testing.'
    )
    parser.add_argument(
        '--subset_size', type=int, default=None,
        help='Number of datapoints to take as subset. If None, full dataset is taken.'
    )
    parser.add_argument(
        '--log_interval', type=int, default=5, help="Number of iterations between printing metrics."
    )
    # parser.add_argument('--padding_index', type=int, default=0,
    #                     help="Pos. int. value to use as padding when collating input batches.")

    # Parse command line arguments
    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # Parse and print command line arguments for model configurations
    args = parse_arguments()

    print(f"Configuration: {args}")

    if args.mode == 'train':
        print("Starting training mode...")
        # Create detailed experiment tag for tensorboard summary writer
        cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')

        log_dir = f'runs/{args.data}/lstm_emb_{args.embedding_dim}_hid_{args.hidden_dim}_l_{args.num_layers}_' \
                  f'bd_{args.bidirectional}_drop_{args.dropout}_bs_{args.batch_size}_epochs_{args.epochs}_' \
                  f'lr_{args.lr}_subset_{args.subset_size}_train_{args.train_frac}_val_{args.val_frac}_' \
                  f'test_{args.test_frac}_clip_{args.clip_grad}_maxnorm_{args.max_norm}_' \
                  f'es_{args.early_stopping_patience}_seed_{args.seed}_device_{args.device}_dt_{cur_datetime}'

        writer = SummaryWriter(log_dir)

        # Train model
        train(**vars(args), writer=writer)

        # close tensorboard summary writer
        writer.close()

    elif args.mode == 'val':
        print("Starting validation/development mode...")

        # hyper parameter search
        hp_search(**vars(args))

    else:
        train(**vars(args))
