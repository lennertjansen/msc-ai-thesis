import torch.nn as nn
import torch
from pdb import set_trace

class TextClassificationLSTM(nn.Module):

    def __init__(self, batch_size, vocab_size, embedding_dim, hidden_dim,
                 num_classes, num_layers, bidirectional = False,
                 dropout = 0, device = 'cpu',
                 batch_first = True):
        """Text classification LSTM module.

        Args
        ----
        batch_size (int): No. datapoints simultaneously fed through model.
        voacb_size (int): No. unique tokens in dictionary.
        embedding_dim (int): No. dimensions to represent each token with.
        hidden_dim (int): Dimensionality of hidden layer.
        num_classes (int): size of output layer.
        num_layers (int): No. layers to be stacked in LSTM.
        bidirectional (bool): If True, introduces bidirectional LSTM, i.e.,
            sequence will be processed in both directions simultaneously.
            Default: False
        dropout (float): probability of dropping out connections in all but
            last layer. Default: 0
        device (str): Hardware device to store/run model on. Default: cpu
        batch_first (bool): Processing input assuming first dimension is
            batch dimension. Default: True

        ----------------
        For inspiration:
            https://www.analyticsvidhya.com/blog/2020/01/first-text-classification-in-pytorch/
            https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
            https://towardsdatascience.com/text-classification-with-pytorch-7111dae111a6
            https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df

        """

        # Constructor #TODO: why is this necessary? And what about this:
        # super().__init__()
        super(TextClassificationLSTM, self).__init__()

        # initialize embedding layer
        self.embedding = nn.Embedding(num_embeddings = vocab_size,
                                      embedding_dim = embedding_dim)

        # useful for later in forward function
        self.batch_first = batch_first

        # initialize LSTM
        self.lstm = nn.LSTM(input_size = embedding_dim,
                            hidden_size = hidden_dim,
                            num_layers = num_layers,
                            bidirectional = bidirectional,
                            dropout = dropout,
                            batch_first = self.batch_first)

        # fully connected output layer (pre-activation)
        self.num_directions = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * self.num_directions, num_classes)

        # Activation function
        # TODO: figure out over which dimension to do this
        self.act = nn.ReLU()

    def forward(self, text, text_lengths, vidhya = False):
        """ Performs forward pass of input text through classification module.
        Args
        ----
        text (Tensor): Numericalized input text.
            dim = [batch_size, sequence_length]
        text_lengths (Tensor or list(int)): pre-padding lengths of input
            sequences. dim = [batch_size]

        Returns
        -------
        output (Tensor): TODO: log probabilities.
            dim = [batch_size, num_classes]
        """

        # embed numericalized input text
        embedded = self.embedding(text)
        # embedded dims: [batch_size, sequence_length, embedding_dim]

        # pack padded sequence
        # here's why: https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        packed_embedded = nn.utils.rnn.pack_padded_sequence(input = embedded,
                                                            lengths = text_lengths,
                                                            batch_first = self.batch_first,
                                                            enforce_sorted = False)

        # Do forward pass through lstm model
        # NB: output tuple (hidden, cell) is ignored
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # NB: the layers can be separated using h_n.view(num_layers, num_directions, batch, hidden_size)
        # cell: [num_layers * num_directions, batch_size, hidden_dim]

        if self.num_directions == 1:
            # packed_output, (final_hidden, final_cell) = self.lstm(packed_embedded)
            # take last layer of final hidden state (i.e., shape [batch_size, hidden_dim])
            # pass through linear layer and activation
            output = self.fc(hidden[-1, :, :])
            # OR: THIS
            # unpacked_output = nn.utils.rnn.pad_packed_sequence(packed_output)
            # output = unpacked_output[0]
            # output = self.fc(output[-1, :, :])
            # out[-1, :, :] and hidden[-1, :, :] are supposed to be identical
            output = self.act(output)
        else:
            # Vidhya's method
            # concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1)
            # hidden dim: [batch_size, hidden_dim * num_directions]

            output = self.fc(hidden)
            output = self.act(output)

            # # inverse operation of pack_padded_sequence(). i.e., unpacks packed
            # # sequences, returns padded sequences and corresponding lengths
            # # Cheng's blog way
            # unpacked_output, _ = nn.utils.rnn.pad_packed_sequence(sequence = packed_output,
            #                                                       batch_first = True)
            # # forward direction
            # out_forward = unpacked_output[]

        return output


class TextClassificationLogit(nn.Module):
    """
    Based on this tutorial: https://medium.com/biaslyai/pytorch-linear-and-logistic-regression-models-5c5f0da2cb9
    """
    def __init__(self, num_classes):
        super(TextClassificationLogit, self).__init__()
        self.linear = nn.Linear(1, num_classes)

    def forward(self, x):
        out = self.linear(x)
        out = nn.functional.sigmoid(out)

        return out

class TextClassificationBERT():
    """
    For inspiration: https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
    """
    pass

if __name__ == "__main__":

    set_trace()