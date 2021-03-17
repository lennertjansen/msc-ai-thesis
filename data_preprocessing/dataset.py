# Handles pre-processing of data and implementation of dataset type.
#
# Manual Vocabulary based on: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/custom_dataset_txt/loader_customtext.py
# Date created: 1 March 2021
################################################################################
# What do I want to achieve?
# Main goal: Convert text to numerical values
# To do so, wee need:
# 1. A vocabulary that maps every word to an index
# 2. To set up a PyTorch Dataset type to load the data
# 3. Set up padding for each batch so every sequence in a batch has equal length


# Import statements
from pdb import set_trace # for easier debugging
import os # for path file reading/loading
import numpy as np # for linear algebra and numerical methods
import pandas as pd # for easier csv parsing
from torch import is_tensor
from torch.utils.data import Dataset, DataLoader
import spacy # for tokenizer
# TODO: use huggingface tokenizers and experiment with BPE, WordPiece, and SentencePiece encodings: https://huggingface.co/transformers/tokenizer_summary.html
from nltk.corpus import stopwords
import re
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch

# spacy english tokenizer
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    '''
    Manual vocabulary using spacy tokenizer.
    '''

    def __init__(self, freq_threshold):

        # integer to string mappings (and vice versa) with tokens for padding,
        # start of sentence, end of sentence, and unknown
        self.itos = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"} # int to str dict
        self.stoi = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3} # str to int dict

        # frequency threshold for token to be added to vocabulary
        self.freq_threshold = freq_threshold

    def __len__(self):
        ''' Returns length (aka size) of vocabulary.'''
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}

        index = 4 # start integer index at 4 because 0 through 3 are special tokens

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                # add dict entry for word if not yet in dict, else increment
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                # add word to vocab if it passes frequency threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = index
                    self.itos[index] = word
                    index += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]




class BlogDataset(Dataset):
    '''
    Text dataset type. Inherits functionality from data.Dataset.
    '''

    def __init__(self, file_path = 'data/blogs_kaggle/blogtext.csv',
                 transform = None, freq_threshold = 1):
        """
        Args:
            file_path (string): Path to csv file with blogtext data and labels.
            transform (callable, optional): Optional transform to be applied on
                a sample.
            freq_threshold (int): Required number of occurrences of a token for
                it to be added to the vocabulary
        """

        # check if file csv format
        assert os.path.splitext(file_path)[1] == ".csv"

        # read csv as dataframe
        self.df = pd.read_csv(file_path, encoding="utf-8") # to keep no. unique chars consistent across platforms
        self.df = self.df[:1000] #TODO; remove this when done testing.
        self.data_size = len(self.df)
        self.transform = transform

        #TODO Pre-processing steps happen here -- implement them st the less
        # standard ones can be switched on and off to evaluate their impact on
        # performance:
        # Remove all non-alphabetical characters
        self.df['clean_text'] = self.df['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))

        # make all letters lowercase
        self.df['clean_text'] = self.df['clean_text'].apply(lambda x: x.lower())

        # remove whitespaces from beginning or ending
        self.df['clean_text'] = self.df['clean_text'].apply(lambda x: x.strip())

        # remove stop words
        stopwords_dict = set(stopwords.words('english')) # use set (hash table) data structure for faster lookup
        self.df['clean_text'] = self.df['clean_text'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords_dict]))

        # Remove instances empty strings
        self.df.drop(self.df[self.df.clean_text == ''].index, inplace = True)

        # TODO: filter out non-English

        # TODO: mask age-disclosing utterances

        # number of datapoints removed by all pre-processing steps
        self.dropped_instances = self.data_size - len(self.df)
        self.data_size = len(self.df)

        # initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.df.clean_text)

        # TODO: save age cateogry targets (maybe also astrological sign as a joke)
        self.age_exact = self.df['age']


    def __len__(self):
        '''
        Overrides the inherited method s.t. len(dataset) returns the size of the
        data set.
        '''
        return self.data_size

    def __getitem__(self, index):
        '''
        Overrides the inherited method s.t. indexing dataset[i] will get the
        i-th sample. Also handles reading of examples (as opposed to init()),
        which is more memory efficient, as all examples are not stored in
        memory at once, but as they are required.
        '''

        # Convert index to list if its a tensor
        if is_tensor(index):
            index = index.tolist()

        # get blog at desired index
        blog = self.df.clean_text[index]

        # start and end numericalized/embedded blog with respective special
        # tokens. Numericalize content
        numericalized_blog = [self.vocab.stoi["<BOS>"]] \
                             + self.vocab.numericalize(blog)\
                             + [self.vocab.stoi["<BOS>"]]

        return torch.tensor(numericalized_blog)


class MyCollate:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, batch):

        blogs = [blog[0].unsqueeze(0) for blog in batch]
        blogs = torch.cat(blogs, dim = 0)
        blogs = pad_sequence(blogs, batch_first = False, padding_value = self.pad_index)

        return blogs

def padded_collate(batch, pad_index = 0):

    texts = batch
    lengths = [len(text) for text in texts]

    max_length = max(lengths)

    # Pad blogs
    padded_texts = [text + [pad_index] * (max_length - len(text)) for text in texts]

    return torch.LongTensor(padded_texts), lengths

# create dataset instance
dataset = BlogDataset()

data_loader = DataLoader(dataset, batch_size = 4,
                         collate_fn=MyCollate(pad_index = 0))

set_trace()
