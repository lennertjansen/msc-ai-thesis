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
from nltk.corpus import stopwords
import re
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from tokenizers import Vocabulary
from itertools import islice

class BlogDataset(Dataset):
    '''
    Text dataset type. Inherits functionality from data.Dataset.
    '''

    def __init__(self, file_path = 'data/blogs_kaggle/blogtext.csv',
                 transform = None, freq_threshold = 4):
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
        self.df = self.df[:1] #TODO; remove this when done testing.
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

        # Add labels for age categories
        def age_to_cat(age):
            '''Returns age category label for given age number.'''

            if 13 <= int(age) <= 17:
                return 0 #'13-17'
            elif 23 <= int(age) <= 27:
                return 1 #'23-27'
            elif 33 <= int(age) <= 47:
                return 2 #'33-47'
            else:
                raise ValueError("Given age not in one of pre-defined age groups.")

        self.df['age_cat'] = self.df['age'].apply(age_to_cat)

        # TODO: make this dynamic.
        self.num_classes = 3




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
        # target = self.df.age[index]
        target = self.df.age_cat[index]

        # start and end numericalized/embedded blog with respective special
        # tokens. Numericalize content
        numericalized_blog = [self.vocab.stoi["<BOS>"]] \
                             + self.vocab.numericalize(blog)\
                             + [self.vocab.stoi["<BOS>"]]

        return torch.tensor(numericalized_blog), torch.tensor(target)


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

#TODO: write a function get_datasets() that handles the splitting of train, val
# test sets etc. and returns the desired sets
def get_datasets():
    """
    Args
    ----
    ....

    Returns
    -------
    desired datasets
    """

    dataset = BlogDataset()

    return dataset


if __name__ == "__main__":

    # create dataset instance
    dataset = BlogDataset()

    # TODO: add colate function for batching that also returns lengths
    data_loader = DataLoader(dataset, batch_size = 2, collate_fn = padded_collate)

    for a in islice(data_loader, 10):
        print(a)

    set_trace()