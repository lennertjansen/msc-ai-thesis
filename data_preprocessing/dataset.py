# Handles pre-processing of data and implementation of dataset type.
#
# Date created: 1 March 2021
################################################################################
# What do I want to achieve?
# Main goal: Convert text to numerical values
# To do so, wee need:
# 1. A vocabulary that maps every word to an index
# 2. To set up a PyTorch Dataset type to load the data
# 3. Set up padding for each batch so every sequence in a batch has equal length


# Import statements
import pdb # for easier debugging
import os # for path file reading/loading
import numpy as np # for linear algebra and numerical methods
import pandas as pd # for easier csv parsing
import torch.utils.data as data # for pytorch data utilities. nb: could specify which classes to import for efficiency
import spacy # for tokenizer


class TextDataset(data.Dataset):

    '''
    Text dataset type. Inherits functionality from data.Dataset.
    '''

    def __init__(self, file_path = '../data/blogs_kaggle/blogtext.csv'):

        # check if file csv format
        assert os.path.splitext(filename)[1] == ".csv"

        # read csv as dataframe
        self.df = pd.read_csv('../data/blogs_kaggle/blogtext.csv', encoding="utf-8") # to keep no. unique chars consistent across platforms

    def __len__(self):
        pass
