from nltk.corpus import stopwords
import pandas as pd
import re

def preprocess_df(df):

    data_size = len(df)

    #TODO Pre-processing steps happen here -- implement them st the less
    # standard ones can be switched on and off to evaluate their impact on
    # performance:
    # Remove all non-alphabetical characters
    df['clean_text'] = df['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))

    # make all letters lowercase
    df['clean_text'] = df['clean_text'].apply(lambda x: x.lower())

    # remove whitespaces from beginning or ending
    df['clean_text'] = df['clean_text'].apply(lambda x: x.strip())

    # remove stop words
    stopwords_dict = set(stopwords.words('english')) # use set (hash table) data structure for faster lookup
    df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords_dict]))

    # Remove instances empty strings
    df.drop(df[df.clean_text == ''].index, inplace = True)

    # TODO: filter out non-English

    # TODO: mask age-disclosing utterances

    # number of datapoints removed by all pre-processing steps
    dropped_instances = data_size - len(df)
    data_size = len(df)



    # Add labels for age categories
    def age_to_cat(age):
        '''Returns age category label for given age number.'''

        if 13 <= int(age) <= 17:
            return 0 #'13-17'
        elif 23 <= int(age) <= 27:
            return 1 #'23-27'
        elif 33 <= int(age):
            return 2 #'33-47'
        else:
            raise ValueError("Given age not in one of pre-defined age groups.")

    df['age_cat'] = df['age'].apply(age_to_cat)

    return df[['clean_text', 'age_cat']]