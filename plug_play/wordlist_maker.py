from collections import Counter
from string import punctuation
import nltk
import pandas as pd
import numpy as np
import pdb



def df_to_wordlist(df, top_k=None, age=None):

    # set of stopwords
    stopwords = set(nltk.corpus.stopwords.words('english')) # 0(1) lookups

    # frequency counter without stopwords
    without_stp  = Counter()

    # remove apostrophe (') from punctuation so tokens like "i'm" stay
    punc = punctuation.translate(str.maketrans('', '', "'"))
    numbers = '0123456789'
    # any(p in ts for p in punctuation)

    # extract text from dataframe
    if not age:
        text = df.text
    else:
        text = df.text[df.label == age]

    for line in text:
        # split line into tokens
        spl = line.split()

        # update count off all words in the line that are not in stopwords
        without_stp.update(w.lower().rstrip(punctuation) for w in spl if w not in stopwords and not any(p in w for p in punctuation) and not any(p in w for p in numbers))

    # return a list with top ten most common words from each
    if top_k:
        return [word for word, count in without_stp.most_common(top_k)]
    else:
        return [word for word, count in without_stp.most_common()]



if __name__ == '__main__':

    bnc_rb_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0_rand_balanced.csv'
    bnc_path = 'data/bnc/bnc_subset_19_29_vs_50_plus_nfiles_0.csv'
    print('Loading data...')
    df = pd.read_csv(bnc_rb_path, encoding="utf-8")  # to keep no. unique chars consistent across platforms

    most_common_words = df_to_wordlist(df, top_k=500)

    with open('plug_play/wordlists/bnc_rb_500_most_common.txt', 'w') as f:
        for word in most_common_words:
            f.write("%s\n" % word)

    df_full = pd.read_csv(bnc_path, encoding="utf-8")  # to keep no. unique chars consistent across platforms
    mcw_young = df_to_wordlist(df_full, age='19_29')
    mcw_old = df_to_wordlist(df_full, age='50_plus')

    cutoff = 3000



    mcw_young_unique = np.setdiff1d(mcw_young[:cutoff], mcw_old[:cutoff])
    mcw_old_unique = np.setdiff1d(mcw_old[:cutoff], mcw_young[:cutoff])

    print(len(mcw_young_unique))
    print(len(mcw_old_unique))

    with open('plug_play/wordlists/bnc_young_mcwu.txt', 'w') as f:
        for word in mcw_young_unique:
            f.write("%s\n" % word)

    with open('plug_play/wordlists/bnc_old_mcwu.txt', 'w') as f:
        for word in mcw_old_unique:
            f.write("%s\n" % word)

    pdb.set_trace()

