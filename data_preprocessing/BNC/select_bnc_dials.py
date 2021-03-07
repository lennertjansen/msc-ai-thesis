
import argparse
import os
from collections import defaultdict, Counter
import pandas as pd
import spacy
from nltk import word_tokenize as nltk_tokenize

from BNC import Corpus

bnc_path = '/Users/janie/Documents/Corpora/bnc2014spoken-xml/spoken/'

corpus = Corpus(
    untagged_path=os.path.join(bnc_path, "untagged"),
    tagged_path=os.path.join(bnc_path, "tagged"),
    n=100,
    add_speaker_id=True
)

relevant_conversation_ids = []
N = 2
for conv_id, conv in corpus.conversations.items():
    if conv.n_speakers == N:
        relevant_conversation_ids.append(conv_id)
for conv_id in relevant_conversation_ids:
    for u in corpus.conversations[conv_id].utterances:
        print(u.sentence)
