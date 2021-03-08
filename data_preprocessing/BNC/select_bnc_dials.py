
import argparse
import os
from collections import defaultdict, Counter
import pandas as pd
import spacy
from nltk import word_tokenize as nltk_tokenize

from BNC import Corpus

import pdb

def main(args):

    bnc_path = args.data_path

    corpus = Corpus(
        untagged_path=os.path.join(bnc_path, "untagged"),
        tagged_path=os.path.join(bnc_path, "tagged"),
        n=args.no_files,
        add_speaker_id=True
    )

    relevant_conversation_ids = []
    N = args.no_speakers

    age_cat = args.age_cat

    for conv_id, conv in corpus.conversations.items():
        if conv.n_speakers == N:
            age_ranges_list = list(conv.speakers_age_ranges.values())
            if age_ranges_list[0] == age_cat and age_ranges_list[1] == age_cat:
                relevant_conversation_ids.append(conv_id)
    for conv_id in relevant_conversation_ids:
        print(100 * "=")
        print('|' + 98 * " " + "|")
        print('|' + 98 * " " + "|")
        print('|' + 98 * " " + "|")
        print('|' + 98 * " " + "|")
        print(100 * "=")
        print(f"Speaker age ranges: {corpus.conversations[conv_id].speakers_age_ranges}")
        for u in corpus.conversations[conv_id].utterances:
            print(u.sentence)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--age_cat', type = str, default = '30-39',
        choices = ["11_18", "19_29", "30_39", "40_49", "50_59", "60_69", "70_79"],
        help = "Desired age category of all speakers in queried conversations."
    )
    parser.add_argument(
        '--no_speakers', type = int, default = 2,
        help = "Number of speakers in conversations"
    )
    parser.add_argument(
        '--no_files', type = int, default = 500,
        help = "Maximum number of files to load. 0: all"
    )
    parser.add_argument(
        '--data_path', type = str, default = '../../data/bnc2014spoken-xml/spoken/'
    )

    args = parser.parse_args()

    main(args)
