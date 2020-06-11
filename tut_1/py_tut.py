"""
This module is a template containing basic utilities to read raw dataset
and generate appropriately processed files.
"""


import os
import sys
import json
import random
from tqdm import tqdm
from pprint import pprint


def read_dataset(data_path):
    """
    reads the raw dataset and returns all the lines as a list of string
    """
    # os.path can be used for seamless path construction across different
    # operating systems.
    with open(os.path.join(data_path, 'pos.txt')) as f:
        pos_lines = f.readlines()
    with open(os.path.join(data_path, 'neg.txt')) as f:
        neg_lines = f.readlines()
    all_lines = pos_lines + neg_lines
    return list(zip(all_lines, [1]*len(pos_lines) + [0]*len(neg_lines)))


def main(data_path):
    """
    reads the raw dataset from data_path, creates a vocab dictionary,
    "tokenizes" the sentences in the dataset.
    NOTE: In our case tokenization simply refers to splitting a sentence at ' '
    """

    # Read raw data
    all_lines = read_dataset(data_path)
    total_lines = len(all_lines)

    vocab = {}
    csv_data = ''
    train_data = ''
    val_data = ''
    test_data = ''
    labels_data = ''

    train_size = int(0.8*total_lines)
    val_size = int(0.1*total_lines)

    random.shuffle(all_lines)

    for idx, line in tqdm(enumerate(all_lines)):
        sentence = line[0].strip().split()
        label = line[1]
        # construct the entry to be added to the csv files
        csv_line = '{}\n'.format(','.join(sentence))
        csv_data += csv_line
        labels_data += '{}\n'.format(label)

        # decide whether to add the entry to train/val/test set based on idx
        if idx < train_size:
            train_data += csv_line
        elif idx >= train_size and idx < train_size + val_size:
            val_data += csv_line
        else:
            test_data += csv_line

        # constuct vocab dictionary
        for word in sentence:
            word = word.lower()
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

    # save processed files
    with open('data/processed/tokenized.csv', 'w') as f:
        f.write(csv_data)
    with open('data/processed/train.csv', 'w') as f:
        f.write(train_data)
    with open('data/processed/val.csv', 'w') as f:
        f.write(val_data)
    with open('data/processed/test.csv', 'w') as f:
        f.write(test_data)
    with open('data/processed/labels.csv', 'w') as f:
        f.write(labels_data)
    with open('data/processed/vocab.json', 'w') as f:
        json.dump(vocab, f)


if __name__ == '__main__':
    # use a command-line argument to input raw data path
    main(sys.argv[1])
