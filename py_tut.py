import os
import sys
import json
import random
from tqdm import tqdm
from pprint import pprint


def main(data_path):
    """
    reads the raw dataset from data_path, creates a vocab dictionary,
    "tokenizes" the sentences in the dataset
    """

    # os.path can be used for seamless path construction across different
    # operating systems.
    with open(os.path.join(data_path, 'pos.txt')) as f:
        pos_lines = f.readlines()
    with open(os.path.join(data_path, 'neg.txt')) as f:
        neg_lines = f.readlines()

    all_lines = pos_lines + neg_lines
    total_lines = len(all_lines)

    vocab = {}
    csv_data = ''
    train_data = ''
    val_data = ''
    test_data = ''

    train_size = int(0.8*total_lines)
    val_size = int(0.1*total_lines)

    random.shuffle(all_lines)

    for idx, line in tqdm(enumerate(all_lines)):
        line = line.strip().split()
        csv_line = '{}\n'.format(','.join(line))
        csv_data += csv_line

        if idx < train_size:
            train_data += csv_line
        elif idx >= train_size and idx < train_size + val_size:
            val_data += csv_line
        else:
            test_data += csv_line

        for word in line:
            word = word.lower()
            if word not in vocab:
                vocab[word] = 0
            vocab[word] += 1

    with open('data/processed/tokenized.csv', 'w') as f:
        f.write(csv_data)
    with open('data/processed/train.csv', 'w') as f:
        f.write(train_data)
    with open('data/processed/val.csv', 'w') as f:
        f.write(val_data)
    with open('data/processed/test.csv', 'w') as f:
        f.write(test_data)
    with open('data/processed/vocab.json', 'w') as f:
        json.dump(vocab, f)


if __name__ == '__main__':
    main(sys.argv[1])
