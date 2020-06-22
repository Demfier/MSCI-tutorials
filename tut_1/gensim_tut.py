"""
This module includes basic utilities to train a word2vec model using gensim
"""
import os
import sys
from py_tut import read_dataset
from gensim.models import Word2Vec


def main(data_path):
    """
    Train a word2vec model on the given dataset
    """
    all_lines = read_dataset(data_path)
    print('Splitting lines in the dataset')
    all_lines = [line[0].strip().split() for line in all_lines]
    print('Training word2vec model')
    # This will take some to finish
    w2v = Word2Vec(all_lines, size=100, window=5, min_count=1, workers=4)
    w2v.save('data/processed/w2v.model')


if __name__ == '__main__':
    main(sys.argv[1])
