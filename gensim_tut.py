import gensim
from gensim.models import Word2Vec


def main():
    """
    Train a word2vec model on the given dataset
    """
    with open('data/raw/pos.txt') as f:
        pos_lines = f.readlines()
    with open('data/raw/neg.txt') as f:
        neg_lines = f.readlines()

    all_lines = pos_lines + neg_lines
    print('Splitting lines in the dataset')
    all_lines = [line.strip().split() for line in all_lines]
    print('Training word2vec model')
    w2v = Word2Vec(all_lines, size=100, window=5, min_count=1, workers=4)
    w2v.save('data/processed/w2v.model')


if __name__ == '__main__':
    main()
