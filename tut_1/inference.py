import sys
from gensim.models import Word2Vec


def main(text_path):
    with open(text_path) as f:
        sample_text = f.readlines()
    # this sample.txt contains one word per line
    sample_text = [w.strip() for w in sample_text]
    # this path would be a3/data/w2v.model for assignments
    w2v = Word2Vec.load('data/processed/w2v.model')
    # you can choose your own method to get the most similar words
    # this is just an example.
    return ['{} => {}'.format(w, [o[0] for o in w2v.most_similar([w2v[w]], topn=21)[1:]]) for w in sample_text]


if __name__ == '__main__':
    most_similar = main(sys.argv[1])
    print('\n\n'.join(most_similar))
