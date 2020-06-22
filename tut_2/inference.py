import sys
import pickle
from pprint import pprint
from main import load_data


def predict(line, clf, count_vect, tfidf_transformer):
    """slightly different from the evalaute function in main.py"""
    # this function could've been called on all the lines at once but doing one
    # line at a time to avoid confusion
    line2feature = tfidf_transformer.transform(count_vect.transform([line]))
    return 'Postive' if clf.predict(line2feature) else 'Negative'


def main(text_path, model_code):
    # load data
    with open(text_path) as f:
        sample_text = f.readlines()
    sample_text = [line.strip() for line in sample_text]

    with open('tut_2/data/{}.pkl'.format(model_code), 'rb') as f:
        clf = pickle.load(f)

    with open('tut_2/data/count_vect.pkl', 'rb') as f:
        count_vect = pickle.load(f)

    with open('tut_2/data/tfidf_transformer.pkl', 'rb') as f:
        tfidf_transformer = pickle.load(f)

    return ['{} => {}'.format(line, predict(line, clf, count_vect, tfidf_transformer))
            for line in sample_text]


if __name__ == '__main__':
    predictions = main(sys.argv[1], sys.argv[2])
    print('\n'.join(predictions))
