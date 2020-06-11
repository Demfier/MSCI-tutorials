import os
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support


def read_csv(csv_path):
    with open(csv_path) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]


def load_data(data_dir):
    x_train = read_csv(os.path.join(data_dir, 'train.csv'))
    x_val = read_csv(os.path.join(data_dir, 'val.csv'))
    x_test = read_csv(os.path.join(data_dir, 'test.csv'))
    labels = read_csv(os.path.join(data_dir, 'labels.csv'))
    labels = [int(label) for label in labels]
    y_train = labels[:len(x_train)]
    y_val = labels[len(x_train): len(x_train)+len(x_val)]
    y_test = labels[-len(x_test):]
    return x_train, x_val, x_test, y_train, y_val, y_test


def train(x_train, y_train):
    print('CountVectorizer')
    count_vect = CountVectorizer()
    x_train_counts = count_vect.fit_transform(x_train)
    print('TfIdf')
    tfidf_transformer = TfidfTransformer()
    x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
    print('MNB')
    clf = MultinomialNB().fit(x_train_tfidf, y_train)
    return clf, count_vect, tfidf_transformer


def main(data_dir):
    x_train, x_val, x_test, y_train, y_val, y_test = load_data(data_dir)
    # train
    clf, count_vect, tfidf_transformer = train(x_train, y_train)
    # validate
    x_val_counts = count_vect.transform(x_val)
    x_val_tfidf = tfidf_transformer.transform(x_val_counts)
    predicted = clf.predict(x_val_tfidf)

    prfa = precision_recall_fscore_support(y_val, predicted, average='macro')
    print(prfa)


if __name__ == '__main__':
    main(sys.argv[1])
