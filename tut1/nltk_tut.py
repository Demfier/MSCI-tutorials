import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


def main():
    """
    tokenizes a sample dataset and removes stopwords
    """
    data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
    stop_words = stopwords.words('english')
    words = word_tokenize(data)
    data_set_stopwords = []
    filtered_words = []
    for w in words:
        if w.lower() in stop_words:
            data_set_stopwords.append(w)
            continue
        filtered_words.append(w)
    print(f'Tokens in the dataset: {words}')
    print(f'Stopwords in the dataset: {data_set_stopwords}')
    print(f'Non-stopword tokens in the dataset: {filtered_words}')


if __name__ == '__main__':
    main()
