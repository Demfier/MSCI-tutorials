"""
Trains a simple text-autoencoder on the amazon corpus
"""
import os
import sys
import json
import numpy as np
import tensorflow as tf
from config import config
from gensim.models import Word2Vec

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers.experimental.preprocessing import TextVectorization


def load_data(data_dir):
    """
    In an autoencoder, we don't have labels as input = output.
    For simplicity purposes, I've combined the train and val set into a single
    """
    with open(os.path.join(data_dir, 'train_val.txt')) as f:
        x = f.readlines()
    with open(os.path.join(data_dir, 'test.txt')) as f:
        x_test = f.readlines()
    return ['<sos> ' + ' '.join(line.split()[:config['max_seq_len'] - 2]) + ' <eos>'
            for line in x], ['<sos> ' + ' '.join(line.split()[:config['max_seq_len'] - 2]) + ' <eos>'
                             for line in x_test]


def build_embedding_mat(data_dir, vocab, w2v):
    """
    Build the embedding matrix which will be used to initialize weights of
    the embedding layer in our seq2seq architecture
    """
    # we have 4 special tokens in our vocab
    token2word = {0: '<sos>', 1: '<pad>', 2: '<eos>', 3: '<unk>'}
    word2token = {'<sos>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3}
    # +4 for the four vocab tokens
    vocab_size = len(vocab) + 4
    embedding_dim = config['embedding_dim']
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # randomly initizlize embeddings for the special tokens
    # you can play with different types of initializers
    embedding_matrix[0] = np.random.random((1, embedding_dim))
    embedding_matrix[1] = np.random.random((1, embedding_dim))
    embedding_matrix[2] = np.random.random((1, embedding_dim))
    embedding_matrix[3] = np.random.random((1, embedding_dim))
    for i, word in enumerate(vocab):
        # since a word in the vocab of our vectorizer is actually stored as
        # byte values, we need to decode them as strings explicitly
        word = word.decode('utf-8')
        try:
            # again, +4 for the four special tokens in our vocab
            embedding_matrix[i+4] = w2v[word]
            # build token-id -> word dict (will be used when decoding)
            token2word[i+4] = word
            # build word -> token-id dict (will be used when encoding)
            word2token[word] = i+4
        except KeyError as e:
            # skip any oov words from the perspective of our trained w2v model
            continue
    # save the two dicts
    with open(os.path.join(data_dir, 'token2word.json'), 'w') as f:
        json.dump(token2word, f)
    with open(os.path.join(data_dir, 'word2token.json'), 'w') as f:
        json.dump(word2token, f)
    return embedding_matrix, word2token


def tokenize(sentence, word2token):
    tokenized = []
    for w in sentence.lower().split():
        token_id = word2token.get(w)
        if token_id is None:
            tokenized.append(word2token['<unk>'])
        else:
            tokenized.append(token_id)
    return tokenized


def main(data_dir):
    print('Loading data')
    x_train_val, x_test = load_data(data_dir)
    # decrease dataset size for quick testing
    # x_train_val = x_train_val[:1000]
    # x_test = x_test[:100]

    # build vocab
    # NOTE: this script only considers tokens in the training set to build the
    # vocabulary object.
    vectorizer = TextVectorization(max_tokens=config['max_vocab_size'],
                                   output_sequence_length=config['max_seq_len'])
    text_data = tf.data.Dataset.from_tensor_slices(x_train_val).batch(config['batch_size'])
    print('Building vocabulary')
    vectorizer.adapt(text_data)
    # NOTE: in this vocab, index 0 is reserved for padding and 1 is reserved
    # for out of vocabulary tokens
    vocab = vectorizer.get_vocabulary()

    # load pre-trained w2v model (this model was trained in tut_1)
    w2v = Word2Vec.load(os.path.join(data_dir, 'w2v.model'))
    print('Building embedding matrix')
    # This matrix will be used to initialze weights in the embedding layer
    embedding_matrix, word2token = build_embedding_mat(data_dir, vocab, w2v)
    print('embedding_matrix.shape => {}'.format(embedding_matrix.shape))

    print('Building Seq2Seq model')

    # build the embedding layer to convert token sequences into embeddings
    # set trainable to True if you wish to further finetune the embeddings.
    # It will increase train time but may yield better results. Try it out
    # on a more complex task (like neural machine translation)!
    embedding_layer = Embedding(
        input_dim=len(vocab)+4,
        output_dim=config['embedding_dim'],
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
        )

    # build the encoding layers
    # encoder_inputs accepts padded tokenized sequences as input,
    # which would be converted to embeddings by the embedding_layer
    # finally, the embedded sequences are fed to the encoder LSTM to get
    # encodings (or vector representation) of the input sentences
    # you can add droputs the input/embedding layers and make your model robust
    encoder_inputs = Input((None,), name='enc_inp')
    enc_embedding = embedding_layer(encoder_inputs)
    # you can choose a GRU/Dense layer as well to keep things easier
    # note that we are not using the encoder_outputs for the given generative
    # task, but you'll need it for classification
    # Also, there hidden dimension is currently equal to the embedding dimension
    _, state_h, state_c = LSTM(config['embedding_dim'],  # try a different value
                               return_state=True,
                               name='enc_lstm')(enc_embedding)
    encoder_states = [state_h, state_c]

    # build the decoding layers
    # decoder_inputs and dec_embedding serve similar purposes as in the encoding
    # layers. Note that we are using the same embedding_layer to convert
    # token sequences to embeddings while encoding and decoding.
    # In this case, we initialize the decoder using `encoder_states`
    # as its initial state (i.e. vector representation learned by the encoder).
    decoder_inputs = Input((None,), name='dec_inp')
    dec_embedding = embedding_layer(decoder_inputs)
    dec_lstm = LSTM(config['embedding_dim'],
                    return_state=True,
                    return_sequences=True,
                    name='dec_lstm')
    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=encoder_states)
    # finally, we add a final fully connected layer which performs the
    # transformation of decoder outputs to logits vectors
    dec_dense = Dense(len(vocab)+4, activation='softmax', name='out')
    output = dec_dense(dec_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print(model.summary())

    # note that decoder_input_data and decoder_target_data will be same
    # as we are training a vanilla autoencoder
    # we are using np.ones as pad tokens are represented by 1 in our vocab
    # TODO: switch to a generator instead of creating such huge matrics.
    # will reduce memory consumption a lot.
    encoder_input_data = np.ones(
        (len(x_train_val), config['max_seq_len']),
        dtype='float32')
    decoder_input_data = np.ones(
        (len(x_train_val), config['max_seq_len']),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(x_train_val), config['max_seq_len'], len(vocab)+4),
        dtype='float32')

    for i, input_text in enumerate(x_train_val):
        tokenized_text = tokenize(input_text, word2token)
        for j in range(len(tokenized_text)):
            encoder_input_data[i, j] = tokenized_text[j]
            decoder_input_data[i, j] = tokenized_text[j]
            decoder_target_data[i, j, tokenized_text[j]] = 1.0
    # Run training (will take some time)
    print('Training model')
    # try different optimizers, learning rates, and analyze different metrics
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=config['batch_size'],
              epochs=10,  # try increasin #epochs
              validation_split=0.2)
    # Save model
    # this model is saved inside the tut_3/data folder just to showcase how
    # you can save your models as well inside respective assignment folders
    # and use them later
    model.save('tut_3/data/ae.model')


if __name__ == '__main__':
    main(sys.argv[1])
