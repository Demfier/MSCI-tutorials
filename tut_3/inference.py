"""
Inference script to test reconstruction quality of the trained autoencoder
"""
import os
import sys
import json
import numpy as np
from config import config
from main import load_data, tokenize

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers.experimental.preprocessing import TextVectorization


def main(data_dir, model_code):
    # load the two crucial dictionaries
    with open(os.path.join(data_dir, 'word2token.json')) as f:
        word2token = json.load(f)
    with open(os.path.join(data_dir, 'token2word.json')) as f:
        token2word = json.load(f)
    # Restore the model and construct the encoder and decoder.
    model = keras.models.load_model(
        os.path.join(data_dir, '{}.model'.format(model_code)))
    print(model.summary())

    # reconstruct the decoding layer using the loaded model
    decoder_inputs = model.input[1]
    embedding_layer = model.layers[2]
    dec_embedding = embedding_layer(decoder_inputs)
    dec_state_input_h = Input((config['embedding_dim'],))
    dec_state_input_c = Input((config['embedding_dim'],))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
    # you can load layers by their names as well!
    dec_lstm = model.get_layer('dec_lstm')
    dec_outputs, state_h, state_c = dec_lstm(dec_embedding,
                                             initial_state=dec_states_inputs)
    dec_dense = model.layers[5]
    dec_states = [state_h, state_c]
    dec_outputs = dec_dense(dec_outputs)
    dec_model = Model(
        inputs=[decoder_inputs] + dec_states_inputs,
        outputs=[dec_outputs] + dec_states)
    print('Inference decoder:')
    dec_model.summary()

    # reconstruct the encoding layer
    encoder_inputs = model.input[0]
    enc_embedding = embedding_layer(encoder_inputs)
    enc_lstm = model.layers[3]
    _, state_h, state_c = enc_lstm(enc_embedding)
    encoder_states = [state_h, state_c]
    enc_model = Model(inputs=encoder_inputs, outputs=encoder_states)
    print('Inference encoder:')
    enc_model.summary()

    _, x_test = load_data(data_dir)
    # test the entire pipeline on 100 test sentences
    x_test = x_test[:100]

    encoder_input_data = np.ones(
        (len(x_test), config['max_seq_len']),
        dtype='float32')

    # tokenize test sentences
    for i, input_text in enumerate(x_test):
        tokenized_text = tokenize(input_text, word2token)
        for j in range(len(tokenized_text)):
            encoder_input_data[i, j] = tokenized_text[j]

    for seq_index in range(len(x_test)):
        states_values = enc_model.predict(
            encoder_input_data[seq_index: seq_index+1])
        empty_target_seq = np.zeros((1, 1))
        empty_target_seq[0, 0] = word2token['<sos>']
        stop_condition = False
        decoded_translation = ''
        while not stop_condition:
            dec_outputs, h, c = dec_model.predict([empty_target_seq]
                                                  + states_values)
            sampled_word_index = np.argmax(dec_outputs[0, -1, :])
            sampled_word = None
            for word, index in word2token.items():
                if sampled_word_index == index:
                    if word != '<eos>':
                        decoded_translation += ' {}'.format(word)
                    sampled_word = word

            if sampled_word == '<eos>' \
                    or len(decoded_translation.split()) \
                    > config['max_seq_len']:
                stop_condition = True

            empty_target_seq = np.zeros((1, 1))
            empty_target_seq[0, 0] = sampled_word_index
            states_values = [h, c]

        print('Input sentence: {}\nReconstructed sentence: {}\n'.format(
            x_test[seq_index][len('<sos> '):-len(' <eos>')], decoded_translation))
        print('===============')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
