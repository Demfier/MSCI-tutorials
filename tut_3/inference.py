import json
import keras
import numpy as np
from config import config
from main import load_data, tokenize

import keras
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
from keras.layers.experimental.preprocessing import TextVectorization


def main():
    with open('data/processed/word2token.json') as f:
        word2token = json.load(f)
    with open('data/processed/token2word.json') as f:
        token2word = json.load(f)
    # Define sampling models
    # Restore the model and construct the encoder and decoder.
    model = keras.models.load_model("data/processed/s2s.model")
    print(model.summary())
    decoder_inputs = model.input[1]
    embedding_layer = model.layers[2]
    dec_embedding = embedding_layer(decoder_inputs)
    dec_state_input_h = Input((config['embedding_dim'],))
    dec_state_input_c = Input((config['embedding_dim'],))
    dec_states_inputs = [dec_state_input_h, dec_state_input_c]
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
    print('Inference encoder:')
    encoder_inputs = model.input[0]
    enc_embedding = embedding_layer(encoder_inputs)
    enc_lstm = model.layers[3]
    _, state_h, state_c = enc_lstm(enc_embedding)
    encoder_states = [state_h, state_c]
    enc_model = Model(inputs=encoder_inputs, outputs=encoder_states)
    enc_model.summary()

    _, x_test = load_data('data/processed')
    x_test = x_test[:100]

    encoder_input_data = np.ones(
        (len(x_test), config['max_seq_len']),
        dtype='float32')
    for i, input_text in enumerate(x_test):
        tokenized_text = tokenize(input_text, word2token)
        for j in range(len(tokenized_text)):
            encoder_input_data[i, j] = tokenized_text[j]

    for seq_index in range(20):
        states_values = enc_model.predict(encoder_input_data[seq_index: seq_index+1])
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

        print(decoded_translation)


if __name__ == '__main__':
    main()
