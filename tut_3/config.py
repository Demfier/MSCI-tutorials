# contains hyperparameters for our seq2seq model. you can add some more
config = {
    'batch_size': 200,
    'max_vocab_size': 20000,
    'max_seq_len': 26,  # decided based on the sentence length distribution of amazon corpus
    'embedding_dim': 100,  # we had trained a 100-dim w2v vecs in tut 1
}
