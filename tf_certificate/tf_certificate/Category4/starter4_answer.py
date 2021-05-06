# ======================================================================
# There are 5 questions in this exam with increasing difficulty from 1-5.
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score significantly
# less than your Category 5 question.
#
# Don't use lambda layers in your model.
# You do not need them to solve the question.
# Lambda layers are not supported by the grading infrastructure.
#
# You must use the Submit and Test button to submit your model
# at least once in this category before you finally submit your exam,
# otherwise you will score zero for this category.
# ======================================================================
#
# NLP QUESTION
#
# Build and train a classifier for the sarcasm dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown.
# It will be tested against a number of sentences that the network hasn't previously seen
# and you will be scored on whether sarcasm was correctly detected in those sentences.

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    with open('sarcasm.json', 'r') as d:
        datas = json.load(d)

    for data in datas:
        sentences.append(data['headline'])
        labels.append(data['is_sarcastic'])

    print(sentences[:5])
    print(labels[:5])

    training_size = 20000

    train_sentences = sentences[0:training_size]
    train_labels = labels[0:training_size]

    valid_sentences = sentences[training_size:]
    valid_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    # tokenizer.fit_on_texts(valid_sentences)

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    valid_sequences = tokenizer.texts_to_sequences(valid_sentences)

    print(train_sequences[:5])
    print(valid_sequences[:5])

    train_padded = pad_sequences(train_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)
    valid_padded = pad_sequences(valid_sequences, truncating=trunc_type, padding=padding_type, maxlen=max_length)

    train_labels = np.asarray(train_labels)
    valid_labels = np.asarray(valid_labels)

    print(train_padded.shape)  # (20000, 120)

    # 변환전
    sample = np.array(train_padded[0])
    print(sample)

    # 변환 후
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)
    print(x(sample)[0])

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint_path = 'tmp_checkpoint.ckpt'
    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)
    model.fit(train_padded, train_labels, validation_data=(valid_padded, valid_labels), epochs=100,
              callbacks=[checkpoint], )
    model.load_weights(checkpoint_path)
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/Users/bitcamp/study/tf_certificate/Category4/mymodel.h5")

#loss: 0.0293 - accuracy: 0.9874 - val_loss: 2.7623 - val_accuracy: 0.7696

