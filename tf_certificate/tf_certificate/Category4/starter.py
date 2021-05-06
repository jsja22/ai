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


def solution_model():

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url,'sarcasm.json')
    
    with open('sarcasm.json') as file:
        data = json.load(file)

    dataset =[]
    for elem in data:
        sentences.append(elem['headline'])
        labels.append(elem['is_sarcastic'])

    training_size = int(len(data)*0.2)
    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]
    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    #토큰화, 단어집합 생성, 정수인코딩, 패딩
    #1.토크나이즈 정의
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<oov>')
    #oov_token: 단어 집합에 없는 단어를 어떻게 표기할 것인지 지정
    #2. 학습시킬 문장을 토큰화 하고 단어집합을 만듬
    tokenizer.fit_on_texts(sentences)
    word_to_idx = tokenizer.word_index

    #3. 문장을 각각 고유한 정수로 매핑
    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

    #padding

    train_padded = pad_sequences(train_sequences, truncating=trunc_type,padding=padding_type , maxlen=max_length)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type , maxlen=max_length)


    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(23,activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    es = EarlyStopping(monitor='val_acc', patience=8, mode='auto')
    lr = ReduceLROnPlateau(factor=0.3, patience=5, verbose=1, mode='auto')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, validation_split=0.2,epochs=200, callbacks=[es,lr] )
    return model


# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.
if __name__ == '__main__':
    model = solution_model()
    model.save("C:/data/tf_certificate/category4/mymodel4.h5")

