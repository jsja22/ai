#embedding model


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train, maxlen=240)
x_test = pad_sequences(x_test, maxlen=240)

# y_train = to_categorical(y_train) 
# y_test = to_categorical(y_test) 
print(x_train.shape)
# model

model = Sequential()
model.add(Embedding(10000, 100, input_length = 240))
model.add(LSTM(128))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

es = EarlyStopping(patience = 10)
lr = ReduceLROnPlateau(factor = 0.25, patience = 5, verbose = 1)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_train, y_train, validation_split = 0.2, epochs = 1000, callbacks = [es, lr])

#4. 평가
loss, acc = model.evaluate(x_test, y_test, batch_size = 32)
print('acc : ', acc)

#acc :  0.8492799997329712