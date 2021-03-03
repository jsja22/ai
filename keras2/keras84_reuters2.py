from tensorflow.keras.datasets import reuters
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

(x_train, y_train), (x_test,y_test) = reuters.load_data(num_words=1000,test_split=0.2)

x_train = pad_sequences(x_train,padding='pre', maxlen=100)
x_test = pad_sequences(x_test,padding='pre', maxlen=100)

y_train = to_categorical(y_train) 
y_test = to_categorical(y_test) 
print(x_train.shape)
# model

model = Sequential()
model.add(Embedding(1000,120))
model.add(LSTM(120,activation='tanh'))
model.add(Dense(46, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode ='min', verbose=1, patience=4)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics = ['acc'] )

hist = model.fit(x_train,y_train, validation_data=(x_test,y_test), batch_size=128, epochs = 30, callbacks=[es])

acc = model.evaluate(x_test,y_test)[1]
print(acc) #0.7150489687919617