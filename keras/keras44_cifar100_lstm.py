
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test,y_test)= cifar100.load_data()

print(x_train.shape, y_train.shape)

print(x_train[0])
print(y_train[0])


print(x_train[0].shape) # (28,28)

x_train = x_train.reshape(-1,32,32,3).astype('float32')/255
x_test= x_test.reshape(-1,32,32,3).astype('float32')/255

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(LSTM(512, activation='relu', input_shape=(32*32,3)))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=16,verbose=1,validation_split=0.2, callbacks=[early_stopping])

loss = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test,axis=-1)
y_predict = np.argmax(y_predict,axis=-1)
