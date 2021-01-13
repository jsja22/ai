#인공지능계의 hello world라 불리는 mnist!!
#to_categorical
#전처리
#실습
#지표는 acc 0.985이상
#응용 -> y_test 10개와 y_pred를 출력하시오. 
# y_test[:10]= (?,?,?,?,?,?,?,?,?,?)
# y_pred[:10]= (?,?,?,?,?,?,?,?,?,?)


import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_train.shape)    #(60000, 28, 28)  ->흑백
print(y_train.shape)    #(60000,)
print(x_test.shape)     #(10000, 28, 28)
print(y_test.shape)     #(10000,)

print(x_train[0])
print(y_train[0])  #5

print(x_train[0].shape)   #(28,28)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.  #(0~1사이로 수렴) =>전처리
x_test = x_test.reshape(10000,28,28,1)/255.  #이것도 실수로 인식되어서 가능 이렇게 쓰자!

print(x_train.shape)
print(x_test.shape)

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

print(y_train.shape)
print(y_test.shape)

from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters=100, kernel_size=(2,2), padding='same',input_shape=(28,28,1)))
model.add(MaxPooling1D(pool_size=2))
#model.add(Dropout(0.2))
model.add(Conv1D(30,2))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10, mode='auto')
model.fit(x_train, y_train, epochs=100, batch_size=128,validation_split=0.2, callbacks=[early_stopping])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=128)
print(loss, acc)

y_pred = model.predict(x_test[:-10])
y_pred1 = np.argmax(y_pred, axis=1).reshape(-1,1)
print("y_pred1",y_pred1)
print(y_test[-10:-1])

#loss,acc
#0.12682780623435974 0.982699990272522

#lstm으로 
