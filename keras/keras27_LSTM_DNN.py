# kears23_LSTM3_scale 를  DNN 으로 코딩 
# 결과치 비교
#dnn으로 23번 파일보다 loss 를 좋게 만들것
import numpy as np
import tensorflow as tf

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])

print(x.shape) #(13,3)
print(y.shape) #(13,)
print(x_pred.shape) #(3,)

x_pred = x_pred.reshape(1,3)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=3, activation='relu'))  
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=24, mode='auto')
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[early_stopping])

loss = model.evaluate(x_test, y_test)
y_pred = model.predict(x_pred)
print('loss : ', loss)
print('y_pred : ', y_pred)

#loss :  0.018260391429066658
#y_pred :  [[81.62889]]
