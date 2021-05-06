#6개의 파일을 만드시오.
#1. earlystopping 을 적용하지 않은 최고의 모델
#2. earlystopping 을 적용한 최고의 모델

import numpy as np
from tensorflow.keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape)
print(x_test.shape)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

'''
print("x_train :",x_train)
print("y_train :",y_train)
print("x_test :",x_test)
print("y_test :",y_test)
'''

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)


#2. 모델구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
inputs = Input(shape=(13,))
dense1 = Dense(104, activation='relu')(inputs)
dense1 = Dense(64)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dense(16)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, batch_size=12, epochs=200, validation_data=(x_val, y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=12)
print("loss :", loss)
print("mae : ", mae)
y_predict = model.predict(x_test)
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

#batch_size=12,  epochs=200
#loss : 10.99948501586914
#mae :  2.1619343757629395
#RMSE :  3.316547244933523
#R2 :  0.8678642827852938
