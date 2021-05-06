
#1. 데이터
import numpy as np
import tensorflow as tf

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) #(4,3)      
y = np.array([4,5,6,7])                            #(4,)          
x = x.reshape(4, 3, 1)

#2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM             #LSTM은 3차원을 받아들인다. , Dense 는 2차원 RNN 4차원

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(3,1))) # input_shape는 한차원씩 줄어든다. 1개의 timestep이 3개있음 
model.add(Dense(20))                                      # 3 x ( 1 + 1 + 10) x 10 = 480
model.add(Dense(10))                                      # 4개의 게이트(sigmoid 3개, tanh 1개), input, bias, output, output
model.add(Dense(1))                                     

#컴파일, 훈련

model.compile(loss = 'mse', optimizer='adam')
model.fit(x, y, epochs=100, batch_size=1)


#4. 평가 예측
loss = model.evaluate(x)
print('loss : ', loss)

x_pred = np.array([5,6,7]) 
x_pred = x_pred.reshape(1,3,1)  

result = model.predict(x_pred)
print(result) #[[8.126679]]
# LSTM > GRU > RNN  LSTM이 가장 복잡하고 성능이 좋음
