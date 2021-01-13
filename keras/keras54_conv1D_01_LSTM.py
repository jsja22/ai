# LSTM
#conv1D로 코딩하시오


import numpy as np

# 1. 데이터
x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
            [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
            [20,30,40], [30,40,50], [40,50,60]])

y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x_pred = np.array([50,60,70])

# 코딩하시오!!! LSTM
# 나는 80을 원하고 있다

print(x.shape) # (13,3)
print(y.shape) # (13,)

x = x.reshape(13,3,1) # 3차원 


# 2.모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPooling1D,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential()

model.add(Conv1D(filters=64, kernel_size =2 , activation='relu',padding='same', input_shape=(3,1)))
model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten())    #flatten 을 하면 2차원으로 변경되어서 다음에 conv1D를 또 하려고하면 오류가 뜬다.
model.add(Dense(50, activation ='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(30, kernel_size =2 , strides =1, padding='same'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.2))
#model.add(Conv1D(30, kernel_size =2 , strides =1, padding='same'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)


# 4. 평가, 예측

loss = model.evaluate(x,y)
print(loss)

x_pred = x_pred.reshape(1,3,1)
# LSTM에 쓸 수 있는 데이터 구조로 변경

result = model.predict(x_pred)
print(result)

#LSTM
#LOSS: 0.732516348361969
#[[78.32462]]

#CONV1D 
#2.160428762435913
#[[[83.5801]]]