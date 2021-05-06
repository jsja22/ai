# LSTM

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

#x = x.reshape(13,3,1) # 3차원 
x = x.reshape(x.shape[0],x.shape[1],1)

# 2.모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

model = Sequential()

model.add(LSTM(10, activation='relu', input_shape=(3,1),return_sequences=True))  #(None, 3, 10)  여기서 input 10이 가장차원의끝으로 감. 즉 나가는 노드가 가장차원의 끝으로감.  
model.add(LSTM(10, activation='relu'))  #4x(input+bias+output)xoutput  4x(10+1+10)x10
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

model.summary()
#LSTM 2개이상을 엮는건 좋지않다.

# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=100, batch_size=1)


# 4. 평가, 예측

loss = model.evaluate(x,y)
print(loss)

x_pred = x_pred.reshape(1,3,1)
# lstm에 쓸 수 있는 데이터 구조로 변경 (reshape)

result = model.predict(x_pred)
print(result)

# keras23_LSTM3_scale 결과
# 0.020502837374806404
# [[80.2449]]

#LSTM 2개 엮었을떄 성능 나빠짐.
#0.8742650151252747
#[[[75.4295  ]
#  [76.353676]
#  [83.43847 ]]]
