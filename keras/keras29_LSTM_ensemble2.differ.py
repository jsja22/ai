#2개의 모델을 하나는 LSTM, 하나는 Dense로
#앙상블 구현 !!
#29_1과 성능비교

import numpy as np
from numpy import array
#1. 데이터
x1 = array([[1,2,3,],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predict = array([55,65,75])
x2_predict = array([65,75,85])

#LSTM이 3차원이기에 reshape로 3차원을 만들어줘야함
#x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
print(x1.shape) #(13,3,1)
print(x2.shape) #(13,3,1)
print(y.shape)

x1_pred=x1_predict.reshape(1, 3, 1)
x2_pred=x2_predict.reshape(1, 3, 1)
'''
from sklearn.model_selection import train_test_split
x1_train,x1_test, y_train,y_test = train_test_split(x1,y, train_size=0.8, shuffle=True)
x2_train,x2_test, y_train,y_test = train_test_split(x2,y, train_size=0.8, shuffle=True)
'''
#2. 모델 구성

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

input1 =Input(shape=(3,))
Dense1 = Dense(10, activation='relu')(input1)
Dense1 = Dense(20, activation='relu')(Dense1)
Dense1 = Dense(40, activation='relu')(Dense1)
Dense1 = Dense(20, activation='relu')(Dense1)
Dense1 = Dense(20, activation='relu')(Dense1)

input2 = Input(shape=(3,1))
LSTM1 = LSTM(10,activation='relu')(input2)
LSTM1 = Dense(20, activation='relu')(LSTM1)


######concatenate
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([LSTM1,Dense1])

output = Dense(10)(merge1)
output = Dense(5)(output)
output = Dense(1)(output) 

#모델 선언
model = Model(inputs = [input1,input2], outputs = [output]) #2개 이상 들어갈 때는 []리스트로 묶어서 넣어줘라 이건 컴파일에도 마찬가지
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=25, mode='auto')
model.fit([x1,x2], y, epochs=1000, batch_size=16, callbacks=[early_stopping])

loss = model.evaluate([x1,x2],y,batch_size=16 )
print(loss)

y_predict = model.predict([x1_pred, x2_pred])
print("y_predict: ", y_predict)


#predict는 85의 근사치. 

#[815.7168579101562, 815.7168579101562]  85가 들어가있음
#print(y_test)
#y_predict= y_predict.reshape(3,1)
#y_test = y_test.reshape(3,1)
'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)
'''

#[0.0004981888341717422, 0.0004981888341717422]
#y_predict:  [[84.96388]]
#lstm이 낫다, dense가 낫다 그건 경험을 통해 알수 있다. 