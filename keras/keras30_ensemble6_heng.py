##행이 다른 앙상블 모델에 대해 공부!!!!
#
import numpy as np
from numpy import array
#1. 데이터
x1 = array([[1,2,3,],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12]])
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])
y1 = array([4,5,6,7,8,9,10,11,12,13])
y2 = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])

#LSTM이 3차원이기에 reshape로 3차원을 만들어줘야함
x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)
print(x1.shape) #(10,3,1)
print(x2.shape) #(13,3,1)
#print(y.shape)

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

input1 =Input(shape=(3,1))
LSTM1 = LSTM(10, activation='relu')(input1)
LSTM1 = Dense(20, activation='relu')(LSTM1)
LSTM1 = Dense(40, activation='relu')(LSTM1)
LSTM1 = Dense(20, activation='relu')(LSTM1)
LSTM1 = Dense(10, activation='relu')(LSTM1)


input2 = Input(shape=(3,1))
LSTM2 = LSTM(10,activation='relu')(input2)
LSTM2 = Dense(20, activation='relu')(LSTM2)
LSTM2 = Dense(40, activation='relu')(LSTM2)
LSTM2 = Dense(20, activation='relu')(LSTM2)
LSTM2 = Dense(10, activation='relu')(LSTM2)

######concatenate
from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([LSTM1,LSTM2])

output1 = Dense(10)(merge1)
output1 = Dense(5)(output1)
output1 = Dense(1)(output1) 

output2 = Dense(10)(merge1)
output2 = Dense(5)(output2)
output2 = Dense(1)(output2) 
#모델 선언
model = Model(inputs = [input1,input2], outputs = [output1,output2]) #2개 이상 들어갈 때는 []리스트로 묶어서 넣어줘라 이건 컴파일에도 마찬가지
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=24, mode='auto')
model.fit([x1,x2], [y1,y2], epochs=1000,batch_size=16,callbacks=[early_stopping])

loss = model.evaluate([x1,x2],[y1,y2],batch_size=16)
print(loss)

y1_predict ,y2_predict= model.predict([x1_pred, x2_pred])

print("y1_predict,y2_predict: ", y1_predict,y2_predict)
'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)
'''

#predict는 85의 근사치. 

#[815.7168579101562, 815.7168579101562]  85가 들어가있음

#[0.0008926899754442275, 0.0008926899754442275]
#y_predict:  [[88.51729]]  

#ValueError: Data cardinality is ambiguous:
#  x sizes: 10, 13
# y sizes: 10, 13
#Please provide data which shares the same first dimension.
#=>앙상블할때는 행을 반드시 맞춰줘야한다.

