#dense 모델을 구성하시오

import numpy as np

a= np.array(range(1,11))
size = 5

def split_x(seq,size):
    aaa=[]
    for i in range(len(seq)-size +1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size)
print("================")
print(dataset)

x = dataset[:,0:4]
y = dataset[:,-1]

print(x.shape)
print(y.shape)

#x= x.reshape(x.shape[0],x.shape[1],1)
print(x.shape)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense ,LSTM
model= Sequential()
model.add(Dense(10, activation='relu', input_shape=(4,))) 
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(80, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam',metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
model.fit(x,y,epochs=100, batch_size=8,validation_split=0.2,callbacks=[early_stopping])

loss,mae = model.evaluate(x,y,batch_size=8)
print('loss:',loss)
print('mae:',mae)

x_pred = dataset[-1,1:]
print(x_pred)
x_pred = x_pred.reshape(1,4)
print(x_pred)
y_predict = model.predict(x_pred)
print(y_predict)
'''
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)
'''
#LSTM일때
#loss: 0.2218434363603592
#mae: 0.4229716360569
#y_predict= [[10.20452]]

#Dense일때
#loss: 0.2148687094449997
#mae: 0.39338937401771545
#[[12.046032]]