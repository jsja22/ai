#실습 : 19_1,2,3,4,5, Earlystopping까지 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target

print(x[:5])
print(y[:10])
print(x.shape,y.shape) #(442,10) (442,)

print("np.max(x),np.min(x) :",np.max(x),np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)



#데이터 전처리(MinMax)
x = (x+0.137767225690012)/(0.198787989657293+0.137767225690012)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=66)

print(x_train.shape)    
print(y_train.shape)    

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(128,input_dim = 10,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=8, validation_split=0.2)

loss,mae = model.evaluate(x_test,y_test,batch_size=8)
print('loss:',loss)
print('mae:',mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)

#전처리 후
#loss: 3343.386962890625
#mae: 47.89411926269531
#RMSE :  57.822028575912405
#R2 : 0.48484342421351634