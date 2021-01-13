#실습 : 19_1,2,3,4,5, Earlystopping까지 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Dropout,LSTM,Conv1D
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 

dataset = load_diabetes()
x= dataset.data
y= dataset.target

print(x[:5]) 
print(y[:10])
print(x.shape,y.shape) #(442,10) (442,)

print("np.max(x),np.min(x) :",np.max(x),np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True  ,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.2,shuffle=True,random_state=66)

print(x_train.shape)   #(70,10)
print(y_train.shape)    #(70,)


scalar = MinMaxScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

model = Sequential()
model.add(LSTM(128,input_shape = (10,1),activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])


  
early_stopping = EarlyStopping(monitor='loss',patience=30,mode='auto')
model.fit(x_train,y_train,epochs=500, batch_size=4,validation_data=(x_val,y_val),callbacks=[early_stopping]) #batch_size 8 

loss,mae = model.evaluate(x_test,y_test,batch_size=4)
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

#modelcheckpoint
#loss: 3455.11865234375
#mae: 48.15746307373047
#RMSE :  58.780257135263895
#R2 : 0.4676275621562681

#lstm
#loss: 5915.591796875
#mae: 66.69463348388672
#RMSE :  76.91289320074328
#R2 : 0.0885121236725196