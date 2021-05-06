#보스턴 모델 구성 r2 ->0.99로 올리자

import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x= dataset.data
y= dataset.target
print(x.shape)  #(506,13)
print(y.shape)  #(506, )
print("===================")
print(x[:5])
print(y[:10])

#x= x/711.
#x=np.transpose(x)
#y=np.transpose(y)
print("np.max(x),np.min(x) :",np.max(x),np.min(x))
#print(np.max(x),np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,train_size=0.8,shuffle=True,random_state=66)

from sklearn.preprocessing import MinMaxScaler   #칼람별로 전처리를 하게된다.
scalar = MinMaxScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train) #x_test,x_pred,x_val 모두 traSnsform만 해주면된다. => x_train이 기준으로 잡혔기 때문에
x_test=scalar.transform(x_test) #x_test,x_pred,x_val 모두 traSnsform만 해주면된다. => x_train이 기준으로 잡혔기 때문에
x_val=scalar.transform(x_val)

print(np.max(x),np.min(x)) 
print(np.max(x[0]))

print(x_train.shape)    #(80,3)
print(y_train.shape)    #(80,3)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(128,input_dim = 13,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=10,mode='auto')

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=2000, batch_size=8, validation_data=(x_val,y_val),callbacks=[early_stopping])

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

#전처리전
#loss: 22.612598419189453
#mae: 3.674478530883789
#RMSE :  4.75527008142155
#R2 : 0.7294592141755114

#전처리 후
#loss: 11.208980560302734
#mae: 2.456184148788452
#RMSE :  3.3479816744587594
#R2 : 0.8658939054987277

#전처리 MInmax scalar로 한 후
#loss: 10.43908977508545
#mae: 2.12522029876709
#RMSE :  3.2309583171052823
#R2 : 0.8751049917588777

#전처리 x_test도 스칼라 추가
#loss: 9.8983736038208
#mae: 2.169928550720215
#RMSE :  3.1461678894503797
#R2 : 0.8815742458378134

#x_val까지 추가한거
#loss: 7.153820991516113
#mae: 1.9578073024749756
#RMSE :  2.6746628852032566
#R2 : 0.9144105031738045
#=>최종 다해서 0.91까지 올림 

#earlystopping 돌린거
#loss: 6.76370906829834
#mae: 2.07267427444458
#RMSE :  2.600713394713574
#R2 : 0.9190778599393677
#=> 가장좋음

