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

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True  ,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.2,shuffle=True,random_state=66)

print(x_train.shape)    
print(y_train.shape)    

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)
print(np.max(x),np.min(x))
print(np.max(x[0]))
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
model.fit(x_train,y_train,epochs=100, batch_size=8,validation_data=(x_val,y_val))

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

#프로세싱 사용해서 다 전처리 흠냐... 값이 또 떨어짐
#loss: 3491.162841796875
#mae: 48.59267044067383
#RMSE :  59.08606226676835
#R2 : 0.46207380238143303

#validation 추가한거 나는 값이 더 떨어지냐 왜 계속 ...
#loss: 3529.97900390625
#mae: 47.64751434326172
#RMSE :  59.41362407946267
#R2 : 0.45609295023302077