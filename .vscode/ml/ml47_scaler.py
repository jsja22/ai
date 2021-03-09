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

#x=np.transpose(x)
#y=np.transpose(y)

print(np.max(x),np.min(x))  #711.0 0.0 
print(dataset.feature_names)
#print(dataset.DESCR)

#데이터 전처리(MinMax)
#x = x/711.   => 이는 잘못된 전처리 칼람별로 민맥스값이 다르기때문에
#x가 0부터 시작한다는걸 안다는 가정하에하지만 모를경우엔
#x = (x-최소)/(최대-최소)
# = (x - np.min(x))/(np.max(x)-np.min(x))
print(np.max(x[0]))

from sklearn.preprocessing import MaxAbsScaler, PowerTransformer
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,QuantileTransformer  #이상치 제거를 하지 않은 상태에 RobustSCALER 사용하면 standardscaler보다 효과적이다 
scaler = PowerTransformer(method='yeo-johnson')
#scaler = MaxAbsScaler()
#scaler = PowerTransformer(method='box-cox')
scaler.fit(x)
x=scaler.transform(x) #x_test,x_pred,x_val 모두 transform만 해주면된다. => x_train이 기준으로 잡혔기 때문에
#print(np.max(x),np.min(x)) 
#print(np.max(x[0]))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=66)

print(x_train.shape)    #(80,3)
print(y_train.shape)    #(80,3)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(10,input_dim = 13,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
model.fit(x_train,y_train,epochs=100, batch_size=8, validation_split=0.2,verbose=0)

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
# loss: 13.444050788879395
# mae: 2.496389150619507
# RMSE :  3.6666128801911975
# R2 : 0.8391531759668155