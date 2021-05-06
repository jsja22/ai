from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import arange,array
#np.array()
#array()

#1. 데이터
x= np.array(range(1,101))
y= np.array(range(101,201))

x_train = x[:60]    #순서 0-59번째까지 :::: 값 1-60
x_val = x[60:80] #61-80
x_test = x[80:]  #81-100
#리스트의 슬라이싱

y_train = y[:60]    #순서 0-59번째까지 :::: 값 1-60
y_val = y[60:80] #61-80
y_test = y[80:]  #81-100
#리스트의 슬라이싱

'''
x_train = arange(1,11)
y_train = arange(1,11)
x_test = arange(11,16)
y_test = arange(11,16)
x_pred = array([16,17,18])
'''

#2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim =1, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics = 'mae')
model.fit(x_train, y_train, epochs = 100, batch_size=1, validation_data=(x_val,y_val))   #x_train , y_train을 validation_split =0.2로 20%를 나누어쓰겠다.


#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print("mse, mae= ",results)

y_predict = model.predict(x_test)          #x_pred를 넣어서 얻은 결과값은 y_pred
#print("y_predict :", y_predict)

#sklearn
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
#print("RMSE : ",RMSE(y_test,y_predict))
#print("mse :",mean_squared_error(y_predict,y_test))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)
model.summary()
