#실습
#R2를 음수가 아닌 0.5 이하로 줄이기 
#1. 레이어는 인풋과 아웃풋을 포함 6개 이상
#2. batch_size =1
#3 epochs =100이상
#4 데이터 조작 금지

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from numpy import arange,array
#np.array()
#array()

#1. 데이터
x_train = arange(1,11)
y_train = arange(1,11)
x_test = arange(11,16)
y_test = arange(11,16)
x_pred = array([16,17,18])


#2. 모델 구성
model = Sequential()
model.add(Dense(1000, input_dim =1, activation='relu'))
model.add(Dense(60))
model.add(Dense(999))
model.add(Dense(1000))
model.add(Dense(64))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer = 'adam', metrics = 'mae')
model.fit(x_train, y_train, epochs = 200, batch_size=1, validation_split=0.2)   #x_train , y_train을 validation_split =0.2로 20%를 나누어쓰겠다.


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