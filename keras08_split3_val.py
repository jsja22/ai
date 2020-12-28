from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np



#np.array()
#array()

#1.데이터

x = np.array(range(1,101))
y = np.array(range(1,101))

x_train = x[:60]   # 순서 0부터 59번째 까지 :::: 값 1 ~ 60
x_val = x[60:80]   # 61 ~ 80
x_test = x[80:100] # 81 ~ 100
# 리스트 슬라이싱

y_train = y[:60]   # 순서 0부터 59번째 까지 :::: 값 1 ~ 60
y_val = y[60:80]   # 61 ~ 80
y_test = y[80:100] # 81 ~ 100
#리스트 슬라이싱

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, shuffle=True) # train 60퍼 test 40퍼 
print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#2.모델 구성

model = Sequential()
model.add(Dense(20,input_dim = 1,activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam', metrics= ['mae'])
model.fit(x_train, y_train, epochs=100, validation_split =0.2)

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss: ',loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
print(y_predict)

#loss:  0.007727864198386669
#mae:  0.08685149997472763
#shuffle = false

#loss:  0.0007547657587565482
#mae:  0.023014377802610397
#shuffle = True

#loss:  0.0036331559531390667
#mae:  0.0481703095138073
#validation_split = 0.2 

'''
#2.모델구성
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(111))
model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_val, y_val))

#4. 평가, 예측
results = model.evaluate(x_test, y_test, batch_size=1)
print('results : ', results)

y_predict = model.predict(x_test) #x_pred를 넣어서 예측한 값
#print('y_predict : ', y_predict)

#사이킷런이 뭔지 2분 검색 sikit-learn
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
print("RMSE :" , RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )
'''