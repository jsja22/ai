
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
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, test_size=0.2, shuffle=False) # train 60퍼 test 40퍼

#train_size 를 0.9로 하면 range가 (0,1)을 벗어남으로 에러가 발생한다. 0.7을 하였을때는 정상범위내에서  #loss:  0.0020308936946094036
#mae:  0.04068293422460556 확인가능  0.8일떄는 loss:  0.0006389260524883866 mae:  0.021098803728818893  -> loss값 확 줄어듬
#train_size를 0.1로하면 loss:  0.014265969395637512 mae:  0.10108375549316406    -> train size를 줄일수록 loss 커짐
# test_size를 0.8로 키우면 loss:  0.0019206745782867074  mae:  0.04379231482744217  -> 다시 loss 작아짐 but loss값이 tet_size =0.2, train_size=0.8 일때보다 loss가 훨씬 큼.

print(x_train)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,train_size=0.8, shuffle=False) 
print(x_train)
print(x_train.shape)
print(x_val.shape)
print(y_train.shape)
print(y_val.shape)

#2.모델 구성

model = Sequential()
model.add(Dense(20,input_dim = 1,activation='relu'))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(5))
model.add(Dense(1))


#3. 컴파일, 훈련

model.compile(loss='mse',optimizer='adam', metrics= ['mae'])
model.fit(x_train, y_train, epochs=100, validation_data =(x_val,y_val))

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('loss: ',loss)
print('mae: ', mae)

y_predict = model.predict(x_test)
print(y_predict)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  #sqrt는 루트
print("RMSE :" , RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )


#loss:  0.007727864198386669
#mae:  0.08685149997472763
#shuffle = false

#loss:  0.0007547657587565482
#mae:  0.023014377802610397
#shuffle = True

#loss:  0.0036331559531390667
#mae:  0.0481703095138073
#validation_split = 0.2 

#loss:  0.04155946522951126
#mae:  0.20172885060310364
#validation data add
