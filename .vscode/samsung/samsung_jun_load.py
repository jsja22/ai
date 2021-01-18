import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error,r2_score

#함수정의
def split_x(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

#           시가       고가       저가       종가         거래량
df = np.load('C:/data/npy/samsung_1.npy')
print(df)

#x,y 데이터 
x = df[:,:-1]
y= df[:,3]
print(y)
#전처리

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

#데이터 split
size=6
x_data = split_x(x, size)
print(x_data.shape) #(2393, 6, 5)

x = x_data[:-1,:,:]
y= y[size:]
x_pred = x_data[-20:,:,:]
print("x_pred",x_pred)
print("x_pred",x_pred.shape)

print(y.shape)
print(y)  #[17780. 17940. 18480. ... 90600. 89700. 89700.]

####################################

#train_test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=99)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=99)
print(x_train.shape, x_test.shape)  #(1916, 6, 5) (479, 6, 5)
print(y_train.shape, y_test.shape)  #(1916,) (479,)

#모델 구성

model = Sequential()

model.add(LSTM(256, activation='relu', input_shape=(6,x.shape[2])))
#model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
"""
model.add(LSTM(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(6,x.shape[2])))
#model.add(Dropout(0.2))
model.add(Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
#model.add(Dropout(0.2))
model.add(Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
#model.add(Dropout(0.2))
model.add(Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.01)))
model.add(Dense(1))
"""

#컴파일 및 훈련
#sgd = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=80, mode='auto')
modelpath= 'c:/data/modelcheckpoint/samsung_ckp.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist= model.fit(x_train, y_train, batch_size=64, epochs=2000, validation_data=(x_val,y_val),verbose=1, callbacks=[es, cp])

model.save('C:/data/h5/samsung_1.h5')

loss, mae = model.evaluate(x_test, y_test, batch_size=64)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("loss, mae : ", loss, mae)
print("RMSE : ", RMSE(y_test, y_predict))
print("R2 : ", r2)

y_predict = model.predict(x_pred)
p_p = int(y_predict[-1])  

print("1/15일 예측 삼성 종가 : ", p_p)
print(y_predict.shape)

for i in range(len(y_predict)):
    print("실제 종가",y[-(y_predict.shape[0])+i],"예측 종가", y_predict[i] )



plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y)[-20:], label='actual')
plt.plot(y_predict, label='prediction')
plt.legend()
plt.show()

#RMSE :  2145.247416108882
#R2 :  0.974547665209186
#1/15일 예측 삼성 종가 :  90060


#loss, mae :  1350901.125 883.1961059570312
#RMSE :  1162.2826651221956
#R2 :  0.9926943452143626
#1/15일 예측 삼성 종가 :  89130

#loss, mae :  1188342.125 684.1337280273438
#RMSE :  1090.111061154947
#R2 :  0.9935734614319652
#1/15일 예측 삼성 종가 :  92576

#loss, mae :  495872.65625 502.39056396484375
#RMSE :  704.1822552945873
#R2 :  0.9973183272453855
#1/15일 예측 삼성 종가 :  92604
#(20, 1)