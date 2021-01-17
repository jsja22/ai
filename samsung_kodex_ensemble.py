import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Input, LSTM, Dropout, Conv1D, Flatten, MaxPooling1D, LeakyReLU
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import concatenate, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#함수정의
def split_data(seq, size):
    aaa = []
    for i in range(len(seq)-size+1):
        subset = seq[i : (i+size)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

# 1. 데이터
samsung = np.load('C:/data/npy/samsung_2.npy')
kodex = np.load('C:/data/npy/kodex.npy')

print(samsung.shape) #(1085, 6)
print(kodex.shape) #(1085, 6)
print(samsung)
print(kodex)

x1 = samsung
x2 = kodex
y= x1[:,0]

print(x1)
print(x2)

print(x1.shape, x2.shape) #(1085, 6) (1085, 6)
print(y.shape) #(1085,)

#전처리
scaler1 = MinMaxScaler()
scaler1.fit(x1)
x1 = scaler1.transform(x1)

scaler2 = MinMaxScaler()
scaler2.fit(x2)
x2 = scaler2.transform(x2)

print(x1)
print(x2)
################################전처리 완료
size1 = 12
x1 = split_data(x1, size1)
x2 = split_data(x2, size1)

x1 = x1[:-2,:,:]
x2 = x2[:-2,:,:]

size2 =  2

y= split_data(y,size2)
y= y[12:]

x1_pred = x1[-20:,:,:]
x2_pred = x2[-20:,:,:]
print(x1.shape) #(1072, 12, 6)
print(x2.shape) #(1072, 12, 6)
print(y.shape)  #(1078, 2)


#train test 분류
x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y, train_size=0.8, shuffle=True, random_state=99)
x1_train, x1_val, x2_train, x2_val, y_train, y_val = train_test_split(x1_train, x2_train, y_train, train_size=0.8, shuffle = True,random_state=99)

#모델구성
input1 = Input(shape=(6,6))
dense1 = LSTM(256,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input1)
dense1 = Dense(256,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense1)
dense1 = Dense(128,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense1)
dense1 = Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense1)
dense1 = Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense1)
dense1 = Dense(16,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense1)
dense1 = Dense(2)(dense1)


input2 = Input(shape=(6,6))
dense2 = LSTM(128,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input2)
dense2 = Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense2)
dense2 = Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense2)
dense2 = Dense(16,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense2)
dense2 = Dense(4,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense2)
dense2 = Dense(4,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(dense2)
dense2 = Dense(2)(dense2)

merge1 = concatenate([dense1, dense2])
middle1 = Dense(128,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(merge1)
middle1 = Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(middle1)
middle1 = Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(middle1)
middle1 = Dense(64,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(middle1)
middle1 = Dense(16,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(middle1)
middle1 = Dense(4,activation=tf.keras.layers.LeakyReLU(alpha=0.01))(middle1)
outputs = Dense(2)(middle1)

model = Model(inputs=[input1,input2], outputs=outputs)

model.summary()

#컴파일 및 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
es = EarlyStopping(monitor='val_loss', patience=80, mode='auto')
modelpath= 'c:/data/modelcheckpoint/samsung_kodex_ckp.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit([x1_train,x2_train], y_train, batch_size=64, epochs=1000, validation_data=([x1_val,x2_val],y_val),verbose=1, callbacks=[es, cp])

model.save('C:/data/h5/samsung_kodex.h5')

loss, mae = model.evaluate([x1_test,x2_test], y_test, batch_size=64)
y_predict = model.predict([x1_test, x2_test])
r2 = r2_score(y_test, y_predict)
print("loss, mae : ", loss, mae)
print("RMSE : ", RMSE(y_test, y_predict))
print("R2 : ", r2)

y_predict = model.predict([x1_pred,x2_pred])
print(y_predict)
print(y_predict.shape) #(20, 2)

for i in range(len(y_predict)):
    print("다 다음날의 실제 시가",y[-(y_predict.shape[0])+i,-1],"다 다음날의 예측 시가", y_predict[i][1] )


plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y)[-20:,1], label='actual')
plt.plot(y_predict[:,1], label='prediction')
plt.legend()
plt.grid()
plt.show()

print("1/19일 예측 시가는?", y_predict[-1,1])

#loss, mae :  10918207.0 1807.4273681640625
#RMSE :  3304.271139318148
#R2 :  0.8926200914040961

#1/19일 예측 시가는? 92482.266

#1/19일 예측 시가는? 91997.305


 
