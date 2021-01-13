#dnn구성

import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,TensorBoard
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,LSTM, Conv1D, MaxPooling1D, Flatten

#1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

#2. 모델

model = Sequential()
model.add(Conv1D(filters=64, kernel_size =2 , activation='relu',padding='same', input_shape=(30,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Dense(24, activation='sigmoid'))
model.add(Conv1D(filters=64, kernel_size =2 , activation='relu',padding='same', input_shape=(10,1)))
model.add(Dense(48, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))   #이진분류일때 마지막 activation은 무조건 sigmoid
model.summary()
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','mae'])

early_stopping = EarlyStopping(monitor='loss',patience=8,mode='auto')
model.fit(x_train,y_train,epochs=1000, batch_size=8,validation_data=(x_val,y_val),verbose=0,callbacks=[early_stopping]) #batch_size 8 
loss = model.evaluate(x_test,y_test,batch_size=8)
print(loss) #(loss,acuracy)

y_pred = model.predict(x_test)
print(y_pred)
#print(y[-5:-1]) 

#dnn model
#[0.08289500325918198, 0.9736841917037964, 0.04366859048604965]
#modelcheckpoint
#[0.08530618250370026, 0.9824561476707458, 0.057757798582315445]
#LSTM
#loss: 0.6556 - accuracy: 0.6404 - mae: 0.4703

#conv1d
# [0.4683796465396881, 0.8005847334861755, 0.29500722885131836]

#conv1d에서 성능이 오히려 더좋아짐.