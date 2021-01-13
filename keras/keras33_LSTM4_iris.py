import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler #x값을 minmax전처리 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
dataset = load_iris()
x,y = load_iris(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)

#원핫 인코딩  (다중분류에 인해서 y를 원핫인코딩한것이다.=>y를 벡터화 하는것) 값이 있는부분에만 1을 넣고 나머지엔 0 
#다중분류에서 y는 반드시 원핫인코딩 해줘야함

y_train = y_train.reshape(-1,1) # reshape에서 -1은 재배열의 의미이다.
y_val = y_val.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_val = ohencoder.transform(y_val).toarray()
y_test = ohencoder.transform(y_test).toarray()

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

print(y)
print(x.shape) #(150, 4)
print(y.shape) #(150, 3)

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(4,1)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  #softmax : output node 다합치면 1이되는함수, 3인이유는 y.shape가 (150,3)이기에 3 softmax는 다중분류에서만 사용
                                          #가장 큰 값이 결정된다. 
model.summary()
 
#컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #다중분류에서는 loss='categorical_crossentropy'사용 
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath= 'C:/data/modelcheckpoint/k46_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss',patience=30,mode='auto')
hist = model.fit(x_train,y_train, epochs=200, batch_size=8,validation_data=(x_test,y_test),verbose=0,callbacks=[early_stopping,cp])
loss = model.evaluate(x_test,y_test,batch_size=8)
print(loss) #(loss,acuracy)

#실습1. acc 0.985 이상 
#실습2. predict 출력해볼것
'''
y_pred = model.predict(x[-5:-1])
print(y_pred)
print(y[-5:-1]) 

#[0.0403638556599617, 1.0, 0.03442172706127167] acc =1 굿!

#argmax 최대값 찾기
print(np.argmax(y_pred,axis=-1))
'''
y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred.argmax(axis=1))

#modelcheckpoint
#loss,acc: 0.001796197728253901 1.0

#LSTM 
#[0.3044010102748871, 0.8666666746139526]