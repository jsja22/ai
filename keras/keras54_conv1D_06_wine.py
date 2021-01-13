import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

dataset = load_wine()

x=dataset.data
y=dataset.target

print(x)
print(y)
print(x.shape)
print(y.shape)

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


y_train = y_train.reshape(-1,1) 
y_val = y_val.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_val = ohencoder.transform(y_val).toarray()
y_test = ohencoder.transform(y_test).toarray()


model = Sequential()
model.add(Conv1D(filters=64, kernel_size =2 , activation='relu',padding='same', input_shape=(13,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #다중분류에서는 loss='categorical_crossentropy'사용 

early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')

early_stopping = EarlyStopping(monitor='loss',patience=24,mode='auto')
model.fit(x_train,y_train, epochs=1000, batch_size=8,validation_data=(x_val,y_val),verbose=1,callbacks=[early_stopping])
loss,acc= model.evaluate(x_test,y_test,batch_size=8)
print("loss,acc:",loss,acc)

y_pred = model.predict(x[-14:-1])
print(y_pred)
print(y[-14:-1])


#dense 모델 loss,acc: 0.04328326880931854 0.9722222089767456

#modelcheckpoint
#loss,acc: 0.23765717446804047 0.9722222089767456

#lstm
#loss,acc: 1.2798327207565308 0.9722222089767456
