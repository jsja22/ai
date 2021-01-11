import numpy as np
import tensorflow as tf
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt

dataset = load_wine()
print(dataset.DESCR)
print(dataset.feature_names)

x=dataset.data
y=dataset.target

print(x)
print(y)
print(x.shape)
print(y.shape)

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) 
y_val = y_val.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_val = ohencoder.transform(y_val).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #다중분류에서는 loss='categorical_crossentropy'사용 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath= './modelCheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss',patience=30,mode='auto')
hist = model.fit(x_train,y_train, epochs=2000, batch_size=8,validation_data=(x_test,y_test),verbose=1,callbacks=[early_stopping,cp])
loss,acc= model.evaluate(x_test,y_test,batch_size=8)
print("loss,acc:",loss,acc)

y_pred = model.predict(x[-14:-1])
print(y_pred)
print(y[-14:-1])

plt.figure(figsize=(10,6))  #figsize 판깔아주는거 #면적을 잡아주는것

plt.subplot(2,1,1)  #(2,1)짜리 그림을 만들겠다 (2행1열중에 첫번쨰)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()

plt.title('cost loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2,1,2)  #(2,1)짜리 그림을 만들겠다 (2행1열중에 두번쨰)
plt.plot(hist.history['accuracy'],marker='.',c='red',label='accuracy')
plt.plot(hist.history['val_accuracy'],marker='.',c='blue',label='val_accuracy')
plt.grid()

plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

#dense 모델 loss,acc: 0.04328326880931854 0.9722222089767456

#modelcheckpoint
#loss,acc: 0.23765717446804047 0.9722222089767456