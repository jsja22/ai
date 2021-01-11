import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
import matplotlib.pyplot as plt
dataset = load_iris()
x,y = load_iris(return_X_y=True)

from sklearn.preprocessing import MinMaxScaler #x값을 minmax전처리 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)

#원핫 인코딩  (다중분류에 인해서 y를 원핫인코딩한것이다.=>y를 벡터화 하는것) 값이 있는부분에만 1을 넣고 나머지엔 0 
#다중분류에서 y는 반드시 원핫인코딩 해줘야함
from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) # reshape에서 -1은 재배열의 의미이다.
y_val = y_val.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_val = ohencoder.transform(y_val).toarray()
y_test = ohencoder.transform(y_test).toarray()
'''
from tensorflow.keras.utils import to_categorical

y=to_categorical(y)
y_train= to_categorical(y_train)
y_test = to_categorical(y_test)
'''
print(y)
print(x.shape) #(150, 4)
print(y.shape) #(150, 3)

'''
print(dataset.DESCR)
print(dataset.feature_names) 

print(x.shape)  #(150,4)
print(y.shape)  #(150, )
print(x[:5])
print(y)
'''

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  #softmax : output node 다합치면 1이되는함수, 3인이유는 y.shape가 (150,3)이기에 3 softmax는 다중분류에서만 사용
                                          #가장 큰 값이 결정된다. 
model.summary()
 
#컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #다중분류에서는 loss='categorical_crossentropy'사용 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath= './modelCheckpoint/k46_iris_{epoch:02d}-{val_loss:.4f}.hdf5'
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
#modelcheckpoint
#loss,acc: 0.001796197728253901 1.0