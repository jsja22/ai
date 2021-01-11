#dnn구성

import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
#1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(24, activation='sigmoid', input_shape=(30,)))
model.add(Dropout(0.2))
model.add(Dense(24, activation='sigmoid'))
model.add(Dense(48, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(24, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))   #이진분류일때 마지막 activation은 무조건 sigmoid
model.summary()
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','mae'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,TensorBoard
modelpath = './modelCheckpoint/k46_cancer_{epoch:02d}-{val_loss:.4f}.hdf5'   
early_stopping = EarlyStopping(monitor='loss',patience=8,mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True, mode='auto')
tb = TensorBoard(log_dir='./graph',histogram_freq=0,write_graph=True,write_images=True) # 
hist = model.fit(x_train,y_train,epochs=1000, batch_size=8,validation_data=(x_val,y_val),verbose=0,callbacks=[early_stopping,cp,tb]) #batch_size 8 
loss = model.evaluate(x_test,y_test,batch_size=8)
print(loss) #(loss,acuracy)

y_pred = model.predict(x_test)
print(y_pred)
#print(y[-5:-1]) 

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
plt.plot(hist.history['mae'],marker='.',c='red',label='mae')
plt.plot(hist.history['val_mae'],marker='.',c='blue',label='val_mae')
plt.grid()

plt.title('mae')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()
#dnn model
#[0.08289500325918198, 0.9736841917037964, 0.04366859048604965]
#modelcheckpoint
#[0.08530618250370026, 0.9824561476707458, 0.057757798582315445]