#실습 : 19_1,2,3,4,5, Earlystopping까지 총 6개의 파일을 완성

import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x= dataset.data
y= dataset.target

print(x[:5])
print(y[:10])
print(x.shape,y.shape) #(442,10) (442,)

print("np.max(x),np.min(x) :",np.max(x),np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True  ,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.2,shuffle=True,random_state=66)

print(x_train.shape)    
print(y_train.shape)    

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)
print(np.max(x),np.min(x))
print(np.max(x[0]))
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(64,input_dim = 10,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = './modelCheckpoint/k46_diabets_{epoch:02d}-{val_loss:.4f}.hdf5'   
early_stopping = EarlyStopping(monitor='loss',patience=4,mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True, mode='auto')
hist = model.fit(x_train,y_train,epochs=1000, batch_size=8,validation_data=(x_val,y_val),callbacks=[early_stopping,cp]) #batch_size 8 

loss,mae = model.evaluate(x_test,y_test,batch_size=8)
print('loss:',loss)
print('mae:',mae)

y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)

import matplotlib.pyplot as plt
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


#전처리 후
#loss: 3343.386962890625
#mae: 47.89411926269531
#RMSE :  57.822028575912405
#R2 : 0.48484342421351634

#modelcheckpoint
#loss: 3601.672119140625
#mae: 48.273624420166016
#RMSE :  60.013929522224736
#R2 : 0.44504633017583817