import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,MaxPooling2D, Conv2D, Flatten
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number]
        tmp_y = dataset[x_end_number:y_end_number]     
            
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y) 

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
solar = np.load('C:/data/npy/solar.npy')
print(solar.shape)  #(52560, 9)

solar = solar.reshape(1095,48,9)
print(solar.shape)  #(1095, 48, 9)

time_steps = 7
y_column = 2

x,y = split_xy(solar,time_steps,y_column)

print(x.shape) #(1087, 7, 48, 9)
print(y.shape) #(1087, 2, 48, 9)

x = x[:,:,:,3:]
y = y[:,:,:,3:]
print(x.shape)  #(1087, 7, 48, 6)
print(y.shape)  #(1087, 2, 48, 6)
y = y.reshape(1087, 2*48*6)
#train test 분리
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

model = Sequential()
model.add(Conv2D(filters = 1024,kernel_size = 2, strides=1 , input_shape=(x.shape[1],x.shape[2],x.shape[3]),padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters = 512,kernel_size = 2, strides=1 , input_shape=(x.shape[1],x.shape[2],x.shape[3]),padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(2*48*6))

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=24,factor=0.5,verbose=1)  #5번해도 개선이 없으면 러닝레이트 (lr)0.5로 감축시키겠다는 뜻 
early_stopping = EarlyStopping(monitor='val_loss', patience=70, mode='auto')
modelpath= 'C:/data/modelcheckpoint/solar_system1_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, batch_size=32, epochs=700, validation_split=0.2, callbacks=[early_stopping, cp,reduce_lr])


model.save('C:/data/h5/solar_system1.h5')
#model = load_model('C:/data/h5/samsung_kodex.h5')

loss, mae = model.evaluate(x_test, y_test, batch_size=32)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("loss, mae : ", loss, mae)
print("RMSE : ", RMSE(y_test, y_predict))
print("R2 : ", r2)

print(y_predict)
print(y_predict.shape) 
