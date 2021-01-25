#1095일의 데이터를 같은시간끼리 묶어서 해보기!

import pandas as pd
import numpy as np
import os
import glob
import sys
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential,Model ,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,MaxPooling2D, Conv2D, Flatten, Reshape, Conv1D, MaxPooling1D, Input,LeakyReLU
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
def split_to_seq(data): 
    tmp = []
    for i in range(48):
        tmp1 = pd.DataFrame()
        for j in range(int(len(data)/48)):
            tmp2 = data.iloc[j*48+i,:]
            tmp2 = tmp2.to_numpy()
            tmp2 = tmp2.reshape(1,tmp2.shape[0])
            tmp2 = pd.DataFrame(tmp2)
            tmp1 = pd.concat([tmp1,tmp2])
        x = tmp1.to_numpy()
        tmp.append(x)
    return np.array(tmp)
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
day = 7
def preprocess_data(data,is_train=True):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(1,'GHI',data['DNI']*data['cos']+data['DHI'])
    temp = data.copy()
    temp = temp[['DHI','DNI','GHI','T','WS','RH','TARGET']]
    
    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        return temp.iloc[-48*day:, :]

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]
        tmp_y1 = data[x_end-1:x_end,-2]
        tmp_y2 = data[x_end-1:x_end,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))

def Conv1dmodel():
    model = Sequential()
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (7,7)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))
    return model

#train 데이터 준비 
train= pd.read_csv('C:/data/csv/solar/train/train.csv',index_col=None, header=0)
print(train .shape)     #(52560, 9)
print(train .tail())
submission = pd.read_csv('C:/data/csv/solar/sample_submission.csv')

df_train = preprocess_data(train,is_train=True)
print(df_train.columns) #Index(['DHI', 'DNI', 'GHI', 'T', 'WS', 'RH', 'TARGET', 'TARGET1', 'TARGET2'], dtype='object')

x_train_data = df_train.iloc[:,:-2]
scale = StandardScaler()
scale.fit(x_train_data)
x_train_data = scale.transform(x_train_data)
x_train = split_to_seq(df_train)
print(x_train.shape) #(48, 1093, 9)

# test data 준비

x_test = []

for i in range(81):
    file_path = 'C:/data/csv/solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp,is_train=False)
    temp = scale.transform(temp)
    temp = pd.DataFrame(temp)
    temp = split_to_seq(temp)
    x_test.append(temp)

test = np.array(x_test)
print(test.shape)

x,y1,y2 = [],[],[]
for i in range(48):
    tmp1,tmp2,tmp3 = split_xy(x_train[i],day)
    x.append(tmp1)
    y1.append(tmp2)
    y2.append(tmp3)

x = np.array(x) 
y1 = np.array(y1) 
y2 = np.array(y2) 
print(x.shape)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 8, factor = 0.3, verbose = 1)
epochs = 100000
bs = 32
hour = 0
for i in range(48):
    x1_train, x1_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x[i],y1[i],y2[i], train_size = 0.7,shuffle = True, random_state = 0)
    
    print(x1_val.shape)
    print(y1_val.shape)
    print(y2_val.shape)

    if i%2 == 0:
      minute = 0
    elif i%2 == 1:
      minute = 30
    hour += hour+int(i/2) 
    
    for j in quantiles:
        print("##############내일 {}시,{}분, q_0.{} 훈련 시작!!###########".format(hour,minute,j))
        model = Conv1dmodel()
        filepath_cp = f'C:/data/modelcheckpoint/solar_checkpoint_0124_{i:2d}_day1_{j:.1f}.hdf5'
        cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        model.fit(x1_train,y1_train,epochs = epochs, batch_size = bs, validation_data = (x1_val,y1_val),callbacks = [es,cp,lr])

    
    for j in quantiles:
        print("##############모레 {}시,{}분 q_0.{} 훈련 시작!!############".format(hour,minute,j))
        model = Conv1dmodel()
        filepath_cp = f'C:/data/modelcheckpoint/solar_checkpoint_0124_{i:2d}_day2_{j:.1f}.hdf5'
        cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        model.fit(x1_train,y2_train,epochs = epochs, batch_size = bs, validation_data = (x1_val,y2_val),callbacks = [es,cp,lr]) 



