import pandas as pd
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,MaxPooling2D, Conv2D, Flatten, Reshape, Conv1D, MaxPooling1D, Input,LeakyReLU
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam

def preprocess_data (data, is_train=True) :
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    if is_train == True :    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날 TARGET을 붙인다.
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날 TARGET을 붙인다.
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 이틀치 데이터만 빼고 전체
    elif is_train == False :         
        # Day, Minute 컬럼 제거
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :] # 마지막 하루치 데이터

def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number]
        tmp_y = dataset[x_end_number:y_end_number,:,-2:]         
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)


'''
def split_xy(dataset, time_steps, y_column):
    x, y1,y2 = list(), list(),list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number]
        tmp_y1 = dataset[x_end_number:y_end_number,:,-2]     
        tmp_y2 = dataset[x_end_number:y_end_number,:,-1]     
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)

    return np.array(x), np.array(y1) ,np.array(y2) 
'''
def tilted_loss(q,y,f):
    e = (y-f)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

#data 준비
train_df = pd.read_csv('C:/data/csv/solar/train/train.csv')
print(train_df .shape)     #(52560, 9)
print(train_df .tail())
sample = pd.read_csv('C:/data/csv/solar/sample_submission.csv')

train_data = preprocess_data(train_df)
print(train_data.columns)
#Index(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Target1', 'Target2'], dtype='object')

train = train_data.to_numpy()
print(train.shape) #(52464, 9)

train = train.reshape(-1,48,9)
print(train.shape)  #(1093, 48, 9)

time_steps =7
y_column = 2 
x,y = split_xy(train,time_steps,y_column)
# x,y1,y2 = split_xy(train,time_steps,y_column)

x= x[:,:,:,3:]
#y1 = y1
#y2 = y2
print(x.shape)  #(1085, 7, 48, 6)
#print(y1.shape) #(1085, 2, 48)
#print(y2.shape) #(1085, 2, 48)

y = y.reshape(-1,96)
#y1 = y1.reshape(-1,96)
#y2 = y2.reshape(-1,96)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def Conv2dmodel():
    drop = 0.2
    input1 = Input(shape=(7,48,6))
    dense1 = Conv2D(256, 2,  padding='same')(input1)
    dense1=(LeakyReLU(alpha = 0.05))(dense1)
    dense1 = Conv2D(128, 2, padding='same')(dense1)
    dense1=(LeakyReLU(alpha = 0.05))(dense1)
    dense1 = Conv2D(64, 2,  padding='same')(dense1)
    dense1=(LeakyReLU(alpha = 0.05))(dense1)
    dense1 = Flatten()(dense1)
    dense1 = Dense(128)(dense1)
    dense1 = Dense(96)(dense1)
    outputs = Dense(96)(dense1)

    model = Model(inputs=input1, outputs=outputs)

    return model

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min')
lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    
    model = Conv2dmodel()

    i=10*q
    cp = ModelCheckpoint('C:/data/modelcheckpoint/solar_model2_{}.h5'.format(i), monitor='val_loss', mode='auto', save_best_only=True)
    model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer='adam')
    model.fit(x, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, cp,lr], verbose=1)

    c = []
    for i in range(81):
        test_data = pd.read_csv('C:/data/csv/solar/test/%d.csv'%i)
        test_data.drop(['Hour','Minute','Day'], axis =1, inplace = True)
        test_data = test_data.to_numpy()  
        test_data = test_data.reshape(1,7,48,6)
        
        y_pred = model.predict(test_data)
        y_pred = y_pred.reshape(2,48)
          
        a = []
        for j in range(2):
            b = []
            for k in range(48):
                b.append(y_pred[j,k])
            a.append(b)   
        c.append(a)
    c = np.array(c)
    c = c.reshape(81*2*48,)
    sample.loc[:, "q_%d"%q] = c
sample = sample.iloc[:,:-1]
print(sample)
sample.to_csv('C:/data/csv/solar/sample_submission4.csv', index=False)