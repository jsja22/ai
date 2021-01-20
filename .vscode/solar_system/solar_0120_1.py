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
from tensorflow.keras.models import Sequential,Model ,load_model
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

# model 학습 
# for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

#     print("##### {} model fit start! #####".format(i))

#     q = i
    
#     model =  Conv2dmodel()
#     early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min')
#     cp = ModelCheckpoint('C:/data/modelcheckpoint/solar_0120model1_{}.h5'.format(i), monitor='val_loss', mode='auto', save_best_only=True)
#     lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

#     model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer='adam')
#     model.fit(x, y, epochs=1, batch_size=128, validation_split=0.2, callbacks=[early_stopping, cp,lr], verbose=1)


#x_test = pd.concat(test_data)
#print(x_test.shape) #(3888, 7) # 81day 48 hour 8 columns
#print(x_test.columns)   #Index(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T'], dtype='object')
# 모델 trainning
"""
for q in ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    print("##### {} model fit start! #####".format(q))
    model =  Conv2dmodel()
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min')
    cp = ModelCheckpoint('C:/data/modelcheckpoint/solar_0120model2_{}.h5'.format(q), monitor='val_loss', mode='auto', save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.5, verbose=1)

    model.compile(loss=lambda y,f: tilted_loss(q,y,f), optimizer='adam')
    model.fit(x, y, epochs=1, batch_size=128, validation_split=0.2, callbacks=[early_stopping, cp,lr], verbose=1)
    # model_1_path = 'C:/data/modelcheckpoint/solar_0120model1_{}.h5'.format(q)
    # #print(model_1_path.shape)
    # print(model_1_path)  
    
    test_data = []
    for i in range(81):
        file_path = pd.read_csv('C:/data/csv/solar/test/%d.csv'%i)  
        file_path.drop(['Hour','Minute','Day'], axis =1, inplace = True)
        test = file_path.to_numpy()  
        test = test.reshape(1,7,48,6)
        y_pred = model.predict(test)
        #print(y_pred.shape)
        y_pred = y_pred.reshape(2,48)
        print(y_pred.shape)
        a = []
        for j in range(2):
            b = []
            for k in range(48):
                b.append(y_pred[j,k])
            a.append(b)
        test_data.append(a)
    test_data = np.array(test_data)
    test_data = test_data.reshape(81*2*48,)
    sample.loc[:, "q_%d"%q] = test_data
    test_data = pd.DataFrame(test_data)
    test_data.to_csv('C:/data/csv/predict1/0120predict3_{}.csv'.format(q))

#sample= sample.iloc[:,:-1]
#sample.to_csv('C:/data/csv/solar/sample_submission5.csv', index=False)
    # model_1 = load_model(model_1_path, compile=False)
    # predict_1 = model_1.predict(test)
    # print(predict_1.shape)
    # predict_1 = predict_1.reshape(-1, 1)
    # print(predict_1.shape)
    # predict_1 = pd.DataFrame(predict_1)
    # predict_1.to_csv('C:/data/csv/predict1/0120predict1_{}.csv'.format(i))


# print(predict_1)
# print(predict_1.shape) #(96, 1)

pred1 = np.zeros((81*48*2,9))
for i , num in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    temp = pd.read_csv('C:/data/csv/predict1/0120predict3_'+str(num)+'.csv',index_col=None, header=0)
    #temp = pd.read_csv(file_path)
    temp.drop('Unnamed: 0', axis=1)
    temp = np.array(temp)
    print(temp.shape)
    temp = temp[:,0]
    print(temp.shape)
    #print(temp.tail())
    #temp = temp.reshape(-1)
    print(temp.shape) #(15552,)
    pred1[:,i] = temp
pred1 = pd.DataFrame(pred1)
print(pred1.shape)
print(pred1)
"""
result = pd.read_csv('C:/data/csv/predict1/result2.csv')
sample.iloc[1:, 1:] = result.to_numpy()
print(result.shape)
sample.iloc
# #sample to numpy
# submission = pd.concat([pred1])
# submission[submission.values<0] = 0
# sample.iloc[:, 1:] = submission.to_numpy()
sample.to_csv('C:/data/csv/solar/sample/sample_submission2.csv',header=True, index=False)


