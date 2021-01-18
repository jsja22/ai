import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,Conv1D
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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
solar = np.load('C:/data/npy/solar.npy')
print(solar.shape)  #(52560, 8)

solar = solar.reshape(1095,48,8)
print(solar.shape) 

time_steps = 7
y_column = 2

x,y = split_xy(solar,time_steps,y_column)

print(x.shape) #(1087, 7, 48, 8)
print(y.shape) #(1087, 2, 48, 8)




scaler1 = MinMaxScaler()
scaler1.fit(x)
x = scaler1.transform(x)

print(x.shape)