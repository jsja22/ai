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
        tmp_x = dataset[i:x_end_number,:]
        tmp_y = dataset[x_end_number:y_end_number,:]     
        
        x.append(tmp_x)
        y.append(tmp_y)     
    return np.array(x), np.array(y)
solar= pd.read_csv('C:/data/csv/train.csv', index_col=0, header=0, encoding='cp949')
print(solar.columns) #Index(['Hour', 'Minute', 'DHI', 'DNI', 'WS', 'RH', 'T', 'TARGET'], dtype='object')
submit = pd.read_csv('C:/data/csv/sample_submission.csv',index_col =0, header =0)
test = pd.read_csv('C:/data/csv/sum_test.csv',index_col =0, header =0)
print(solar.shape) #(52560, 8)
print(solar.isnull().sum())

solar1 = solar.values
print(solar1)


np.save('C:/data/npy/solar.npy', arr=solar1)

#print(solar_x.shape)
#print(solar_y.shape)




