import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#samsung = np.load('C:/data/npy/samsung_2.npy')

data = np.array(range(1,10))
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
time_steps =4
y_column =2
x, y = split_xy(data, time_steps, y_column)

print(x)
print(y)

print(x.shape)

