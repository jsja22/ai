import numpy as np
import tensorflow as tf

data = np.load('C:/data/npy/samsung_2.npy')
print(data.shape) #(1085, 6)

def split_xy(dataset, x_row, y_row,x_col,y_col):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + x_row
        y_end_number = x_end_number + y_row

        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number]
        tmp_y = dataset[x_end_number:y_end_number]     
            
        x.append(tmp_x)
        y.append(tmp_y)
    x = np.array(x)
    y = np.array(y)
    x= x[:,:,:x_col]
    y= y[:,:,:y_col]

    print(x.shape) # (1080, 4, 6)
    print(y.shape) # (1080, 2, 6)
    
    return np.array(x), np.array(y) 
x_row =4
y_row =2
x_col = 3
y_col = 1
x, y = split_xy(data,  x_row, y_row,x_col,y_col)

print(x)
print(y)

print(x.shape)  #(1080, 4, 3)#
print(y.shape)  #(1080, 2, 1)
