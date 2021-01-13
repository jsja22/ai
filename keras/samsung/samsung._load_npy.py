import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Conv1D,MaxPool1D,Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data_samsung = np.load('C:/data/npy/samsung1.npy',allow_pickle=True)
data_samsung = np.load('C:/data/npy/samsung2.npy',allow_pickle=True)

print(data_samsung)
print(data_samsung.shape) #(2400, 14)

# x= data[:662,:4]
# y= data[:662,3:4]  
# print(x)
# print(x.shape)
# print(y)
# print(y.shape)

#(결측치가 있는 3개의행 제거)
data_samsung1 = data_samsung.iloc[:662,:]
data_samsung2 = data_samsung.iloc[665:,:]

data_samsung = pd.concat(data_samsung1,data_samsung2)
print(data_samsung)

"""
def split_xy(dataset, time_steps, y_column):
    x,y = list(), list()
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

x,y = split_xy(data_samsung,5,1)

print(x)  #5행씩 분리 날짜별로 과거순으로 정렬 
print(y)
print(y.shape)  # (2395, 1, 14)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True  ,random_state=66)
#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)


#x_train = x_train.reshape(1916,5,14,1).astype('float32')/255.  #(0~1사이로 수렴) =>전처리
#x_test = x_test.reshape(479,5,14,1)/255.

print(x_train.shape)    #(1916, 5, 14)
print(x_test.shape)     #(479, 5, 14)
print(y_train.shape)    #(1916, 1, 14)
print(y_test.shape)     #(479, 1, 14)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_train.shape[2],1)
#x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

print(x_train.shape)    #(1916, 5, 14, 1)
print(x_test.shape)     # (479, 5, 14, 1)


# y_train = y_train.reshape(-1,1) 
# #y_val = y_val.reshape(-1,1)
# y_test = y_test.reshape(-1,1)

# ohencoder = OneHotEncoder()
# ohencoder.fit(y_train)
# y_train = ohencoder.transform(y_train).toarray()
# #y_val = ohencoder.transform(y_val).toarray()
# y_test = ohencoder.transform(y_test).toarray()

model = Sequential()
model.add(LSTM(128,input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))
model.compile(loss = 'mse',optimizer = 'adam')
plt.figure(figsize =(10,6))
plt.plot(x['종가'])

plt.show()



#data를 먼저 662,5 사이즈로 분할
#data를 가지고 컨볼루션 모델함수 구성
#컴파일 훈련
#y_predict값 출력
"""
