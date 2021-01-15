import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
#함수정의
def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data.iloc[i:(i+size), 0:len(data.columns)]))
    return  np.array(a)

dataset = pd.read_csv('C:/data/csv/samsung.csv', index_col=0, header=0, encoding='cp949')

dataset['시가'] = dataset.loc[:,['시가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['고가'] = dataset.loc[:,['고가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['저가'] = dataset.loc[:,['저가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['종가'] = dataset.loc[:,['종가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['거래량'] = dataset.loc[:,['거래량']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['금액(백만)'] = dataset.loc[:,['금액(백만)']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['개인'] = dataset.loc[:,['개인']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['기관'] = dataset.loc[:,['기관']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['외인(수량)'] = dataset.loc[:,['외인(수량)']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['외국계'] = dataset.loc[:,['외국계']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['프로그램'] = dataset.loc[:,['프로그램']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)

print(type(dataset.iloc[0, 1]))   #<class 'numpy.float64'>

dataset = dataset.iloc[:662, :][::-1]

print(dataset.shape)      # (662, 14)
print(dataset)

dataset = dataset.sort_values(by='일자' ,ascending=True) 
print(dataset)

#결측값 제거
datasets_1 = dataset.iloc[:662,:]
datasets_2 = dataset.iloc[665:,:]

dataset = pd.concat([datasets_1,datasets_2],ignore_index=True)

print(dataset.shape)      # (662, 14)
print(dataset)


x = dataset.drop(['종가'], axis=1)
x = dataset.iloc[:,:5]
y = dataset.iloc[:,3]

print(x) #[662 rows x 5 columns]
print(y)  

print(x.shape)         # (662, 5)
print(y.shape)         # (662,)


size = 20
x_data = split_x(x,size)
y_target = y[size:]

print(x_data.shape)  #(643, 20, 5)
print(y_target.shape)  #(642,)

#train, test 분류
x_train, x_test, y_train, y_test = train_test_split(x_data[:-1], y_target, test_size=0.2, shuffle=False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1]*x_val.shape[2])

#x_data = x_data.reshape(x_test.shape[0], x_test.shape[1],x_train.shape[2])

print(x_train.shape)
print(x_test.shape)
print("전처리 전:print(x_test)",x_test)
#전처리

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_val = scaler.transform(x_val)

print("전처리 후:print(x_test)",x_test)


x_data1 = x_data.reshape(x_data.shape[0], x_data.shape[1]*x_data.shape[2])
x_data2 = scaler.transform(x_data1)
x_data = x_data2.reshape(x_data.shape[0],x_data.shape[1],x_data.shape[2])

x_train = x_train.reshape(x_train.shape[0], x_data.shape[1], x_data.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_data.shape[1], x_data.shape[2])
x_val = x_val.reshape(x_val.shape[0], x_data.shape[1], x_data.shape[2])


print("형변환후:print(x_test)",x_test)
print("형변환후:print(x_test)",y_test)
print(x_test.shape) #(129, 20, 5)
print(y_test.shape) #(129,)

np.save('C:/data/npy/samsung_x_train.npy', arr=x_train)
np.save('C:/data/npy/samsung_x_test.npy', arr=x_test)
np.save('C:/data/npy/samsung_y_train.npy', arr=y_train)
np.save('C:/data/npy/samsung_y_test.npy', arr=y_test)
np.save('C:/data/npy/samsung_x_data.npy', arr=x_data)
np.save('C:/data/npy/samsung_x_val.npy', arr=x_val)
np.save('C:/data/npy/samsung_y_val.npy', arr=y_val)

print(x_train.shape)
print(x_test)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#(513, 20, 5)
#(129, 20, 5)
#(513,)
#(129,)

# 모델
model = Sequential()

model.add(Conv1D(filters=32, kernel_size=5,padding="causal",activation="relu",input_shape=[20, 5]))
model.add(LSTM(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1))


# model.add(LSTM(256, activation='relu', input_shape=(20,5)))
# model.add(Dropout(0.2))
# model.add(Dense(1024, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.25))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

model.summary()
#loss = Huber()
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
modelpath = 'C:/data/modelcheckpoint/samsung_stock.hdf5'
es = EarlyStopping(monitor='val_loss', patience=30, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=16, validation_data=(x_val,y_val), verbose=1, callbacks=[es,cp])

model.save('C:/data/h5/samsung_stock.h5')

loss = model.evaluate(x_test, y_test,batch_size=16)
print('loss :', loss)


y_predict = model.predict(x_test)
print(x_test)
print(y_predict)
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE :" , RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )

print(x_data[-1].shape)
y_pred1 = x_data[-1].reshape(-1,x_train.shape[1],x_train.shape[2])
print(y_pred1.shape)

value = model.predict(y_pred1)
print('종가= ', value)

#RMSE : 4418.284712139119
#R2 :  0.7527832754717
#(20, 5)
#(1, 20, 5)
#종가=  [[89049.13]]

#RMSE : 3610.0151647628704
#R2 :  0.8349602701029095
#(20, 5)
#(1, 20, 5)
#종가=  [[88400.414]]