import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#함수정의
def split_x(data,size):
    a = []
    for i in range(data.shape[0] - size + 1):
        a.append(np.array(data[i:(i+size), :5]))
    return  np.array(a)

dataset = pd.read_csv('C:/data/csv/samsung.csv', index_col=0, header=0,encoding='cp949')

print(dataset.columns) #['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
                       #'외인(수량)', '외국계', '프로그램', '외인비'],

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

dataset = dataset[::-1]

print(dataset)
print(dataset.shape)

#시가,고가,저가,종가, 거래량, 금액 까지 50액면분할 해보자

dataset.loc[:'2018-05-03','시가':'종가']  = dataset.loc[:'2018-05-03','시가':'종가']/50
dataset.loc[:'2018-05-03':,'거래량'] = dataset.loc[:'2018-05-03','거래량']/50
#dataset.loc[:'2018-05-03':,'금액(백만)'] = dataset.loc[:'2018-05-03','금액(백만)']/50

#필요없는 열 삭제
dataset.drop(['금액(백만)','등락률','신용비', '개인', '기관','외인(수량)', '외국계', '프로그램', '외인비'],axis='columns', inplace=True)

dataset = dataset.sort_values(by='일자' ,ascending=True) 

dataset_data = dataset.iloc[0:2399,:]
dataset_target = dataset.iloc[1:2400,3]
x_pred = dataset.iloc[2399,:]
print(dataset_data)
print(dataset_target)


print(dataset_data.shape)   #(2399, 6)
print(dataset_target.shape) #(2399,)

dataset_data = dataset_data.to_numpy()
dataset_target = dataset_target.to_numpy()

print(dataset_data)
print(dataset_target)

print(dataset_data.shape)   #(2399, 6)
print(dataset_target.shape) #(2399,)


###################################### data 설정 완료 


# #결측값 제거
# datasets_1 = dataset.iloc[:662,:]
# datasets_2 = dataset.iloc[665:,:]

# dataset = pd.concat([datasets_1,datasets_2],ignore_index=True)

size = 20
dataset_data1 = split_x(dataset_data,size)
dataset_target = dataset_target[size:]
print(dataset_target.shape)  #(2379,)
print(dataset_data1.shape)  #((2380, 20, 6)


# print(dataset)
# print(dataset_target.shape)

#train, test 분류
x_train, x_test, y_train, y_test = train_test_split(dataset_data1[:-1], dataset_target, test_size=0.2, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
x_val = x_test.reshape(x_test.shape[0], x_val.shape[1]*x_val.shape[2])
#x_data = dataset_data1.reshape(x_test.shape[0], x_test.shape[1],x_train.shape[2])

print(x_train.shape) # (1522, 100)
print(x_test.shape)#(476, 100)


#전처리
scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)

# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
# x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)

x_data1 = dataset_data1.reshape(dataset_data1.shape[0], dataset_data1.shape[1]*dataset_data1.shape[2])
x_data2 = scalar.transform(x_data1)
x_data = x_data2.reshape(x_data2.shape[0],x_data2.shape[1],x_data2.shape[2])

x_train = x_train.reshape(x_train.shape[0], x_data.shape[1], x_data.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_data.shape[1], x_data.shape[2])


np.save('C:/data/npy/samsung_x_train.npy', arr=x_train)
np.save('C:/data/npy/samsung_x_test.npy', arr=x_test)
np.save('C:/data/npy/samsung_y_train.npy', arr=y_train)
np.save('C:/data/npy/samsung_y_test.npy', arr=y_test)
np.save('C:/data/npy/samsung_x_val.npy', arr=x_val)
np.save('C:/data/npy/samsung_y_val.npy', arr=y_val)
np.save('C:/data/npy/samsung_x_data.npy', arr=x_data)


# 모델
model = Sequential()

model.add(LSTM(256, activation='relu', input_shape=(x_train.shape[1],x_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam',metrics=['mae'])
modelpath = 'C:/data/modelcheckpoint/samsung_stock.hdf5'
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=64, validation_split=0.2, verbose=1, callbacks=[es,cp])

model.save('C:/data/h5/samsung_stock.h5')

result = model.evaluate(x_test, y_test,batch_size=64)
print('loss :',result[0] )
print('mae :', result[1] )
y_predict = model.predict(x_test)
print(x_test)
print(y_predict)

y_pred1 = x_data[-1].reshape(-1,x_train.shape[1],x_train.shape[2])
value = model.predict(y_pred1)
print('종가= ', value)

"""
def RMSE(y_test, y_predict) : 
    return np.sqrt(mean_squared_error(y_test, y_predict))  
print("RMSE :" , RMSE(y_test, y_predict))

r2 = r2_score(y_test, y_predict)
print("R2 : ", r2 )

y_pred = model.predict(x_test)
print(y_pred.shape)
print('종가= ', y_pred)

"""