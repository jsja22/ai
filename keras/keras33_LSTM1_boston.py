#lstm과 비교 하기
#conv1d로 

import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler   
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input ,LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score 

dataset = load_boston()
x= dataset.data
y= dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)  #fit에 정의한 x_train기준에 따르게 된다. 
x_val = scalar.transform(x_val)

print(x_train.shape)   # (323, 13)
print(x_test.shape)  # (102, 13)
print(x_val.shape)    #  (81, 13)

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)
x_val = x_val.reshape(x_val.shape[0],x_val.shape[1],1)


print(x_train.shape)   # (323, 13,1)
print(x_test.shape)  # (102, 13,1)
print(x_val.shape)     # (81, 13,1)


model = Sequential()
model.add(LSTM(128,input_shape = (13,1), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])


modelpath = 'C:/data/modelcheckpoint/k46_boston_{epoch:02d}-{val_loss:.4f}.hdf5'   
early_stopping = EarlyStopping(monitor='loss', patience=30,mode='auto') 
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_data =(x_val,y_val),callbacks=[early_stopping,cp])

loss,mae = model.evaluate(x_test,y_test,batch_size=8)
print('loss:',loss)
print('mae:',mae)
y_predict = model.predict(x_test)


def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))


r2 = r2_score(y_test,y_predict)
print("R2 :",r2)

#전처리 후 (x=x/711.)
#loss: 11.227853775024414
#mae: 2.465669870376587
#RMSE :  3.350798737628819
#R2 : 0.8656681310980472

#lstm
#loss: 24.352632522583008
#mae: 3.281953811645508
#RMSE :  4.934838638380806
#R2 : 0.7620247418087669

#lstm에서 성능 떨어짐.