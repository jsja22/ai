#보스턴 모델 구성 r2 ->0.99로 올리자

import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x= dataset.data
y= dataset.target

from sklearn.preprocessing import MinMaxScaler   
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)  #fit에 정의한 x_train기준에 따르게 된다. 
x_val = scalar.transform(x_val)

print(x_train.shape)    #(80,3)
print(y_train.shape)    #(80,3)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(128,input_dim = 13,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = './modelCheckpoint/k46_boston_{epoch:02d}-{val_loss:.4f}.hdf5'   
early_stopping = EarlyStopping(monitor='loss', patience=30,mode='auto')  #loss의 최소값 loss값까지 더 떨어지지 않는 값을 10번참겠다는 뜻
 
                                                                        #
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True,mode='auto')
#model.fit(x_train,y_train,epochs=100, batch_size=8, validation_split=0.2,verbose=0)
model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_data =(x_val,y_val),callbacks=[early_stopping,cp])
#epochs를 100으로 할때 0.91 200으로 할때 0.93 으로 증가 500으로 할때 0.906  결국 epochs를 늘여봤자 좋은건 없음 
#과적합 되기전에 끊어주자 -> earlystopping 

loss,mae = model.evaluate(x_test,y_test,batch_size=8)
print('loss:',loss)
print('mae:',mae)
y_predict = model.predict(x_test)

from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print("RMSE : ",RMSE(y_test,y_predict))

from sklearn.metrics import r2_score 
r2 = r2_score(y_test,y_predict)
print("R2 :",r2)

#전처리 후 (x=x/711.)
#loss: 11.227853775024414
#mae: 2.465669870376587
#RMSE :  3.350798737628819
#R2 : 0.8656681310980472
