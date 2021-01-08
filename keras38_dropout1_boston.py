import numpy as np
from sklearn.datasets import load_boston

dataset = load_boston()
x= dataset.data
y= dataset.target
print(x.shape)  #(506,13)
print(y.shape)  #(506, )
print("===================")
print(x[:5])
print(y[:10])

#x=np.transpose(x)
#y=np.transpose(y)

print(np.max(x),np.min(x))  #711.0 0.0 
print(dataset.feature_names)
#print(dataset.DESCR)

#데이터 전처리(MinMax)
#x = x/711.   => 이는 잘못된 전처리 칼람별로 민맥스값이 다르기때문에
#x가 0부터 시작한다는걸 안다는 가정하에하지만 모를경우엔
#x = (x-최소)/(최대-최소)
# = (x - np.min(x))/(np.max(x)-np.min(x))
print(np.max(x[0]))

from sklearn.preprocessing import MinMaxScaler   #칼람별로 전처리를 하게된다.

'''
scaler = MinMaxScaler()
scaler.fit(x)
x=scaler.transform(x) #x_test,x_pred,x_val 모두 transform만 해주면된다. => x_train이 기준으로 잡혔기 때문에
print(np.max(x),np.min(x)) 
print(np.max(x[0]))
'''

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
from tensorflow.keras.layers import Dense, Input, Dropout

model = Sequential()

model.add(Dense(128,input_dim = 13,activation='relu'))
model.add(Dropout(0.3))  # (25개를 사용안하겠다는뜻)
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1,activation='relu'))
model.summary()

model.compile(loss='mse',optimizer='adam',metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30,mode='auto')  #loss의 최소값 loss값까지 더 떨어지지 않는 값을 10번참겠다는 뜻
                                                                         #

#model.fit(x_train,y_train,epochs=100, batch_size=8, validation_split=0.2,verbose=0)
model.fit(x_train, y_train, epochs=2000, batch_size=8, validation_data =(x_val,y_val),callbacks=[early_stopping])
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

#x통째로 전처리함 . 민맥스 스칼라 뒤에 흠... 난왜 더 안좋아졌지 다시 확인해보자
#loss: 14.027112007141113
#mae: 2.2565598487854004
#RMSE :  3.7452785280829186
#R2 : 0.832177335462187

#patience =30으로 했을때 
#loss: 12.144424438476562
#mae: 2.085576295852661
#RMSE :  3.484884729222626
#R2 : 0.8547021441976861

#드랍아웃전
#loss: 5.731499671936035
#mae: 1.9138332605361938
#RMSE :  2.3940550210477087
#R2 : 0.9314273986202272

#드랍아웃후
#loss: 9.547056198120117
#mae: 2.3115293979644775
#RMSE :  3.0898312232219087
#R2 : 0.8857774410332282

