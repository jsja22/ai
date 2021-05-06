# 과제 및 실습  Dense으로
# 전처리, earlystopping등등 배운거 다넣을것 
# 데이터 1~100 / 5개씩 
#   x              y
#1,2,3,4,5         6
#....
#95,96,97,98,99   100

#predict를 만들것
#96,97,98,99,100 ->101
#...
#100,101,102,103,104 ->105
# 예상 predict는 (101,102,103,104,105)

import numpy as np
from numpy import array

a = np.array(range(1,101))

size1 = 6
size2 = 5
def split_x(seq,size1):
    aaa=[]
    for i in range(len(seq)-size1 +1):
        subset = seq[i : (i+size1)]
        aaa.append([item for item in subset])
    print(type(aaa))
    return np.array(aaa)

dataset = split_x(a,size1)
print("================")
print(dataset)

x = dataset[:,:-1]
y= dataset[:,-1]

x_pred = np.array(range(96,105))

def pred_x(seq,size2):
    bbb=[]
    for i in range(len(seq)-size2+1):
        ss= seq[i: (i+size2)]
        bbb.append([item for item in ss])
    return np.array(bbb)
x_pred1 = pred_x(x_pred,size2)
print("================")
print("x_pred:",x_pred1)
print(x_pred1.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,shuffle=True, train_size =0.8,random_state=66)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)
print(x_train.shape) #(76,4)
print(y_train.shape) #(76, )
print(x_test.shape)  #(19, 4)
print(y_test.shape)  #(19, )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred1 = scaler.transform(x_pred1)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Input, LSTM

input1 = Input(shape=(5,))
dense1 = Dense(10, activation='relu')(input1)
dense1 = Dense(40)(dense1)
dense1 = Dense(80)(dense1)
dense1 = Dense(40)(dense1)
dense1 = Dense(20)(dense1)
output1 = Dense(1)(dense1)

model = Model(inputs=input1, outputs=output1)

model.summary() 

model.compile(loss='mse',optimizer='adam',metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=24,mode='auto')
model.fit(x_train,y_train,epochs=2000,batch_size=8, validation_data=(x_val,y_val),verbose=1,callbacks=[early_stopping])

loss,mae = model.evaluate(x_test,y_test,batch_size=8)
print("loss,mae :",loss,mae)
y_predict = model.predict(x_pred1)
print(y_predict)

#Dense
#loss,mae : 1.780786806193646e-05 0.0037640018854290247
#[[100.99296 ]
# [101.992874]
# [102.992805]
# [103.99273 ]
# [104.99265 ]]

#Dense의 loss가 훨씬 낫다. 