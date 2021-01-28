import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
from sklearn.svm import LinearSVC
dataset = load_iris()
x,y = load_iris(return_X_y=True)

# from sklearn.preprocessing import MinMaxScaler #x값을 minmax전처리 
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
# #x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

# scalar = MinMaxScaler()
# scalar.fit(x_train)
# x_train = scalar.transform(x_train)
# x_test = scalar.transform(x_test)
# #x_val = scalar.transform(x_val)
'''
#원핫 인코딩  (다중분류에 인해서 y를 원핫인코딩한것이다.=>y를 벡터화 하는것) 값이 있는부분에만 1을 넣고 나머지엔 0 
#다중분류에서 y는 반드시 원핫인코딩 해줘야함
from tensorflow.keras.utils import to_categorical

y=to_categorical(y)
y_train= to_categorical(y_train)
y_test = to_categorical(y_test)

print(y)
print(x.shape) #(150, 4)
print(y.shape) #(150, 3)
'''
'''
print(dataset.DESCR)
print(dataset.feature_names) 

print(x.shape)  #(150,4)
print(y.shape)  #(150, )
print(x[:5])
print(y)
'''

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


model = LinearSVC()
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4,)))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))  #softmax : output node 다합치면 1이되는함수, 3인이유는 y.shape가 (150,3)이기에 3 softmax는 다중분류에서만 사용
#                                           #가장 큰 값이 결정된다. 
# model.summary()
 
#컴파일,훈련
# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #다중분류에서는 loss='categorical_crossentropy'사용 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=30,mode='auto')
# model.fit(x_train,y_train, epochs=110, batch_size=8,validation_data=(x_test,y_test),verbose=0,callbacks=[early_stopping])
model.fit(x,y)
# loss = model.evaluate(x_test,y_test,batch_size=8)
result = model.score(x,y)
print(result) #(loss,acuracy)

#실습1. acc 0.985 이상 
#실습2. predict 출력해볼것

y_pred = model.predict(x[-5:-1])  
print(y_pred)
print(y[-5:-1]) 

#[0.0403638556599617, 1.0, 0.03442172706127167] acc =1 굿!

#argmax 최대값 찾기
print(np.argmax(y_pred,axis=-1))

# y_pred = np.array(model.predict(x_train[-5:-1]))
# print(y_pred.argmax(axis=1))


#0.9666666666666667 -> accuracy   machine learning 
# evaluate -> score
#compile ->x
#model -> model = LinearSVC()
#onehotencoding ->x