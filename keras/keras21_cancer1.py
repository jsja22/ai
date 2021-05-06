import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()

print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

'''
print(x.shape)    #(569,30)
print(y.shape)    #(569, )
print(x[:5])
print(y)
'''
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(30,)))
model.add(Dense(24, activation='sigmoid'))
model.add(Dense(48, activation='sigmoid'))
model.add(Dense(24, activation='sigmoid'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))   #이진분류일때 마지막 activation은 무조건 sigmoid
model.summary()
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy','mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=30,mode='auto')
model.fit(x_train,y_train, epochs=110, batch_size=8,validation_data=(x_val,y_val),verbose=0,callbacks=[early_stopping])
loss = model.evaluate(x_test,y_test,batch_size=8)
print(loss) #(loss,acuracy)

#실습1. acc 0.985 이상 
#실습2. predict 출력해볼것

y_pred = model.predict(x[0:5])
print(y_pred)
print(y[-5:-1]) 