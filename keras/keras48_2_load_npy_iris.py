import numpy as np

x_data = np.load('C:/data/npy/iris_x.npy')
y_data = np.load('C:/data/npy/iris_y.npy')

print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)


from sklearn.preprocessing import MinMaxScaler #x값을 minmax전처리 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2,shuffle=True,random_state=66)
x_train,x_val,y_train,y_val = train_test_split(x_train,y_train, train_size=0.8, shuffle=True)

print(x_train)
print(y_train)
print(x_test)
print(y_test)


scalar = MinMaxScaler()
scalar.fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_val = scalar.transform(x_val)

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) # reshape에서 -1은 재배열의 의미이다.
y_val = y_val.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_val = ohencoder.transform(y_val).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(4,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))  #softmax : output node 다합치면 1이되는함수, 3인이유는 y.shape가 (150,3)이기에 3 softmax는 다중분류에서만 사용
                                          #가장 큰 값이 결정된다. 
model.summary()


#컴파일,훈련
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])  #다중분류에서는 loss='categorical_crossentropy'사용 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss',patience=30,mode='auto')
model.fit(x_train,y_train, epochs=110, batch_size=8,validation_data=(x_test,y_test),verbose=0,callbacks=[early_stopping])
loss = model.evaluate(x_test,y_test,batch_size=8)
print(loss) #(loss,acuracy)
y_pred = np.array(model.predict(x_train[-5:-1]))
print(y_pred.argmax(axis=1))

#loss: 0.0535 - accuracy: 0.9667
#[0.053524479269981384, 0.9666666388511658]
