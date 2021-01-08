import numpy as np
from sklearn.datasets import load_wine

dataset = load_wine()
# print(dataset.DESCR) 
# print(dataset.feature_names)

x = dataset.data
y = dataset.target
# print(x) 
# print(y) # 다중분류
# print(x.shape, y.shape) #(178, 13) (178, )
x_pred = x[-5:-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
x_pred = scaler.transform(x_pred)

x_train = x_train.reshape(-1, 13, 1)
x_val = x_val.reshape(-1, 13, 1)
x_test = x_test.reshape(-1, 13, 1)
x_pred = x_pred.reshape(-1, 13, 1)

from tensorflow.keras.utils import to_categorical
# from keras.utils.up_utils import to_categorical
y = to_categorical(y)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, input_shape=(13,1)))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) # 다중분류일때 loss는 categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=20, mode='auto')
hist = model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks=[early_stopping], batch_size=8)

print(hist)
print(hist.history.keys()) # loss, acc, val_loss, val_acc

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=16)
print(loss, acc)

y_pred = model.predict(x_pred)
print(y_pred)
print(y[-5:-1])
y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
print(y_recovery)


# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])

plt.title('loss & acc')
plt.ylabel('loss, acc')
plt.xlabel('epochs')
plt.legend(['train loss', 'val loss', 'train acc', 'val acc'])
plt.show()