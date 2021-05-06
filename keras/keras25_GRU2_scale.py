import numpy as np
import tensorflow as tf

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12], [20,30,40], [30,40,50], [40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70])
x = x.reshape(13, 3, 1)
print(x.shape) #(13,3)
print(y.shape) #(13,)
print(x_pred.shape) #(3,)

x_pred=x_pred.reshape(1, 3, 1)

#코딩하시오 LSTM
#나는 80을 원하고 있다.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# print(x_train)
# print(x_test)
# print(y_train)
# print(y_test)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU

model = Sequential()
model.add(GRU(100, input_shape=(3,1), activation='relu'))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics='acc')
model.fit(x_train, y_train, epochs=200, validation_split=0.2, batch_size=1)

loss = model.evaluate(x_test)
y_pred = model.predict(x_pred)


print('loss : ', loss)
print('y_pred', y_pred)

# loss :  [0.0, 0.0]
# y_pred  [[79.06978]]