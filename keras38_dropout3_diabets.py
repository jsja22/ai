import numpy as np
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x[:5])
print(y[:10])
print(x.shape, y.shape)  # (442, 10)  (442,)
print(np.max(x), np.min(x))
print(dataset.feature_names)
print(dataset.DESCR)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.6, shuffle=True)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)



#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
inputs = Input(shape=(10,))
dense1 = Dense(128, activation='linear')(inputs)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(128)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(64)(dense1)
dense1 = Dropout(0.2)(dense1)
dense1 = Dense(8)(dense1)
outputs = Dense(1)(dense1)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
hist = model.fit(x_train, y_train, batch_size=8, epochs=1000, validation_data=(x_val, y_val), callbacks=[early_stopping])

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['mae'])
plt.plot(hist.history['val_mae'])

plt.title('loss & mae')
plt.ylabel('loss, mae')
plt.xlabel('epochs')
plt.legend(['train loss', 'val loss', 'train mae', 'val mae'])
plt.show()

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size=8)
y_predict = model.predict(x_test)
print("loss, mae : ", loss, mae)

from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE : ", RMSE(y_test, y_predict))
#print("mse : ", mean_squared_error(y_test, y_predict))


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 : ", r2)

#to_categorical에서는 0이없는부분 0이 채워진다
#[2,3,5]
#onehotencoder로는 3x3
#to_categorical로는 3x5

#드랍아웃전
#loss, mae :  2784.8974609375 43.5482292175293
#RMSE :  52.772128895163796
#R2 :  0.5080720506630692

# 드랍아웃후
#loss, mae :  2801.457763671875 42.48012924194336
#RMSE :  52.92880012218722
#R2 :  0.5857280938394237
#오히려 로스는 증가하였지만 r2는 좋아짐.