import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))  #mse와 같다. 원래값-예측값을 제곱한거를 평균으로 나눈값 

#1. 데이터data
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')


print(x.shape)

#2. 모델

model = Sequential()
model.add(Dense(10,input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss=custom_mean_squared_error, optimizer='adam')

model.fit(x,y,batch_size=1, epochs=30)
# 첫번쨰는 실제값 두번쨰는 fit해서 나온 프레딕트값을 자동으로 인식한다. 
loss = model.evaluate(x,y)

print(loss)