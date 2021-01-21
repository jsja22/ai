import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import tensorflow.keras.backend as K

def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))  #mse와 같다. 원래값-예측값을 제곱한거를 평균으로 나눈값 

def quantile_loss(y_true, y_pred):
    qs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    q = tf.constant(np.array([qs]), dtype=tf.float32)  #텐서플로우 상수형태로 바꾸겠다는 뜻
    e = y_true - y_pred
    v = tf.maximum(q*e,(q-1)*e)
    return K.mean(v)


#1. 데이터
x = np.array([1,2,3,4,5,6,7,8]).astype('float32')
y = np.array([1,2,3,4,5,6,7,8]).astype('float32')


print(x.shape)

#2. 모델

model = Sequential()
model.add(Dense(10,input_shape=(1,)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss=quantile_loss, optimizer='adam')

model.fit(x,y,batch_size=1, epochs=30)
# 첫번쨰는 실제값 두번쨰는 fit해서 나온 프레딕트값을 자동으로 인식한다. 
loss = model.evaluate(x,y)

print(loss)

#custom_mean_squared_error
#loss : 0.027947351336479187

#quantile_loss
#loss : 0.0060495175421237946

#결과치는 quantile loss 가 압승