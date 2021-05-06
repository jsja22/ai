#1. 데이터
import numpy as np
import tensorflow as tf

x = np.array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]]) 
y = np.array([4,5,6,7])                                    
x = x.reshape(4, 3, 1)

#2. 모델 구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN             

model = Sequential()

model.add(SimpleRNN(10, activation='relu', input_shape=(3,1)))   
model.add(Dense(20))                                        
model.add(Dense(10))                                     
model.add(Dense(1))                                         

model.summary()
