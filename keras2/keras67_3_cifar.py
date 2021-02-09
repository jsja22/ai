#cifar10를 flow로 구성해서 완성 ImageDataGenerator fit_generator

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout,MaxPooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train/255.
x_test = x_test/255.

print(x_train.shape) #(50000, 32, 32, 3)
print(y_train.shape) #(50000, 1)

data_generator = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.7,
    shear_range=0.7,
    fill_mode='nearest'
)

data_generator2 = ImageDataGenerator()
batch_size=32
seed=2048
train_generator = data_generator.flow(x_train,y_train,batch_size,seed)
test_generator = data_generator2.flow(x_test,y_test)


model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(32,32,3)))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(128,(5,5),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128,(5,5),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['acc']
)
#categorical_crossentropy ; 다중 분류 손실함수. one-hot encoding 클래스
#+ sparse_categorical_crossentropy ; 다중 분류 손실함수. 위와 동일하지만 , integer type 클래스라는 것이 다르다.
#->one-hot encoding 과정을 하지 않아도 된다. 


history = model.fit_generator(train_generator,validation_data=(x_test,y_test),
                              steps_per_epoch= 50000/batch_size, epochs=100, validation_steps=50000/batch_size)

loss, acc = model.evaluate(test_generator)
print("loss : ", loss)
print("acc : ", acc)    

