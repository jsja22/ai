##CNN

import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1)/255.

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input,Conv2D, MaxPooling2D,Flatten,UpSampling2D,Conv2DTranspose

def CNNmodel(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size*10, kernel_size=(2,2), padding='same',input_shape=(28,28,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=hidden_layer_size*6, kernel_size=(2,2), padding='same',activation='relu'))
    model.add(UpSampling2D(size=(2,2))) #, 적은 해상도를 일부러 고해상도로 올리는것
    #Conv2DTranspose는 Convolution 연산이 들어가서 해상도를 키운다. 이 연산은 당연히 학습과정에서 필터가 학습이 된다.
    model.add(Conv2D(filters=hidden_layer_size*2, kernel_size=(4,4), padding='same',activation='relu'))
    model.add(Conv2D(filters=hidden_layer_size*2, kernel_size=(4,4), padding='same',activation='relu'))
    model.add(Conv2D(filters=1, kernel_size=(4,4), padding='same',activation='sigmoid'))

    return model

def CNNmodel2(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(filters=hidden_layer_size*10, kernel_size=(2,2), padding='same',input_shape=(28,28,1),activation='relu'))
    model.add(Conv2DTranspose(filters=1, kernel_size=(4,4), padding='same',activation='sigmoid'))
    return model
model = CNNmodel2(hidden_layer_size=32)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train, x_train, epochs=10)

output = model.predict(x_test)

import matplotlib.pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5),(ax6,ax7,ax8,ax9,ax10)) = \
    plt.subplots(2,5,figsize=(20,7))

random_images = random.sample(range(output.shape[0]),5)
#원본 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1,ax2,ax3,ax4,ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

#오토 인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax6,ax7,ax8,ax9,ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28),cmap='gray')
    if i==0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.show()

#1. loss: 0.0584 - acc: 0.8155