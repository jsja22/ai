#keras67_1 man female noise 
#기미 주근깨 여드름을 제거하싵오

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import numpy as np

def autoencoder():
    model = Sequential()
    model.add(Conv2D(256, 3, activation= 'relu', padding= 'same', input_shape = (256,256,3)))
    model.add(Conv2D(256, 5, activation= 'relu', padding= 'same'))
    model.add(Conv2D(3, 3, padding = 'same', activation= 'sigmoid'))

    return model

x_train = np.load('C:/data/image/sex/npy/keras81_train_x.npy')
#y_train = np.load('C:/data/image/sex/npy/keras81_train_y.npy')
x_test = np.load('C:/data/image/sex/npy/keras81_test_x.npy')
#y_test = np.load('C:/data/image/sex/npy/keras81_test_y.npy')
print(x_train.shape)
#print(y_train.shape)
print(x_test.shape)
#print(y_test.shape)

x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)
x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

model = autoencoder()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
model.fit(x_train_noised,x_train,epochs=30, verbose=1,steps_per_epoch=93,validation_steps=31,validation_split=0.2)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10), (ax11, ax12, ax13, ax14, ax15)) = plt.subplots(3, 5, figsize=(17, 7))

# 이미지 다섯 개를 무작위로 고른다
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel('INPUT', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 노이즈 있는 친구
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel('NOISE', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

# 오토인코더가 출력한 이미지를 아래에 그린다
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(output[random_images[i]], cmap='gray')
    if i == 0:
        ax.set_ylabel('OUTPUT', size = 25)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()