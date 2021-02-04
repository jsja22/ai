import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('C:/data/computer vision/mnist_data/train.csv')
test = pd.read_csv('C:/data/computer vision/mnist_data/test.csv')
sub = pd.read_csv('C:/data/computer vision/mnist_data/submission.csv',header=0)
##opencv설치##
import cv2
print(cv2.__version__)

#########train, test, 각각 0~9 숫자별로 분류하기 #####
# for idx in range(len(train)) :




#     img = train.loc[idx, '0':].values.reshape(28, 28).astype(int)
#     digit = train.loc[idx, 'digit']
#     cv2.imwrite(f'C:/data/computer vision/images_train/{digit}/{train["id"][idx]}.png', img)

# for idx in range(len(test)) :
#     img = test.loc[idx, '0':].values.reshape(28, 28).astype(int)
#     cv2.imwrite(f'C:/data/computer vision/images_test/{test["id"][idx]}.png', img)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(validation_split=0.2,
                             width_shift_range=0.1,
                             height_shift_range=0.1)
                             
train_generator = datagen.flow_from_directory('C:/data/computer vision/images_train', target_size=(224,224), color_mode='grayscale', class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory('C:/data/computer vision/images_test', target_size=(224,224), color_mode='grayscale', class_mode='categorical', subset='validation')


from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation,Flatten, GlobalAveragePooling2D, BatchNormalization
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
lr = ReduceLROnPlateau(monitor='val_acc',patience=40,verbose=1,factor=0.5)
es = EarlyStopping(monitor='val_acc',patience=80, verbose=1,mode='auto')

cp = ModelCheckpoint('C:/data/computer vision/h5/Dacon_computer_vision_0204_3.h5', monitor='val_acc',save_best_only=True, verbose=1)
model = Sequential()

model.add(InceptionResNetV2(weights=None,include_top=False, input_shape=(224, 224, 1), classes=10))
model.add(GlobalAveragePooling2D())
model.add(Dense(512, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(512, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64, kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax',kernel_initializer='he_normal' ))

model.summary()
    
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit_generator(train_generator, epochs=300, validation_data=val_generator, callbacks=[cp,es,lr])


import matplotlib.pyplot as plt

plt.figure(figsize=(10,6)) 

plt.plot(model.history.history["acc"], label='m1_acc')
#plt.plot(model.history.history["val_acc"], label='m1_vacc')
plt.grid()
plt.legend(loc='upper right')
plt.show()
