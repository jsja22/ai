
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import numpy as np
from PIL import Image

x_train = np.load('C:/data/image/npy/keras67_train_x.npy')
y_train = np.load('C:/data/image/npy/keras67_train_y.npy')
x_test = np.load('C:/data/image/npy/keras67_test_x.npy')
y_test = np.load('C:/data/image/npy/keras67_test_y.npy')

model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(56,56,3)))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])

# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# es = EarlyStopping(monitor='val_acc', patience=30, mode='auto')
# lr = ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=20, mode='max')
# filepath = ('C:/data/image/modelcheckpoint/k67_-{val_acc:.4f}.hdf5')
# cp = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

# model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_split=0.2, callbacks=[es,lr,cp])
model = load_model('C:/data/image/modelcheckpoint/k67_-0.6750.hdf5')

loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print("loss : ", loss)
print("acc : ", acc)


# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     # horizontal_flip=True,
#     # vertical_flip=True,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     # rotation_range=5,
#     # zoom_range=0.7,
#     fill_mode='nearest',
#     validation_split=0.2
# )
# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
#     'C:/data/image/sex',
#     target_size=(150,150),
#     batch_size=14,
#     class_mode='binary',
#     subset='training'
# )
# xy_test = train_datagen.flow_from_directory(
#     'C:/data/image/sex',  #마지막 경로가 반드시 폴더여야함
#     target_size=(150,150), #size
#     batch_size=14,
#     class_mode='binary',
#     subset='validation'
# )
# x_pred = train_datagen.flow_from_directory(
#     'C:/data/image/my/',
#     target_size = (150,150),
#     batch_size= 14,
#     class_mode='binary', 
# )
# print(xy_train)  
# print(xy_test)
# print(xy_train[0][0].shape) 
# print(xy_train[0][1].shape) 

# # Found 1389 images belonging to 1 classes.
# # Found 347 images belonging to 1 classes.
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000224903B8550>
# # <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022490439160>
# # (14, 64, 64, 3)
# # (14,)

# model = Sequential()
# model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(150,150,3)))
# model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.3))

# model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.3))

# model.add(Flatten())
# model.add(Dense(64,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# model.summary()

# model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
# #print(xy_train.samples) #1389
# # cp = ModelCheckpoint('C:/data/modelcheckpoint/my2.hdf5') 
# # es = EarlyStopping(monitor = 'val_acc', patience = 15)
# # lr = ReduceLROnPlateau(monitor = 'val_acc', patience = 6, factor = 0.5, verbose = 1)
# # history = model.fit_generator(xy_train, steps_per_epoch=1389/14 ,epochs=100, validation_data=xy_test, validation_steps=1389/14,callbacks=[es,lr,cp])

# model2 = load_model('C:/data/modelcheckpoint/my2.hdf5', compile=False)
# model.save_weights('C:/data/h5/k67_4_weight.h5')
# model.load_weights('C:/data/h5/k67_4_weight.h5')
import PIL.Image as pilimg
image4 = pilimg.open('C:/data/image/my2/2.jpg')
pix4 = image4.resize((56,56))
pix4 = np.array(pix4)
madong_test = pix4.reshape(1, 56, 56, 3)/255.

# 예측하기 _4
my_pred4 = model.predict(madong_test)
print(my_pred4) 
print('마동석은(두구두구)')
print((my_pred4[0][0])*100,'%의 확률로 남자입니다.')
