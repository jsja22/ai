#여자 남자구별 ImageDataGenerator의 fit 사용 fit으로할거면 numpy저장

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import numpy as np

# train_male = 'C:/data/image/male_female/0'
# train_female = 'C:/data/image/male_female/1'
#rootPath = 'C:/data/image/male_female/2'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=0.7,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
    'C:/data/image/sex',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_test = train_datagen.flow_from_directory(
    'C:/data/image/sex',
    target_size=(150,150),
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
print(xy_train)  
print(xy_test)
print(xy_train[0][0].shape) 
print(xy_train[0][1].shape) 

# np.save('C:/data/image/sex/npy/train_x1.npy', arr=xy_train[0][0])
# np.save('C:/data/image/sex/npy/train_y1.npy', arr=xy_train[0][1])
# np.save('C:/data/image/sex/npy/val_x1.npy', arr=xy_test[0][0])
# np.save('C:/data/image/sex/npy/val_y1.npy', arr=xy_test[0][1])

x_train = np.load('C:/data/image/sex/npy/train_x1.npy')
x_val = np.load('C:/data/image/sex/npy/val_x1.npy')

y_train = np.load('C:/data/image/sex/npy/train_y1.npy')
y_val = np.load('C:/data/image/sex/npy/val_y1.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# (14, 128, 128, 3) (14,)
# (14, 128, 128, 3) (14,)

model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor = 'val_loss', patience = 15)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 6, factor = 0.5, verbose = 1)
history = model.fit(x_train,y_train, epochs=300, validation_data=(x_val,y_val),callbacks=[es,lr])
#fit_generator 은 xy_train을 통으로 넣어주면됨
#steps_per_epoch=32 ->전체데이터의 갯수를 배치사이즈로 나눈 숫자를 넣어줘야함

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

import matplotlib.pyplot as plt
epochs = len(acc)
x_axis = range(0,epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, acc, label='train')
ax.plot(x_axis, val_acc, label='val')
ax.legend()
plt.ylabel('acc')
plt.title('acc')
# plt.show()


fig, ax = plt.subplots()
ax.plot(x_axis, loss, label='train')
ax.plot(x_axis, val_loss, label='val')
ax.legend()
plt.ylabel('loss')
plt.title('loss')
plt.show()

