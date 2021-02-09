
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import numpy as np

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

xy_train_male = train_datagen.flow_from_directory(
    'C:/data/image/male_female/male',
    target_size=(64,64),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_train_female = train_datagen.flow_from_directory(
    'C:/data/image/male_female/female',
    target_size=(64,64),
    batch_size=14,
    class_mode='binary',
    subset='training'
)
xy_val_male = train_datagen.flow_from_directory(
    'C:/data/image/male_female/male',
    target_size=(64,64),
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
xy_val_female = train_datagen.flow_from_directory(
    'C:/data/image/male_female/female',
    target_size=(64,64),
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
# Found 673 images belonging to 1 classes.
# Found 716 images belonging to 1 classes.
# Found 168 images belonging to 1 classes.
# Found 179 images belonging to 1 classes.
print(xy_train_male)  
print(xy_train_female)
print("++++++++++")
print(xy_train_male[0][0].shape)  #(14, 64, 64, 3)
print(xy_train_male[0][1].shape)  #(14,)
print(xy_val_male[0][0].shape)  #(14, 64, 64, 3)
print(xy_val_male[0][1].shape)  #(14,)
print(xy_train_female[0][0].shape) #(14, 64, 64, 3)
print(xy_train_female[0][1].shape)  #(14,)

np.save('C:/data/image/male_female/npy/train_x.npy', arr=xy_train_male[0][0])
np.save('C:/data/image/male_female/npy/train_y.npy', arr=xy_train_female[0][1])
np.save('C:/data/image/male_female/npy/val_x.npy', arr=xy_val_male[0][0])
np.save('C:/data/image/male_female/npy/val_y.npy', arr=xy_val_female[0][1])

x_train = np.load('C:/data/image/male_female/npy/train_x.npy')
x_val = np.load('C:/data/image/male_female/npy/val_x.npy')

y_train = np.load('C:/data/image/male_female/npy/train_y.npy')
y_val = np.load('C:/data/image/male_female/npy/val_y.npy')

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)
# # (14, 64, 64, 3) (14,)
# # (14, 64, 64, 3) (14,)

model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

#optimizer = Adam(lr=0.002)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)
history = model.fit(x_train,y_train,epochs=100, verbose=1,steps_per_epoch=93,validation_steps=31,validation_split=0.2)
#fit_generator 은 xy_train을 통으로 넣어주면됨
#steps_per_epoch=32 ->전체데이터의 갯수를 배치사이즈로 나눈 숫자를 넣어줘야함


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)   # 2행 1열중 첫번째
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) #2행 1열중 두번째
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='.', c='blue', label='val_acc')
plt.grid()

plt.title('Acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

print("acc :",acc[-1])
print("val_acc :",val_acc[:-1])
