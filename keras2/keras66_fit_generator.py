import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
train_datagen = ImageDataGenerator(
    # rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=0.7,
    # fill_mode='nearest'
)

train_datagen = ImageDataGenerator(rescale=1./255) #test는 rescale만

#flow or flow_from_directory  -> 데이터로 변환

#이미 수치화 되어있는걸 가져오려면 그냥 flow
#train_generator
#train에 ad,normal 이있기때문에 y는 0과 1
xy_train = train_datagen.flow_from_directory(
    'C:/data/image/brain/data1/train',
    target_size=(150,150), #주고싶은 사이즈로 줄이거나 늘임
    batch_size=5,
    class_mode='binary'
)
#(80,150,150,1)
#test_generator
xy_test = train_datagen.flow_from_directory(
    'C:/data/image/brain/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
)

model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
# model.add(Conv2D(32,(5,5),padding='same',activation='relu'))
# model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
optimizer = Adam(lr=0.002)
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['acc'])

history = model.fit_generator(xy_train, steps_per_epoch=32, epochs=30, validation_data=xy_test, validation_steps=4)
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
