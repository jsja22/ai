
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import os
import numpy as np
from PIL import Image

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
    'C:/data/image/sex',  #마지막 경로가 반드시 폴더여야함
    target_size=(150,150), #size
    batch_size=14,
    class_mode='binary',
    subset='validation'
)
x_pred = train_datagen.flow_from_directory(
    'C:/data/image/my/',
    target_size = (150,150),
    batch_size= 14,
    class_mode='binary', 
)
print(xy_train)  
print(xy_test)
print(xy_train[0][0].shape) 
print(xy_train[0][1].shape) 

# Found 1389 images belonging to 1 classes.
# Found 347 images belonging to 1 classes.
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000224903B8550>
# <tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x0000022490439160>
# (14, 64, 64, 3)
# (14,)

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
#print(xy_train.samples) #1389
cp = ModelCheckpoint('C:/data/modelcheckpoint/my.hdf5') 
es = EarlyStopping(monitor = 'val_loss', patience = 15)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 6, factor = 0.5, verbose = 1)
history = model.fit_generator(xy_train, steps_per_epoch=1389/14 ,epochs=100, validation_data=xy_test, validation_steps=1389/14,callbacks=[es,lr,cp])

model2 = load_model('C:/data/modelcheckpoint/my.hdf5', compile=False)
# model.save_weights('C:/data/h5/k67_4_weight.h5')
# model.load_weights('C:/data/h5/k67_4_weight.h5')

result = model.predict_generator(x_pred,verbose=True)

print("남자일 확률은",result*100,"%입니다.")