
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt

train_datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=(-0.1,1),
    height_shift_range=(-0.1,1),
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)


xy_train = train_datagen.flow_from_directory(
    'C:/data/image/sex/' 
    ,target_size=(256,256)
    ,batch_size=200
    ,class_mode='binary'
    ,subset='training'      
)     

xy_test = train_datagen.flow_from_directory(
    'C:/data/image/sex/' 
    ,target_size=(256,256)
    ,batch_size=200
    ,class_mode='binary'
    ,subset='validation'
)

np.save('C:/data/image/sex/npy/keras81_train_x.npy', arr = xy_train[0][0])
np.save('C:/data/image/sex/npy/keras81_train_y.npy', arr = xy_train[0][1])
np.save('C:/data/image/sex/npy/keras81_test_x.npy', arr = xy_test[0][0])
np.save('C:/data/image/sex/npy/keras81_test_y.npy', arr = xy_test[0][1])
