import numpy as np
from numpy import asarray
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from glob import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications import EfficientNetB7, MobileNet
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from PIL import Image

###

train_datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split = 0.2,
    # preprocessing_function= preprocess_input,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(
    # preprocessing_function= preprocess_input,
)

xy_train = train_datagen.flow_from_directory(
    'C:/data/LPD_competition/train',
    target_size=(224,224),
    batch_size=48000,
    seed=66,
    subset='training',
    class_mode='sparse'
)   # Found 39000 images belonging to 1000 classes.

xy_val = train_datagen.flow_from_directory(
    'C:/data/LPD_competition/train',
    target_size=(224,224),
    batch_size=48000,
    seed=66,
    subset='validation',
    class_mode='sparse'
)   # Found 9000 images belonging to 1000 classes.  

xy_pred = test_datagen.flow_from_directory(
    'C:/data/LPD_competition/pred',
    target_size=(224,224),
    batch_size=72000,
    class_mode=None, 
    shuffle=False
)   # Found 72000 images belonging to 1 classes.

# train
np.save('C:/data/LPD_competition/npy/data_x_train2.npy', arr=xy_train[0][0], allow_pickle=True)
np.save('C:/data/LPD_competition/npy/data_y_train2.npy', arr=xy_train[0][1], allow_pickle=True)

x_train = np.load('C:/data/LPD_competition/npy/data_x_train2.npy', allow_pickle=True)
y_train = np.load('C:/data/LPD_competition/npy/data_y_train2.npy', allow_pickle=True)
print(x_train.shape) 
print(y_train.shape) 

np.save('C:/data/LPD_competition/npy/data_x_val2.npy', arr=xy_val[0][0], allow_pickle=True)
np.save('C:/data/LPD_competition/npy/data_y_val2.npy', arr=xy_val[0][1], allow_pickle=True)

x_val = np.load('C:/data/LPD_competition/npy/data_x_val2.npy', allow_pickle=True)
y_val = np.load('C:/data/LPD_competition/npy/data_y_val2.npy', allow_pickle=True)
print(x_val.shape)  # (9000, 100, 100, 3)
print(y_val.shape)  # (9000, )

np.save('C:/data/LPD_competition/npy/data_x_pred2.npy', arr=xy_pred[0], allow_pickle=True)

x_pred = np.load('C:/data/LPD_competition/npy/data_x_pred2.npy', allow_pickle=True)
print(x_pred.shape) # (72000, 100, 100, 3)