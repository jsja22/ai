#1. 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Flatten, Activation, GlobalAvgPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
import itertools
import shutil
import os
import random
import matplotlib.pyplot as plt
import glob
import datetime
from time import time

#GPU 세팅
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)


##train, valid 나누기 ###

TRAIN_PATH = 'C:/data/LPD_competition/train/'
VALID_PATH = 'C:/data/LPD_competition/test/'
TEST_PATH = 'C:/data/LPD_competition/pred/'
IMAGE_SIZE = (224, 224, 3)
BATCH_SIZES = 64
EPOCHS = 80

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    fill_mode='nearest',
    validation_split=0.2505
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_batches = train_datagen.flow_from_directory(directory=TRAIN_PATH, color_mode="rgb",target_size=IMAGE_SIZE[:-1], class_mode="categorical", shuffle=True,batch_size=BATCH_SIZES)
valid_batches = train_datagen.flow_from_directory(directory=VALID_PATH, color_mode="rgb",target_size=IMAGE_SIZE[:-1], class_mode="categorical", shuffle=False,batch_size=BATCH_SIZES)
test_batches = test_datagen.flow_from_directory(directory=TEST_PATH,target_size=IMAGE_SIZE[:-1],color_mode="rgb",batch_size=BATCH_SIZES,class_mode=None,shuffle=False)

md = keras.applications.MobileNet(input_shape = IMAGE_SIZE, weights = "imagenet", include_top = False)
x = GlobalAvgPool2D(name='global_avg')(md.output)
prediction = Dense(6, activation='softmax')(x)
model = Model(inputs=md.input, outputs=prediction)
model.summary()

optimizer = Adam(learning_rate = 1e-3)
model.compile(
    loss='categorical_crossentropy',
    optimizer= optimizer,
    metrics=['accuracy']
)
stop_callback = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience = 4, verbose=1, min_lr = 1e-6)
t1 = time()
history = model.fit(train_batches, 
                    validation_data = valid_batches, 
                    epochs= EPOCHS, 
                    steps_per_epoch=len(train_batches), 
                    validation_steps=len(valid_batches), 
                    callbacks = [stop_callback,reduce_lr], 
                    shuffle = True)
t2 = time()
print("execution time: ", t2 - t1)

scoreSeg = model.evaluate(valid_batches)
print("Test Data Accuracy = ",scoreSeg[1])

from sklearn.metrics import classification_report
test_labels = valid_batches.classes 
predictions = model.predict(valid_batches, verbose=1)
y_pred = np.argmax(predictions, axis=-1)
print(classification_report(test_labels, y_pred, target_names = valid_batches.class_indices))