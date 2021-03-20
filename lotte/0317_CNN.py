import os        
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import tensorflow as tf
import PIL
import PIL.Image
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation

    
def CNN_model():
    model = Sequential()
    model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(254,254,3)))
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

    return model

TRAIN_PATH = 'C:/data/LPD_competition/train'
epochs = 20

#training data
train_datagen = ImageDataGenerator(rescale =1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(directory = TRAIN_PATH,
                                                    target_size=(254,254),
                                                    batch_size= 12,
                                                    class_mode = 'categorical',
                                                    subset = 'training')

#testing data
valid_generator = train_datagen.flow_from_directory(directory = TRAIN_PATH,
                                                    target_size=(254,254),
                                                    batch_size= 12,
                                                    class_mode = 'categorical',
                                                    subset = 'validation')

#model
efficientnetb7 = EfficientNetB7(include_top=False,weights='imagenet',input_shape=train_generator.shape[1:])
efficientnetb7.trainable = True
a = efficientnetb7.output
a = GlobalAveragePooling2D() (a)
a = Flatten() (a)
a = Dense(128) (a)
a = BatchNormalization() (a)
a = Activation('relu') (a)
a = Dense(64) (a)
a = BatchNormalization() (a)
a = Activation('relu') (a)
a = Dense(1000, activation= 'softmax') (a)

model = Model(inputs = efficientnetb7.input, outputs = a)

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam', metrics=['acc'])
# es = EarlyStopping(monitor = 'val_loss', patience = 20)
# lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 5, factor = 0.5, verbose = 1)

#model fit
history = model.fit_generator(
      train_generator,
      steps_per_epoch=254,
      epochs=epochs,
      validation_data = valid_generator,
      validation_steps = 64,
      verbose=1)

acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

#This code is used to plot the training and validation accuracy
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# returns accuracy of training
print("Training Accuracy:"), print(history.history['acc'][-1])
print("Testing Accuracy:"), print (history.history['val_acc'][-1])