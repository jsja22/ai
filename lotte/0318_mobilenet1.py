import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, KFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam,SGD
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import MobileNet,MobileNetV2
import datetime
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Flatten, BatchNormalization, Dense, Activation

##train, valid 나누기 ###

TRAIN_PATH = 'C:/data/LPD_competition/train/'
TEST_PATH = 'C:/data/LPD_competition/pred/'
IMAGE_SIZE = (224, 224, 3)
BATCH_SIZES = 16
EPOCHS = 20
SEED = 66

def create_datagen(data_type = "train"):
    if data_type == "train":
            datagen = ImageDataGenerator(
                       
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        fill_mode='nearest',
                        validation_split = 0.2,
                        preprocessing_function= preprocess_input,
        )
    elif data_type == "valid":
            datagen = ImageDataGenerator(
               
                preprocessing_function= preprocess_input,
            )
        
    return datagen
    
def create_dataset(datagen, path, data_type = "train"):
    if data_type == "train":
        dataset = datagen.flow_from_directory(
            path,
            target_size = IMAGE_SIZE,
            batch_size = BATCH_SIZES,
            class_mode = "sparse",
            #shuffle = True,
            seed = SEED,
            subset = 'training'
        )
    elif data_type == "valid":
        dataset = datagen.flow_from_directory(
            path,
            target_size = IMAGE_SIZE,
            batch_size = BATCH_SIZES,
            class_mode = "sparse",
            #shuffle = True,
            seed = SEED,
            subset = 'validation'
        )
    elif data_type =="pred":
        dataset = datagen.flow_from_directory(
            path,
            target_size = IMAGE_SIZE,
            batch_size = BATCH_SIZES,
            class_mode = None,
            shuffle = False,
        )

def create_model():
    md = MobileNet(input_shape = IMAGE_SIZE, weights = "imagenet", include_top = False)
    for layer in md.layers:
        layer.trainable = True
    x = md.output
    x = tf.keras.layers.Dropout(0.3) (x)
    x = tf.keras.layers.Dense(128) (x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2) (x)
    x = tf.keras.layers.GaussianDropout(0.4) (x)
    x = GlobalAvgPool2D(name='global_avg')(x)
    output = Dense(1000, activation='softmax')(x)
    model = Model(inputs=md.input, outputs=output)
    return model

def compile_model(model, lr = 1e-3):

    optimizer = Adam(lr=lr)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer= optimizer,
        metrics=['sparse_categorical_accuracy']
    )

    return model

def create_callbacks():

    cp_path = 'C:/data/LPD_competition/modelcheckpoint/0318_mobilenet.h5'
    cp = ModelCheckpoint(cp_path,monitor = 'val_sparse_categorical_accuracy',mode = 'auto',save_best_only=True, verbose=1)
    es = EarlyStopping(monitor = 'val_sparse_categorical_accuracy',patience=5, restore_best_weights=True, mode='auto', verbose=1)
    lr = ReduceLROnPlateau(monitor = 'val_sparse_categorical_accuracy',factor=0.3,patience = 3, verbose=1,mode ='auto', min_lr = 1e-6)

    callbacks = [cp,es,lr]

    return callbacks

train_datagen = create_datagen("train")
valid_datagen = create_datagen("valid")

train_data = create_dataset(train_datagen,TRAIN_PATH,"train")
valid_data = create_dataset(train_datagen,TRAIN_PATH,"valid")
test_data = create_dataset(valid_datagen, TEST_PATH, "pred")

# classes_dict = train_data.class_indices
# classes_dict = {v: k for k, v in classes_dict.items()}
# print("Prediction to Label Matching: ", classes_dict)

# 모델구성
model = create_model()
model = compile_model(model,lr = 1e-3)
callbacks = create_callbacks()
t1 = time()
history = model.fit(train_data, epochs=EPOCHS,validation_data = valid_data, verbose=1, callbacks = callbacks)
t2 = time()
print("execution time: ", t2 - t1)

'''
# predict
model.load_weights('C:/data/LPD_competition/modelcheckpoint/0318_mobilenet.h5')
score = model.evaluate(valid_data)
print("Test Data acc : ", score[1])

# test_labels = valid_generator.classes
# predictions = model.predict(valid_generator, verbose=1)
# y_pred = np.argmax(predictions, axis=-1)
# print(classification_report(test_labels, y_pred, target_names= valid_generator.class_indices))

result = model.predict(test_data,verbose=True)
print(result.shape) #(72000, 1000)
sub = pd.read_csv('C:/data/LPD_competition/csv/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/submission/answer3.csv',index=False)
'''