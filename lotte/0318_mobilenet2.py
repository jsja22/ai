import numpy as np
import pandas as pd
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

SEED = 66
IMAGE_SIZE = (224,224,3)
EPOCH = 20
OPTIMIZER =Adam(learning_rate= 1e-3)

#데이터 지정 및 전처리
x = np.load("C:/data/LPD_competition/npy/P_project_x6.npy",allow_pickle=True)
y = np.load("C:/data/LPD_competition/npy/P_project_y6.npy",allow_pickle=True)
x_pred = np.load('C:/data/LPD_competition/npy/test_224.npy',allow_pickle=True)

#print(x.shape) # (48000, 150, 150, 3)
x = preprocess_input(x) 
x_pred = preprocess_input(x_pred)   


idg = ImageDataGenerator(
    width_shift_range=(-1,1),   
    height_shift_range=(-1,1),  
    shear_range=0.2) 
   

idg2 = ImageDataGenerator()

y = np.argmax(y, axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x,y, train_size = 0.8, shuffle = True, random_state=SEED)

train_generator = idg.flow(x_train,y_train,batch_size=32)
valid_generator = idg2.flow(x_valid,y_valid)
test_generator = x_pred

md = MobileNet(input_shape = IMAGE_SIZE, weights = "imagenet", include_top = False)
for layer in md.layers:
    layer.trainable = True
x = GlobalAvgPool2D(name='global_avg')(md.output)
prediction = Dense(1000, activation='softmax')(x)
model = Model(inputs=md.input, outputs=prediction)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
cp = ModelCheckpoint('C:/data/LPD_competition/modelcheckpoint/lotte_projcet3.h5',save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy',patience= 5)
lr = ReduceLROnPlateau(monitor='val_loss',patience= 3, factor=0.3)

model.compile(loss='sparse_categorical_crossentropy', optimizer=OPTIMIZER,
                metrics=['acc'])

t1 = time()       
history = model.fit_generator(train_generator,
    validation_data=valid_generator, epochs=EPOCH, steps_per_epoch=len(train_generator), 
                    validation_steps=len(valid_generator),callbacks=[early_stopping,lr,cp])


t2 = time()
print("execution time: ", t2 - t1)
# predict
model.load_weights('C:/data/LPD_competition/modelcheckpoint/lotte_projcet3.h5')
score = model.evaluate(valid_generator)
print("Test Data acc : ", score[1])

# test_labels = valid_generator.classes
# predictions = model.predict(valid_generator, verbose=1)
# y_pred = np.argmax(predictions, axis=-1)
# print(classification_report(test_labels, y_pred, target_names= valid_generator.class_indices))

result = model.predict(test_generator,verbose=True)
    
print(result.shape) #(72000, 1000)
sub = pd.read_csv('C:/data/LPD_competition/csv/sample.csv')
sub['prediction'] = np.argmax(result,axis = 1)
sub.to_csv('C:/data/LPD_competition/submission/answer3.csv',index=False)

#Efficientnet-B7(epochs 85) SOCRE = 56
#Mobilenet(epoch 40)  SCORE = 65 (sparse_categorical_crossentropy)
#Test Data acc :  0.997083306312561  

#ImageSize 2240 ->epoch20 SCORE = 70
#execution time:  4915.963658809662
#Test Data acc :  0.9989583492279053