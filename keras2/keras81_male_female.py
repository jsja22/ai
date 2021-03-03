#vgg 16 
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from   keras.datasets  import cifar10
from   keras.layers    import Conv2D, Dropout, Input, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from   keras.models    import Sequential, Model
from   keras.utils     import np_utils
from   keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from tensorflow.keras.applications import VGG16,VGG19,Xception
from tensorflow.keras.applications import ResNet101,ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

x_train = np.load('C:/data/image/sex/npy/keras81_train_x.npy')
y_train = np.load('C:/data/image/sex/npy/keras81_train_y.npy')
x_test = np.load('C:/data/image/sex/npy/keras81_test_x.npy')
y_test = np.load('C:/data/image/sex/npy/keras81_test_y.npy')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# (200, 64, 64, 3)
# (200,)
# (200, 64, 64, 3)
# (200,)

from tensorflow.keras.applications.vgg16 import preprocess_input

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)

vgg16 = VGG16(weights='imagenet',input_shape =(64,64,3),include_top=False)
vgg16.trainalbe = False
model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(1, activation= 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(1e-5), metrics=['acc'])
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
model_path = 'C:/data/modelcheckpoint/male_female.hdf5'
checkpoint = ModelCheckpoint(filepath=model_path , monitor='val_loss', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)
lr = ReduceLROnPlateau(patience=25, factor=0.5,verbose=1)

history = model.fit(x_train, y_train, batch_size=16, epochs=100, validation_data=(x_test, y_test),callbacks=[early_stopping,
checkpoint,lr])

print("정확도 : %.4f" % (model.evaluate(x_test, y_test)[1]))
# result = model.predict_generator(x_pred,verbose=True)
# result[result < 0.5] =0
# result[result > 0.5] =1
# np.where(result < 0.5, '남자', '여자')
# print("남자일 확률은",result*100,"%입니다.")
