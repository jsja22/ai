import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l1, l2, l1_l2
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0],'GPU')
    except RuntimeError as e:
        print(e)


(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1)/255.


#modelpath = 'C:/data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#cp = ModelCheckpoint(filepath = modelpath, monitor='val_loss',save_best_only = True, mode = 'auto')  #period=5 다섯 번째 에포크마다 가중치를 저장하기 위한 콜백을 만듭니다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(filters = 512,kernel_size = (2,2),padding='same',strides = 1,input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.5))
model.add(Conv2D(32,2,kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(32,2,kernel_regularizer= l1(l1 = 0.01)))

# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(x_train,y_train,validation_split = 0.2,epochs=30,verbose=1,batch_size=16)

result = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict,axis=-1)
print('loss : ',result[0])
print('accuracy : ',result[1])

for i in range(len(y_test)):
    print('actual value : ',np.argmax(y_test[i]),'predict value : ',y_predict[i])
print('accuracy : ',loss[1])





