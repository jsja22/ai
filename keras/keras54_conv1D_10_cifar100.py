
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar100


(x_train, y_train), (x_test,y_test)= cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32,3) (50000,1)
print(x_test.shape, y_test.shape) # (10000, 32,32,3) (10000,1)

print(x_train[0])
print(y_train[0])


print(x_train[0].shape) # (32,32,3)

x_train = x_train.reshape(-1,32,32,3).astype('float32')/255
x_test= x_test.reshape(-1,32,32,3).astype('float32')/255

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv1D(filters = 128,kernel_size = 3, strides=1 ,padding='same', input_shape=(32,32,3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(64,3))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(100,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
modelpath = 'C:/data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5' 
cp = ModelCheckpoint(filepath=modelpath, monitor='val_loss', save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
hist = model.fit(x_train, y_train, epochs=50, batch_size=16,verbose=1,validation_split=0.2, callbacks=[early_stopping,cp])

loss = model.evaluate(x_test,y_test,batch_size=16)
y_predict = model.predict(x_test)
y_test = np.argmax(y_test,axis=-1)
y_predict = np.argmax(y_predict,axis=-1)

#modelcheckpoint
#loss: 3.7109 - acc: 0.2602

