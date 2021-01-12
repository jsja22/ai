
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#(x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
print(x_train.shape)
print(x_test.shape)

#x_train = x_train.reshape(x_train.shape[0],28,28,1)
#x_test = x_test.reshape(x_test.shape[0],28,28,1)
#x_val = x_val.reshape(x_val.shape[0],28,28,1)

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.  #(0~1사이로 수렴) =>전처리
x_test = x_test.reshape(10000,28,28,1)/255.  #

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1) 
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

print(y_train.shape)
print(y_test.shape)

from tensorflow.keras.models import Sequential ,Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=100, kernel_size=(2,2), padding='same',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(30,2))
model.add(Flatten())
model.add(Dense(48, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
modelpath = 'C:/data/modelcheckpoint/k45_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
early_stopping = EarlyStopping(monitor='loss', patience=5, mode='auto')
cp =  ModelCheckpoint(filepath=modelpath, monitor='val_loss',save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, epochs=50, batch_size=32,validation_split=0.2, callbacks=[early_stopping,cp])

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test, batch_size=32)
print(loss, acc)
'''
y_pred = model.predict(x_test[:-10])
y_pred1 = np.argmax(y_pred, axis=1).reshape(-1,1)
print("y_pred1",y_pred1)
print(y_test[-10:-1])
'''

plt.figure(figsize=(10,6))  #figsize 판깔아주는거 #면적을 잡아주는것

plt.subplot(2,1,1)  #(2,1)짜리 그림을 만들겠다 (2행1열중에 첫번쨰)
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.grid()

plt.title('cost loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')


plt.subplot(2,1,2)  #(2,1)짜리 그림을 만들겠다 (2행1열중에 두번쨰)
plt.plot(hist.history['accuracy'],marker='.',c='red',label='accuracy')
plt.plot(hist.history['val_accuracy'],marker='.',c='blue',label='val_accuracy')
plt.grid()

plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

#cnn
#loss:1.1892050504684448 acc:0.8845000267028809

#modelcheckpoint
#0.6187791228294373 0.8920000195503235