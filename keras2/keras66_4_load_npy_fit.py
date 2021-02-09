import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense, Dropout, MaxPooling2D,Flatten
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau

#1. data
x_train = np.load('C:/data/image/brain/data1/npy/keras66_train_x.npy')
y_train = np.load('C:/data/image/brain/data1/npy/keras66_train_y.npy')
x_test = np.load('C:/data/image/brain/data1/npy/keras66_test_x.npy')
y_test = np.load('C:/data/image/brain/data1/npy/keras66_test_y.npy')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (160, 150, 150, 3) (160,)
# (120, 150, 150, 3) (120,)


# from sklearn.preprocessing import MinMaxScaler
# scalar = MinMaxScaler()
# scalar.fit(x_train)
# x_train=scalar.transform(x_train)
# x_test = scalar.transform(x_test)

#모델만들기 실습


#2. model 
model = Sequential()
model.add(Conv2D(64,(3,3),padding='same',activation='relu',input_shape=(150,150,3)))
model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(Dropout(0.3))
# model.add(Conv2D(32,(5,5),padding='same',activation='relu'))
# model.add(Dropout(0.3))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
optimizer = Adam(lr=0.002)

#compile,train
model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['acc'])
es = EarlyStopping(monitor='val_loss',patience=3,mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.5,verbose=1)  
history = model.fit(x_train,y_train, verbose=1,validation_split=0.2, epochs=100,callbacks=[es,reduce_lr])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)   # 2행 1열중 첫번째
plt.plot(loss, marker='.', c='red', label='loss')
plt.plot(val_loss, marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) #2행 1열중 두번째
plt.plot(acc, marker='.', c='red', label='acc')
plt.plot(val_acc, marker='.', c='blue', label='val_acc')
plt.grid()

plt.title('Acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

print("acc :",acc[-1])
print("val_acc :",val_acc[:-1])