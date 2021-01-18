import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   #(10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])
print(x_train[0].shape) # (28, 28)

# plt.imshow(x_train[0], 'gray')
# # plt.imshow(x_train[0])
# plt.show()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32')/255.  # 전처리
x_test = x_test.reshape(10000, 28, 28, 1)/255.  # 전처리

from sklearn.preprocessing import OneHotEncoder

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

ohencoder = OneHotEncoder()
ohencoder.fit(y_train)
y_train = ohencoder.transform(y_train).toarray()
y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same',
                 strides=1, input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv2D(1, 2))
model.add(Conv2D(1, 2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.5,verbose=1)  #5번해도 개선이 없으면 러닝레이트 (lr)0.5로 감축시키겠다는 뜻 
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
modelpath= 'C:/data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
tb = TensorBoard(log_dir='../data/graph',histogram_freq=0,write_graph=True,write_images=True) #  train, validation in graph folder 
hist = model.fit(x_train, y_train, batch_size=16, epochs=70, validation_split=0.5, callbacks=[early_stopping, cp,tb,reduce_lr])
# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=16)
y_pred = model.predict(x_test[:-10])
# y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_recovery)
# print("y_test : ", y_test[:-10])
# print("y_pred : ", y_recovery)
print("loss : ", result[0])
print("accuracy : ", result[1])

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)   # 2행 1열중 첫번째
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()

plt.title('Cost Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2) #2행 1열중 두번째
plt.plot(hist.history['accuracy'], marker='.', c='red', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='blue', label='val_accuracy')
plt.grid()

plt.title('Acurracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='upper right')

plt.show()

#modelcheckpoint
#loss :  0.04159597307443619
#accuracy :  0.9879999756813049