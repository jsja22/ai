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

y_train = x_train  #(60000, 28, 28, 1)
y_test = x_test     #(10000, 28, 28, 1)

print(y_train.shape)
print(y_test.shape)

# from sklearn.preprocessing import OneHotEncoder

# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)

# ohencoder = OneHotEncoder()
# ohencoder.fit(y_train)
# y_train = ohencoder.transform(y_train).toarray()
# y_test = ohencoder.transform(y_test).toarray()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Reshape
model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='same',
#                  strides=1, input_shape=(28, 28, 1)))
# model.add(MaxPooling2D(pool_size=2))
model.add(Dense(64, input_shape=(28,28,1)))
model.add(Dropout(0.5))
model.add(Flatten())  
model.add(Dense(64))
model.add(Dense(784, activation='relu'))  #reshape layer 28*28=784
model.add(Reshape((28,28,1)))  #layer shape change  #(None, 28, 28, 1)
model.add(Dense(1))

model.summary()   # (None, 28, 28, 1)

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,factor=0.5,verbose=1)  #5번해도 개선이 없으면 러닝레이트 (lr)0.5로 감축시키겠다는 뜻 
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')
modelpath= 'C:/data/modelcheckpoint/k57_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
hist = model.fit(x_train, y_train, batch_size=16, epochs=70, validation_split=0.5, callbacks=[early_stopping, cp,reduce_lr])
# 4. 평가, 예측
result = model.evaluate(x_test, y_test, batch_size=16)
y_pred = model.predict(x_test[:-10])
# y_recovery = np.argmax(y_pred, axis=1).reshape(-1,1)
# print(y_recovery)
# print("y_test : ", y_test[:-10])
# print("y_pred : ", y_recovery)
print("loss : ", result[0])
print("accuracy : ", result[1])

y_pred = model.predict(x_test)
print(y_pred[0])
print(y_pred.shape)  #(10000, 28, 28, 1)


##4차원 -> 4차원 덴스모델,conv2d 구성가능

