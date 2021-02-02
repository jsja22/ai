#conv2d model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('C:/data/computer vision/mnist_data/train.csv',index_col=None, header=0)
test = pd.read_csv('C:/data/computer vision/mnist_data/test.csv',index_col=None, header=0)
sub = pd.read_csv('C:/data/computer vision/mnist_data/submission.csv',index_col=None, header=0)

print(train.shape, test.shape) #(2048, 787) (20480, 786)
print(sub.shape) #[20480 rows x 2 columns]
print(train,test,sub) 

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255
y = temp.iloc[:,1]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y = y.to_numpy()
x_test = x_test.to_numpy()

x_train = x.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
#print(x_test.shape) #(20480, 28, 28, 1)

# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,shuffle=True, random_state=66)
# x_train = x_train.reshape(-1,28,28,1)
# x_test = x_test.reshape(-1,28,28,1)

# print(x_train.shape) #(1638, 28, 28, 1)
# print(x_test.shape) #(410, 28, 28, 1)

# one hot encoder
def one_hot_encoder(x):
    y = np.zeros((len(x), len(np.unique(x))))  # 모든 값이 0인 백터
    for i, num in enumerate(x):
        y[i][num] = 1  # Label에 해당하는 인덱스에 1을 입력
    return y

y_train = one_hot_encoder(train['digit'])
print(y_train.shape) #(2048, 10)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D ,BatchNormalization

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(2,2),padding='same',strides=1,activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(BatchNormalization())
model.add(Conv2D(64,(2,2),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Dense(128,activation='relu', input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(10,activation = 'softmax'))

model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=30, mode='auto')
model.fit(x_train, y_train, epochs=200, validation_split=0.2, callbacks=[early_stopping])

import matplotlib.pyplot as plt

plt.plot(model.history.history["acc"], label='model_acc')
plt.legend()
plt.show()

sub['digit'] = np.argmax(model.predict(x_test),axis=1)
print(sub.head())
sub.to_csv('C:/data/computer vision/submission/dacon_computer_vision_0202_1.csv', index=False)