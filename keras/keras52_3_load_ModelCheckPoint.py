
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(-1,28,28,1)/255.

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#
# model = Sequential()
# model.add(Conv2D(filters = 256,kernel_size = (2,2),padding='same',strides = 1,input_shape=(28,28,1)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(Conv2D(256,kernel_size=(2,2)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(64,activation='relu'))
# model.add(Dense(32,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(10,activation='softmax'))

#model.save('C:/data/h5/k52_1_model1.h5')
#model.save_weights('C:/data/h5/k52_1_weight.h5')

#model1 = load_model('C:/data/h5/k52_1_model2.h5')
#model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#modelpath = 'C:/data/modelcheckpoint/k52_1_mnist_{epoch:02d}-{val_loss:.4f}.hdf5'
#cp = ModelCheckpoint(filepath = modelpath, monitor='val_loss',save_best_only = True, mode = 'auto')
#es = EarlyStopping(monitor='val_loss',patience=10)
# hist = model.fit(x_train,y_train,validation_split = 0.2,epochs=50,verbose=1,batch_size=16,callbacks=[es])#, CP])


#model.save('C:/data/h5/k52_1_model2.h5')

#4-1 evaluate, predict
#result1 = model1.evaluate(x_test,y_test,batch_size=16)

#print('model1_loss : ',result1[0])
#print('model1_accuracy : ',result1[1])

#4-2 evaluate, predict
# model.load_weights('C:/data/h5/k52_1_weight.h5')

# result1 = model.evaluate(x_test,y_test,batch_size=16)

# print('model_loss : ',result1[0])
# print('model_accuracy : ',result1[1])

model2 = load_model('C:/data/modelcheckpoint/k52_1_mnist_checkpoint.hdf5')    #hdf5->model과 weight가 들어가있음
result2= model2.evaluate(x_test,y_test,batch_size=8)
print("load_checkpoint_model_loss: ", result2[0])
print("load_checkpoint_model_accuracy: ", result2[1])

#weight값이 동일하기때문에 훈련값은 항상동일

#checkpoint model #loss가 최저를 찍은 부분이 저장됨
#load_checkpoint_model_loss:  0.06835830956697464
#load_checkpoint_model_accuracy:  0.980400025844574

# checkpoint가 성능이 더좋다.
# loss기준으로 더 줄어듬

#modelcheck point->훈련하는 동안 가중치를 저장
# 
# pandas에서 csv 로불러오면 엄청느려서 numpy로 바꿔줘야함. 