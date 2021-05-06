import numpy as np
import tensorflow as tf
import autokeras as ak
from tensorflow.keras.datasets import mnist

(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255.
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255.

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = ak.ImageClassifier(
    overwrite=True,
    max_trials=2,
    loss= 'mse',
    metrics = ['acc']
)
###########
#model.summary()
#모델을 자동으로 완성된 다음에 서머리가 나온다 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

es= EarlyStopping(monitor='val_loss',mode='min', patience=6)
lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=2)
ck = ModelCheckpoint('C:/data/modelcheckpoint/', save_best_only=True, save_weights_only=True, monitor='val_loss', verbose=1)


model.fit(x_train,y_train, epochs=1, validation_split=0.2, callbacks=[es,lr,ck])

results = model.evaluate(x_test,y_test)

print(results)

model2 = model.export_model() #원래 모델형식으로 내보겠다는 뜻

model2.save('C:/data/save/aaa.h5')
#시간이 오래걸리지만 3번 epoch돌렸을때 결과 
# 313/313 [==============================] - 2s 6ms/step - loss: 0.0413 - accuracy: 0.9859
# [0.04129353165626526, 0.9858999848365784]

