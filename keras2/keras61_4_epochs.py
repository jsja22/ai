#epochs 100
# validation_split, callbacks 적용
#es 5 적용
#reduce_lr 3적용
#cp 적용
#modelcheckpoihnt 폴더에 hdf5 파일저장

import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense ,Dropout, Input,Conv2D, Flatten, MaxPooling2D 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
from tensorflow.keras.layers import LeakyReLU

(x_train,y_train), (x_test,y_test) = mnist.load_data()
print(x_test.shape)

#1. 데이터 / 전처리

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5,optimizer='adam',activation='relu',lr=0.001,nodes=64):
    Inputs = Input(shape=(28*28,),name='input')
    x = Dense(512, activation='relu',name='hiddne1')(Inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu',name='hiddne2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu',name='hiddne3')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu',name='hiddne4')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax',name='outputs')(x)

    model = Model(inputs=Inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    batchs = [8,16,32,64,128]
    optimizers = ['rmsprop','adam','adadelta','SGD']
    nodes =[32,64,128]
    lr=[0.1, 0.01, 0.001]
    dropout = [0.2,0.3,0.5]
    activation = ['tanh', 'relu', "selu", "softmax", "sigmoid", LeakyReLU()] #{'optimizer': 'adam', 'nodes': 32, 'lr': 0.01, 'drop': 0.3, 'batch_size': 64, 'activation': 'sigmoid'}
    return {"batch_size" : batchs, "optimizer":optimizers, "drop":dropout, "activation":activation, "lr":lr, "nodes":nodes}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)   #머신러닝이 keras보다 먼저 나왔기에 무언가 알아먹게 해줘야하는데 그것이 이두줄로 정의된다.
#keras모델을 래핑을 해야 그리드서치나 랜덤서치가 알아먹는다

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.5,verbose=1)  
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath = ('C:/data/modelcheckpoint/keras61_4_epochs-{epoch:02d}-{val_loss:.4f}.hdf5')
cp = ModelCheckpoint(modelpath, monitor='val_loss', save_best_only=True, mode='auto')
search = RandomizedSearchCV(model2,hyperparameters,cv=3 )
search.fit(x_train,y_train, verbose=1,validation_split=0.2, epochs=100,callbacks=[reduce_lr,es,cp])
print(search.best_estimator_)  
print(search.best_params_) 
print(search.best_score_)  
acc = search.score(x_test,y_test)
print("최종 스코어 :", acc) 

#dense model es,cp,reudce_lr 적용하고 epochs100으로 한 결과 !
# <tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000002080106BE50>
# {'optimizer': 'adam', 'nodes': 32, 'lr': 0.01, 'drop': 0.3, 'batch_size': 64, 'activation': 'sigmoid'}
# 0.9781833291053772
# 157/157 [==============================] - 0s 1ms/step - loss: 0.0815 - acc: 0.9852
# 최종 스코어 : 0.9851999878883362