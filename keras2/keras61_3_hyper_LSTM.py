#cnn으로 수정
#파라미터 변경할것
#노드의 갯수 

import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense ,Dropout, Input,Conv2D, Flatten, MaxPooling2D ,LSTM
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, Adadelta, Adamax, Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam
(x_train,y_train), (x_test,y_test) = mnist.load_data()
print(x_test.shape)

#1. 데이터 / 전처리

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28,28).astype('float32')/255.
x_test = x_test.reshape(10000,28,28).astype('float32')/255.

#2. 모델
nodes = 64
def model_layers():
    x = Dense(64,activation='relu')
    return x
def build_model(drop=0.5,optimizer='adam',activation='relu',lr=0.1,nodes=64,kernel_size=(2,2)):
    Inputs = Input(shape=(28,28),name='input')
    x = LSTM(512,activation=activation,name='hiddne1')(Inputs)
    x = Dropout(drop)(x)
    x = Dense(nodes, activation=activation,name='hiddne2')(x)
    x = Dense(nodes, activation=activation,name='hiddne3')(x)
    x = Dense(nodes, activation=activation,name='hiddne4')(x)
    
    outputs = Dense(10, activation='softmax',name='outputs')(x)
    model = Model(inputs=Inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],loss='categorical_crossentropy')

    return model
from tensorflow.keras.layers import LeakyReLU
 
def create_hyperparameters():
    batchs = [8,16,32,64,128]
    optimizers = ['rmsprop','adam','adadelta','SGD']
    nodes =[32,64,128]
    lr=[0.1, 0.01, 0.001]
    dropout = [0.2,0.3,0.5]
    activation = ['tanh', 'relu', "selu", "softmax", "sigmoid", LeakyReLU()]
    kernel_size = [2, 3, 4]
    epochs = [100, 200, 300]
    return {"batch_size" : batchs, "optimizer":optimizers, "drop":dropout, "activation":activation, "lr":lr, "nodes":nodes,"kernel_size": kernel_size, "epochs": epochs}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)   #머신러닝이 keras보다 먼저 나왔기에 무언가 알아먹게 해줘야하는데 그것이 이두줄로 정의된다.
#keras모델을 래핑을 해야 그리드서치나 랜덤서치가 알아먹는다

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model2,hyperparameters,cv=3 )

search.fit(x_train,y_train, verbose=1)

####lr추가하지않았을떄##############
print(search.best_estimator_)   #<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x0000012707D0CA00>
print(search.best_params_) #내가 선택한 파라미터 배피사이즈, 드랍아웃, 옵티마이저 중 가장 좋은것 #{'optimizer': 'adam', 'drop': 0.3, 'batch_size': 32, 'activation': 'relu'}
print(search.best_score_)  #0.9751666585604349
acc = search.score(x_test,y_test)
print("최종 스코어 :", acc) #최종 스코어 : 최종 스코어 : 0.980400025844574  

####lr추가했을떄##############
