import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense ,Dropout, Input
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
(x_train,y_train), (x_test,y_test) = mnist.load_data()
print(x_test.shape)

#1. 데이터 / 전처리

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000,28*28).astype('float32')/255.
x_test = x_test.reshape(10000,28*28).astype('float32')/255.

#2. 모델
def build_model(drop=0.5,optimizer='adam'):
    Inputs = Input(shape=(28*28,),name='input')
    x = Dense(512, activation='relu',name='hiddne1')(Inputs)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu',name='hiddne2')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu',name='hiddne3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax',name='outputs')(x)

    model = Model(inputs=Inputs, outputs=outputs)
    model.compile(optimizer=optimizer, metrics=['acc'],loss='categorical_crossentropy')

    return model

def create_hyperparameters():
    batchs = [8]
    optimizers = ['adam']
    dropout = [0.1]
    return {"clf__batch_size" : batchs, "clf__optimizer" : optimizers, "clf__drop" : dropout}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline, make_pipeline  
model2 = KerasClassifier(build_fn=build_model, epochs=1, batch_size=32, verbose=1)   #머신러닝이 keras보다 먼저 나왔기에 무언가 알아먹게 해줘야하는데 그것이 이두줄로 정의된다.
#keras모델을 래핑을 해야 그리드서치나 랜덤서치가 알아먹는다

kfold = KFold(n_splits=3, random_state=66)

#Pipeline에서는 우리가 clf 이런식으로 이름을 정의할 수 있지만 make_pipeline에서는 kerasclassifier_ 과함께 정의되어있는는걸 사용해줘야함!

pipe = Pipeline([("scaler",MinMaxScaler()),('clf',model2)])
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = GridSearchCV(pipe,hyperparameters,cv=kfold )

search.fit(x_train,y_train)  #fit에서 배치사이즈 에폭 벌보스를 정의하지않고 classifier에서 정의해줘야 컴파일 됨~!
print(search.best_estimator_)   
print(search.best_params_) 
print(search.best_score_)  
acc = search.score(x_test,y_test)
print("최종 스코어 :", acc) 

#epochs= 1
# {'clf__optimizer': 'adam', 'clf__drop': 0.1, 'clf__batch_size': 8}
# 0.9591000080108643
# 1250/1250 [==============================] - 1s 917us/step - loss: 0.1153 - acc: 0.9636
# 최종 스코어 : 0.9635999798774719