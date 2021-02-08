#가중치 저장할것
#1. model.save()
#2. pickle로도 해보기

#61카피해서
#model.cv_result를 붙여서 완성

import numpy as np
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense ,Dropout, Input
from tensorflow.keras.datasets import mnist

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
    # batchs = [8,16,32,64,128]
    # optimizers = ['rmsprop','adam','adadelta']
    # dropout = [0.1,0.2,0.3]
    batchs = [8]
    optimizers = ['adam']
    dropout = [0.1]
    return {"batch_size" : batchs, "optimizer":optimizers, "drop":dropout}

hyperparameters = create_hyperparameters()
model2 = build_model()

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1)   #머신러닝이 keras보다 먼저 나왔기에 무언가 알아먹게 해줘야하는데 그것이 이두줄로 정의된다.
#keras모델을 래핑을 해야 그리드서치나 랜덤서치가 알아먹는다

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint, TensorBoard, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=3,factor=0.5,verbose=1)  
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
modelpath = ('C:/data/modelcheckpoint/keras64_save-{epoch:02d}-{val_loss:.4f}.hdf5')
cp = ModelCheckpoint(modelpath, monitor='val_loss',save_weights_only=True, save_best_only=True, mode='auto')

search = RandomizedSearchCV(model2,hyperparameters,cv=3 )
search.fit(x_train,y_train, verbose=1,validation_split=0.2, epochs=1,callbacks=[reduce_lr,es,cp])

# #모델 저장
# model2.save_weights('C:/data/h5/k64_save.h5')
# #모델 로드
# model2 = load_model('C:/data/h5/k64_save.h5')

print(search.best_estimator_)   #<tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier object at 0x000001ED0BB8E430>
print(search.best_params_) #내가 선택한 파라미터 배피사이즈, 드랍아웃, 옵티마이저 중 가장 좋은것 #{'optimizer': 'adam', 'drop': 0.1, 'batch_size': 30}
print(search.best_score_)  #0.9574999809265137
acc = search.score(x_test,y_test)
print("최종 스코어 :", acc) #최종 스코어 : 0.970300018787384
print(search.cv_results_)

# import pickle
# #저장
# pickle.dump(search, open('C:/data/h5/boston.pickle.dat', 'wb'))     # write binary
# print('=====save complete=====')

# # 불러오기
# model2 = pickle.load(open('C:/data/h5/boston.pickle.dat', 'rb'))     # read binary
# print('======read complete=====')

# acc = search.score(x_test,y_test)
# print("최종 스코어 :", acc) 
from sklearn.metrics import accuracy_score
search.best_estimator_.model.save('C:/data/h5/boston0208.h5')
model3 = load_model('C:/data/h5/boston0208.h5')
y_pred = model3.predict(x_test)
y_pred = np.argmax(y_pred, axis = 1)
y_test = np.argmax(y_test, axis = 1)
# argmax 할것 ! 안하면 오류뜸 Classification metrics can't handle a mix of multilabel-indicator and continuous-multioutput targets
 
print(accuracy_score(y_test,y_pred)) #0.9638

#pickle로 save가 안됨 오류는 TypeError: cannot pickle '_thread.RLock' object
#model.save로 저장할때 search.best_estimator_.model.save('C:/data/h5/boston0208.h5')로 저장해주고 스코어는 acuuracy score로 본다.
