#동재모델 참고해서 앙상블 돌리기 
#피쳐가 190이상인것은 알파벳과 숫자가 곂쳐져있다고 보고 그이하인것들 피
#피쳐값을 반으로 줄여서 기존의 모델과 앙상블하면 알파벳에 숨겨진 숫자가 더욱 돋보이지 않을까? 하는 생각!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization, Input,concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception
import warnings

def ensemblemodel():
   
    input1 = Input(shape=(28,28,1))
    dense1 = Conv2D(64,(3,3),activation='relu', padding='same',strides=1)(input1)
    dense1 = MaxPooling2D(pool_size=2)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    dense1 = Conv2D(32,(3,3),activation='relu',padding='same',strides=1)(dense1)
    dense1 = MaxPooling2D(pool_size=2)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)
    dense1 = Conv2D(32,(5,5),activation='relu',padding='same',strides=1)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Conv2D(32,(5,5),activation='relu',padding='same',strides=1)(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = MaxPooling2D(pool_size=2)(dense1)
    dense1 = Dropout(0.3)(dense1)
    dense1 = Flatten()(dense1)
    dense1 = Dense(128,activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)
    dense1 = Dense(64,activation='relu')(dense1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)
    middle1 = Dense(32,activation='relu')(dense1)
    
    input2 = Input(shape=(28,28,1))
    dense2 = Conv2D(64,(3,3),activation='relu', padding='same',strides=1)(input2)
    dense2 = MaxPooling2D(pool_size=2)(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.4)(dense2)
    dense2 = Conv2D(32,(3,3),activation='relu',padding='same',strides=1)(dense2)
    dense2 = MaxPooling2D(pool_size=2)(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    dense2 = Conv2D(32,(5,5),activation='relu',padding='same',strides=1)(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Conv2D(32,(5,5),activation='relu',padding='same',strides=1)(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = MaxPooling2D(pool_size=2)(dense2)
    dense2 = Dropout(0.3)(dense2)
    dense2 = Flatten()(dense2)
    dense2 = Dense(128,activation='relu')(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    dense2 = Dense(64,activation='relu')(dense2)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    middle2 = Dense(32,activation='relu')(dense2)

    middle = concatenate([middle1,middle2])
    dense3 = Dense(32,activation='relu')(middle)
    output = Dense(10,activation='softmax')(dense3)

    model = Model(inputs = [input1,input2],outputs =output)

    return model

train = pd.read_csv('C:/data/dacon/computer_vision//train.csv')
test = pd.read_csv('C:/data/dacon/computer_vision/test.csv')
sub = pd.read_csv('C:/data/dacon/computer_vision/submission.csv',header=0)

print(train.shape, test.shape) #(2048, 787) (20480, 786)
print(sub.shape) #[20480 rows x 2 columns]
print(train,test,sub) 

train_x_data = train.drop(['id','digit','letter'],axis=1)
train_y_data = train.loc[:,'digit']

test_data = test.drop(['id','letter'],axis=1)

x1_train = train_x_data.to_numpy().reshape(-1,28,28,1)
y1_train = train_y_data.to_numpy()

x2_train = train_x_data.copy()
x2_train[x2_train<190] /=2
x2_train = x2_train.to_numpy().reshape(-1,28,28,1)

x1_test = test_data.to_numpy().reshape(-1,28,28,1)
x2_test = test_data.copy()
x2_test[x2_test<190] /=2
x2_test = x2_test.to_numpy().reshape(-1,28,28,1)

# print(x1_train.shape) #(2048, 28, 28, 1)
# print(y1_train.shape) #(2048,)
# print(x1_test.shape) #(20480, 28, 28, 1)

x1_train = x1_train/255.0
x2_train = x2_train/255.0
x1_test = x1_test/255.0
x2_test = x2_test/255.0


skf = StratifiedKFold(n_splits=40, random_state=66, shuffle=True)

datagen = ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2)
datagen2 = ImageDataGenerator()

lr = ReduceLROnPlateau(monitor='val_loss',patience=70, factor=0.5)
cp = ModelCheckpoint('C:/data/dacon/computer_vision/h5/Dacon_computer_vision_0203_8.h5',moniotr='val_loss',save_best_only=True, verbose=1 )
es = EarlyStopping(monitor = 'val_loss',patience=160, verbose=1,mode='auto')

result = []

i= 0
for train_index, valid_index in skf.split(x1_train,train['digit']):
    
    x1_train = x1_train[train_index]
    x1_valid = x1_train[valid_index]
    y1_train = y1_train[train_index]
    y1_valid = y1_train[valid_index]
    x2_train = x2_train[train_index]
    x2_valid = x2_train[valid_index]

    train_generator = datagen.flow([x1_train,x2_train],y1_train,batch_size=8)
    valid_generator = datagen2.flow([x1_valid,x2_valid],y1_valid)
    test_generator = datagen2.flow(x1_test,shuffle=False)

    model = ensemblemodel()
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])
    model.fit_generator(train_generator, epochs = 1000, validation_data= valid_generator, callbacks = [es,cp,lr])
    
    model.load_weights('C:/data/dacon/computer_vision/h5/Dacon_computer_vision_0203_8.h5')
    # result += model.predict_generator(test_generator,verbose=True)/30

    # hist = pd.DataFrame(learning_history.history)
    # val_loss_min.append(hist['val_loss'].min())

    # i +=1
    # print(i,'번째 학습 완료')

    y_pred = model.predict([x1_test,x2_test])
    y_pred = y_pred.argmax(1)
    result.append(y_pred)
result = np.array(result)
mode = stats.mode(result).mode
mode = np.transpose(mode)
sub.loc[:, 'digit'] = mode

sub.drop('tmp', axis = 1,inplace=True)
sub.to_csv('C:/data/dacon/computer_vision/submission/dacon_computer_vision_0203_8.csv', index=False)

count = stats.mode(result).count
for i in count[0]:
    print(i)