import numpy as np
from glob import glob
from tqdm import tqdm
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,Model ,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,MaxPooling2D, Conv2D, Flatten, Reshape, Conv1D, MaxPooling1D, Input,LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime
p_path = os.getcwd()
print(p_path)
base_path = os.path.join('Renju_con', '*/*.npz')

npz_list = glob(base_path)
print(npz_list)

#model = load_model('C:/data/omok/modelcheckpoint/last_model.h5')
x_data, y_data =[],[]
for i in tqdm(npz_list):
    data = np.load(i)
    x_data.extend(data['inputs']) 
    y_data.extend(data['outputs'])  #append로 추가하면 안됨 로 추가하면 안됨 list요소 하나하나를 추가해야하니 expend 사용

    print(y_data)

x_data = np.array(x_data).reshape((-1,15,15,1))
y_data = np.array(y_data).reshape((-1,225))
print(x_data.shape) 
print(y_data.shape) 
#print(y_data)
'''
# (755268, 15, 15, 1)
# (755268, 225)
# (1964736, 15, 15, 1)
# (1964736, 225)
seed =2048
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=seed)

model = Sequential()
model.add(Conv2D(64, (4,4), activation='relu', padding='same', input_shape=(15, 15, 1)))
model.add(Conv2D(128,(5,5),activation='relu',padding='same'))
model.add(Conv2D(256,(5,5),activation='relu',padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128,(6,6),activation='relu',padding='same'))
model.add(Dropout(0.3))

model.add(Conv2D(128,(7,7),activation='relu',padding='same'))
model.add(Conv2D(64,(7,7),activation='relu',padding='same'))

model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(255,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(225,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(225,activation='sigmoid'))


model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])
start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
mc = ModelCheckpoint('C:/data/omok/modelcheckpoint/last_last_%s.h5' % (start_time), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=8, verbose=1, mode='auto')
es = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
model.fit(x=x_train,y=y_train, batch_size=256,validation_data=(x_val, y_val),epochs=150,callbacks=[mc,lr,es], use_multiprocessing=True,workers=16)
# # use_multiprocessing=True 사용하면 gpu batch_size를 계속 올려서 gpu가 쉬지않고 돌게 할수 있음

#model = load_model('C:/data/omok/modelcheckpoint/last_model.h5')

#print(x_val[800])
y_pred = model.predict(np.expand_dims(x_val[800], axis=0)).squeeze() #squeeze는 차원 중 사이즈가 1인 것을 찾아 스칼라값으로 바꿔 해당 차원을 제거
y_pred = y_pred.reshape((15, 15))
print("y_pred",y_pred)
print(y_pred.shape)
y, x = np.unravel_index(np.argmax(y_pred), y_pred.shape)
#np.unravel_index ->플랫 인덱스 또는 플랫 인덱스 배열을 좌표 배열의 튜플로 변환
print(y_pred[y, x]*100,"%로 가장 predict값이 높은 다음 돌의 위치는?",x, y,"이다!!")

# #1. val_loss: 0.0184 - val_acc: 0.2354
# #2. dense255 ->  val_loss: 0.0113 - val_acc: 0.5730
# #3. batch_normalization_1 (Batch (None, 3, 3, 64)  -> val_loss: 0.0133 - val_acc: 0.5022
# #4. dense 255(1) ->val_loss: 0.0117 - val_acc: 0.5583 -> 7 8 0.95984364
# #5. ddense 255(2)-> val_loss: 0.0097 - val_acc: 0.6306 -> 7 8 0.99943084
# #6. loss: 0.0062 - acc: 0.7709 - val_loss: 0.0088 - val_acc: 0.6969  -> recent model
# #val_loss: 0.0082 - val_acc: 0.8020 #epoch20
# #7. last -> loss: 0.0043 - acc: 0.8348 - val_loss: 0.0063 - val_acc: 0.7852 , 7 8 0.99983644
'''