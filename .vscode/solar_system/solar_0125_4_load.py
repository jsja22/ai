##WS 뺸거


import pandas as pd
import numpy as np
import os
import glob
import sys
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential,Model ,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,MaxPooling2D, Conv2D, Flatten, Reshape, Conv1D, MaxPooling1D, Input,LeakyReLU
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import mathfrom math import radians
#DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))   (W/m2)     0 ~ 528
#DNI - 직달일사량(Direct Normal Irradiance (W/m2))               (W/m2)     0 ~ 1569    
#WS - 풍속(Wind Speed (m/s))                                     (m/s)      0.0 ~ 12.0
#RH - 상대습도(Relative Humidity (%))                            (%)        0.00 ~ 100.00
#T - 기온(Temperature (Degree C))                                (C)        -19 ~ 35
# Target : 1
#Target - 태양광 발전량 (kW)                                     (kW)

#train 데이터 준비 
train= pd.read_csv('C:/data/csv/solar/train/train.csv',index_col=None, header=0)
print(train .shape)     #(52560, 9)
print(train .tail())
submission = pd.read_csv('C:/data/csv/solar/sample_submission.csv')

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

quantiles = [.09, .19, .28, .37, .48, .59, .702, .8, .9]
day = 7 

def split_to_seq(data): 
    tmp = []
    for i in range(48):
        tmp1 = pd.DataFrame()
        #train에서 48시간을 간격으로 되어있는 시간 분이 같은 데이터를 묶어줄거임
        for j in range(int(len(data)/48)):
            tmp2 = data.iloc[j*48+i,:]
            tmp2 = tmp2.to_numpy()
            #tmp2.shape가 (9,0)으로 바뀌기 때문에 reshape(1,9)로 바꿔줘야함
            tmp2 = tmp2.reshape(1,tmp2.shape[0])
            tmp2 = pd.DataFrame(tmp2)
            tmp1 = pd.concat([tmp1,tmp2])
            #1093일치 9개의 컬럼에대해서 같은 시간 같은 분으로 쭉 묶었음
        x = tmp1.to_numpy()  #묶은거 numpy로 변환
        tmp.append(x) 
    return np.array(tmp)
def Conv1dml():
    model = Sequential()
    model.add(Conv1D(256,2,padding = 'same', activation = 'relu',input_shape = (7,7)))
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))
    return model

def preprocess_data(data,is_train=True):
    #GHI 계산
    #GHI = DHI + DNIx(cos세타제트)
    #Cosθz = CosФCosбCosω + SinρSinФ
    #w -> 시간각 w= 15*시간 (12시 기준으로 달라짐 +,-)
    #б -> 일적위 -23.45*np.cos(360/365 *(x+10))
    #위도는 대한민국 10개 도시 평균 36도로 기준잡자!
    # SinρSinФ -> (np.sin(np.sin(train['declination']* np.sin(36))
    #시간각 구하기(w)   12시기준으로 12시 이후 1시간 단위로 15도씩 증가 12시 이전으로 1시간 단위로 -15도 씩 증가
    angle = 15
    noon = 12 
    latitude = radians(36)
    data['time_angle'] = [(x - noon ) * angle if x >= noon  else -(noon  - x) * angle for x in data.Hour]

    #일적위 구하기 -23.45*np.cos(360/365 *(x+10))
    data['declination'] = [-23.44 * np.cos(360 /365 *(x+10)) for x in data.Day]
    
    #cos세타제트 구하기  => CosФCosбCosω + SinρSinФ
    data['theta_z'] = 90 - 1/(np.sin(np.sin(data['declination']) *np.sin(36) + \
    np.cos(data['declination'])*np.cos(36) * np.cos(data['time_angle'])))    

    #GHI 구하기  ->DHI + DNIx(cos세타제트)
    data['GHI'] = data.DHI + data.DNI * np.cos(data.theta_z)

    #data['Time'] = data['Hour'] + data['Minute']*(0.5/30)
    #data['sin_time'] = np.sin(2*np.pi*data.Time/24)
    #data['cos_time'] = np.cos(2*np.pi*data.Time/24)
    
    temp = data.copy()
    temp = temp[['DHI','DNI','GHI','T','RH','TARGET']]
    

    if is_train == True:
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:
        return temp.iloc[-48*day:, :]


df_train = preprocess_data(train)
scale.fit(df_train.iloc[:,:-2])
df_train.iloc[:,:-2] = scale.transform(df_train.iloc[:,:-2])

x_test = []

for i in range(81):
    file_path = 'C:/data/csv/solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp,is_train=False)
    temp = scale.transform(temp)
    temp = pd.DataFrame(temp)
    #똑같이 test데이터도 7일씩 시간순으로 되어 있는 것을 1일의 0시0분~7일의 0시 0분 까지로 한묶음으로 처리해서 48시간으로 분리
    temp = split_to_seq(temp)
    #print(temp.shape)
    x_test.append(temp)

test = np.array(x_test) #(81, 48, 7, 9)
train = split_to_seq(df_train)

def split_xy(data,timestep):
    x, y1, y2 = [],[],[]
    for i in range(len(data)):
        x_end = i + timestep
        if x_end>len(data):
            break
        tmp_x = data[i:x_end,:-2]
        tmp_y1 = data[x_end-1:x_end,-2]
        tmp_y2 = data[x_end-1:x_end,-1]
        x.append(tmp_x)
        y1.append(tmp_y1)
        y2.append(tmp_y2)
    return(np.array(x),np.array(y1),np.array(y2))


x,y1,y2 = [],[],[]
for i in range(48):
    tmp1,tmp2,tmp3 = split_xy(train[i],day)
    x.append(tmp1)
    y1.append(tmp2)
    y2.append(tmp3)

x = np.array(x) 
y1 = np.array(y1) 
y2 = np.array(y2) 

#2. 모델링
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.2, verbose = 1)
epochs = 10000
bs = 32

for i in range(48):
    #x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x[i],y1[i],y2[i], train_size = 0.7,shuffle = True, random_state = 0)
    if i%2 == 0:
      minute = 0
    elif i%2 == 1:
      minute = 30
    hour = int(i/2) 
    
    for j in quantiles:
        
        print("##############내일 {}시,{}분, q_0.{} 훈련 시작!!###########".format(hour,minute,j))
      
        filepath_cp = f'C:/data/modelcheckpoint/solar_checkpoint_0125_6{i:2d}_y1seq_{j:.1f}.hdf5'
        model = load_model(filepath_cp, compile = False)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        x = []
        for k in range(81):
            x.append(test[k,i])
        x = np.array(x)
        df_temp1 = pd.DataFrame(model.predict(x).round(2))
        # df_temp1 = pd.concat(pred, axis = 0)
        df_temp1[df_temp1<0] = 0
        num_temp1 = df_temp1.to_numpy()
        if i%2 == 0:
            submission.loc[submission.id.str.contains(f"Day7_{hour}h00m"), [f"q_{j:.1f}"]] = num_temp1
        elif i%2 == 1:
            submission.loc[submission.id.str.contains(f"Day7_{hour}h30m"), [f"q_{j:.1f}"]] = num_temp1
    
    for j in quantiles:
        
        print("##############모레 {}시,{}분 q_0.{} 훈련 시작!!############".format(hour,minute,j))
        
        filepath_cp = f'C:/data/modelcheckpoint/solar_checkpoint_0125_6{i:2d}_y2seq_{j:.1f}.hdf5'
        model = load_model(filepath_cp, compile = False)
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        
        x = []
        for k in range(81):
            x.append(test[k,i])
        x = np.array(x)
        df_temp2 = pd.DataFrame(model.predict(x).round(2))
        # df_temp1 = pd.concat(pred, axis = 0)
        df_temp2[df_temp2<0] = 0
        num_temp2 = df_temp2.to_numpy()
        if i%2 == 0:
            submission.loc[submission.id.str.contains(f"Day8_{hour}h00m"), [f"q_{j:.1f}"]] = num_temp2
        elif i%2 == 1:
            submission.loc[submission.id.str.contains(f"Day8_{hour}h30m"), [f"q_{j:.1f}"]] = num_temp2

submission.to_csv('C:/data/csv/solar/sample_submission0125_last2.csv', index = False)


#####WS뺀게 점수더 좋음!
