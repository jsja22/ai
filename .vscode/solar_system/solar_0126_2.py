import pandas as pd
import numpy as np
import os
import glob
import sys
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.models import Sequential,Model ,load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,MaxPooling2D, Conv2D, Flatten, Reshape, Conv1D, MaxPooling1D, Input,LeakyReLU
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import math
from math import radians
from datetime import date
import time
#DHI - 수평면 산란일사량(Diffuse Horizontal Irradiance (W/m2))   (W/m2)     0 ~ 528
#DNI - 직달일사량(Direct Normal Irradiance (W/m2))               (W/m2)     0 ~ 1569    
#WS - 풍속(Wind Speed (m/s))                                     (m/s)      0.0 ~ 12.0
#RH - 상대습도(Relative Humidity (%))                            (%)        0.00 ~ 100.00
#T - 기온(Temperature (Degree C))                                (C)        -19 ~ 35
# Target : 1
#Target - 태양광 발전량 (kW)                                     (kW)


def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)


##quantiles 분할해주는게 조금더 성능좋음!
quantiles = [.09, .19, .28, .37, .48, .59, .702, .8, .9]
day = 7 

def season24(Date):
  target = 0
  for i in seasonlist:
    if Date < i:
      target = seasonlist.index(i) - 1
      break
  if Date < 5:
    target = 23
  return target

def sep_season(date):
  date1_1 = date(2020,1,1)
  results = (data-date1_1).days
  return results

m=60
남중 = [12+36/m, 12+41/m, 12+44/m, 12+44/m, 12+42/m, 12+38/m, 12+34/m, 12+29/m, 12+27/m, 12+27/m, 12+29/m, 12+32/m, 12+35/m, 12+37/m, 12+36/m, 12+33/m, 12+29/m, 12+23/m, 12+18/m, 12+15/m, 12+14/m, 12+16/m, 12+22/m, 12+29/m]
소한 = sep_season(date(2020,1,5))
대한 = sep_season(date(2020,1,20))
입춘 = sep_season(date(2020,2,4))
우수 = sep_season(date(2020,2,19))
경칩 = sep_season(date(2020,3,5))
춘분 = sep_season(date(2020,3,20))
청명 = sep_season(date(2020,4,4))
곡우 = sep_season(date(2020,4,19))
입하 = sep_season(date(2020,5,5))
소만 = sep_season(date(2020,5,20))
망종 = sep_season(date(2020,6,5))
하지 = sep_season(date(2020,6,21))
소서 = sep_season(date(2020,7,7))
대서 = sep_season(date(2020,7,22))
입추 = sep_season(date(2020,8,7))
처서 = sep_season(date(2020,8,23))
백로 = sep_season(date(2020,9,7))
추분 = sep_season(date(2020,9,22))
한로 = sep_season(date(2020,10,8))
상강 = sep_season(date(2020,10,23))
입동 = sep_season(date(2020,11,7))
소설 = sep_season(date(2020,11,22))
대설 = sep_season(date(2020,12,7))
동지 = sep_season(date(2020,12,21))

seasonlist = [소한, 대한, 입춘, 우수, 경칩, 춘분, 청명, 곡우, 입하, 소만, 망종, 하지, 소서, 대서, 입추, 처서, 백로, 추분, 한로, 상강, 입동, 소설, 대설, 동지]
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
    model.add(Conv1D(128,2,padding = 'same', activation = 'relu',input_shape = (7,7)))
    model.add(Conv1D(64,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Conv1D(32,2,padding = 'same', activation = 'relu'))
    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(16, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dense(1))
    return model
l_angle = [32.92, 36.83, 40.75, 44.67, 48.58, 52.5, 56.42, 60.33, 64.25, 68.16, 72.01, 76, 72.1, 68.16, 64.25, 60.33, 56.42, 52.5, 48.58, 44.67, 40.75, 36.83, 32.92, 29]
##l_angel 다시 계산해야함 

def GHI1(DHI, DNI, season, hour):
    #GHI 계산
    #GHI = DHI + DNIx(cos세타제트)
    #Cosθz = CosФCosбCosω + SinρSinФ
    #w -> 시간각 w= 15*시간 (12시 기준으로 달라짐 +,-)
    #б -> 일적위 -23.45*np.cos(360/365 *(x+10))
    #위도는 대한민국 10개 도시 평균 36도로 기준잡자!
    # SinρSinФ -> (np.sin(np.sin(train['declination']* np.sin(36))
    #시간각 구하기(w)   12시기준으로 12시 이후 1시간 단위로 15도씩 증가 12시 이전으로 1시간 단위로 -15도 씩 증가
    #여름은 100RH중 60~100RH정도되고,겨울은 100RH중 10~50RH정도 따라서 RH도 시즌을 구별할수있는 지표가 될 수 있다. 
    ###########train, test를 시즌별로 나눠서 칼럼을 추가함 (시즌은 태양의 일출 일몰 시간에따라서 )

    
    angle = 15
    latitude = radians(36) #포항을 기준으로 함
    #계절별로 위도가 다르기 때문에 세분화 해보자!
    season = int(season)
    angle = radians(l_angle[season])

    altitude = radians(15*(hour - 남중[season]))  #시즌별 남중고도에 따라 구함 (시간각) 
    
    #일적위 구하기 -23.45*np.cos(360/365 *(x+10))
    #data['declination'] = [-23.44 * np.cos(360 /365 *(x+10)) for x in data.Day365]
    #일적위를 계절별로 구해야함 해당하는 날짜를 x로 잡고 24절기에 대해 -23.45*np.cos(360/365 *(x+10)공식에 대입해서 총 24절기의 일적위를 list로 만들어야함
    #그렇게 만든게 l_angle 인데 24개를 다 구하기 귀찮으니 일단 가져와서 사용해보고 모델이 돌아가면 공식 대입해서 다시 돌리기 !
    #cos세타제트 구하기(천정각) => https://blog.naver.com/sshoe77/220986037173 공식은 여기 참조 
    theta_z = radians(90) - (np.arcsin(np.sin(angle) *np.sin(latitude) +\
    np.cos(angle)*np.cos(latitude) * np.cos(altitude)))    

    #GHI 구하기  ->DHI + DNIx(cos세타제트)#
    ghi = DHI + DNI * np.cos(theta_z)
    
    return ghi
def preprocess_data(data,is_train=True):
    
    if is_train == True:
        data['Day365'] = data['Day']
        data['Day365'] = data['Day365']%365
        data['Time'] = data['Hour'] + data['Minute']*(0.5/30)
        data['season'] = data.apply(lambda x: season24(x['Day365']), axis = 1)  #1.시즌열 만들어주기  
        #1095일 3년짜리를 365일씩 끊어준 날짜를 각 날짜에 대한 24절기로 구분시켜서 시즌 컬럼을 만들어준다.
        data['GHI'] = data.apply(lambda x: GHI1(x.DHI, x.DNI, x.season, x.Time), axis=1)
        temp = data.copy()
        temp = temp[['DHI','DNI','GHI','T','RH','season','TARGET']]
    
        temp['TARGET1'] = temp['TARGET'].shift(-48).fillna(method = 'ffill')
        temp['TARGET2'] = temp['TARGET'].shift(-96).fillna(method = 'ffill')
        temp = temp.dropna()
        return temp.iloc[:-96]

    elif is_train == False:

        ###test 데이터에도 시즌을 나누어줘야함 
        ##일단 만들어진 데이터 사용함
        data['Time'] = data['Hour'] + data['Minute']*(0.5/30)
        data['GHI'] = data.apply(lambda x: GHI1(x.DHI, x.DNI, x.season, x.Time), axis=1)
        temp = data.copy()
        temp = temp[['DHI','DNI','GHI','T','RH','season','TARGET']]
        return temp.iloc[-48*day:, :]

#train 데이터 준비 
train= pd.read_csv('C:/data/csv/solar/train/train.csv',index_col=None, header=0)
print(train .shape)     #(52560, 9)
print(train .tail())
submission = pd.read_csv('C:/data/csv/solar/sample_submission.csv')


df_train = preprocess_data(train)
#Standard scaler 사용
scale = StandardScaler()
scale.fit(df_train.iloc[:,:-2])
df_train.iloc[:,:-2] = scale.transform(df_train.iloc[:,:-2])
train = split_to_seq(df_train)


########test데이터 준비
x_test = []

for i in range(81):
    file_path = 'C:/data/csv/solar/test2/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp,is_train=False)
    temp = scale.transform(temp)
    temp = pd.DataFrame(temp)
    #똑같이 test데이터도 7일씩 시간순으로 되어 있는 것을 1일의 0시0분~7일의 0시 0분 까지로 한묶음으로 처리해서 48시간으로 분리
    temp = split_to_seq(temp)
    #print(temp.shape)
    x_test.append(temp)

test = np.array(x_test) #(81, 48, 7, 9)

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

es = EarlyStopping(monitor = 'val_loss', patience = 20)
lr = ReduceLROnPlateau(monitor = 'val_loss', patience = 10, factor = 0.2, verbose = 1)
epochs = 10000
bs = 32

for i in range(48):
    x_train, x_val, y1_train, y1_val, y2_train, y2_val = train_test_split(x[i],y1[i],y2[i], train_size = 0.7,shuffle = True, random_state = 0)
    if i%2 == 0:
      minute = 0
    elif i%2 == 1:
      minute = 30
    hour = int(i/2) 
    
    for j in quantiles:
        
        print("##############내일 {}시,{}분, q_0.{} 훈련 시작!!###########".format(hour,minute,j))
        model = Conv1dml()
        filepath_cp = f'C:/data/modelcheckpoint/solar_checkpoint_0126_1{i:2d}_day1_{j:.1f}.hdf5'
        cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        model.fit(x_train,y1_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y1_val),callbacks = [es,cp,lr])
        
    
    for j in quantiles:
        
        print("##############모레 {}시,{}분 q_0.{} 훈련 시작!!############".format(hour,minute,j))
        model = Conv1dml()
        filepath_cp = f'C:/data/modelcheckpoint/solar_checkpoint_0126_1{i:2d}_day2_{j:.1f}.hdf5'
        cp = ModelCheckpoint(filepath_cp,save_best_only=True,monitor = 'val_loss')
        model.compile(loss = lambda y_true,y_pred: quantile_loss(j,y_true,y_pred), optimizer = 'adam', metrics = [lambda y,y_pred: quantile_loss(j,y,y_pred)])
        model.fit(x_train,y2_train,epochs = epochs, batch_size = bs, validation_data = (x_val,y2_val),callbacks = [es,cp,lr]) 
        
