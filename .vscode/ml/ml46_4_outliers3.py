'''
import pandas as pd
from pandas import DataFrame
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
import math
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

from sklearn.preprocessing import RobustScaler, StandardScaler

data = train.iloc[:,:-1]
scaler =  RobustScaler()
scaler.fit(data)
scaler_data = scaler.transform(data)
scaler_data = pd.DataFrame(scaler_data).describe()
scaler_data.columns = data.columns
scaler_data = round(scaler_data)
print(scaler_data)


####################RobustScaler##############################
#            Day     Hour   Minute      DHI      DNI       WS       RH        T
# count  52560.0  52560.0  52560.0  52560.0  52560.0  52560.0  52560.0  52560.0
# mean       0.0      0.0      0.0      1.0      1.0      0.0     -0.0      0.0
# std        1.0      1.0      1.0      1.0      1.0      1.0      1.0      1.0
# min       -1.0     -1.0     -0.0      0.0      0.0     -1.0     -2.0     -2.0
# 25%       -0.0     -0.0     -0.0      0.0      0.0     -0.0     -1.0     -0.0
# 50%        0.0      0.0      0.0      0.0      0.0      0.0      0.0      0.0
# 75%        0.0      0.0      0.0      1.0      1.0      1.0      0.0      0.0
# max        1.0      1.0      0.0      6.0      2.0      5.0      1.0      2.0

####################StandardScaler##############################
#            Day     Hour   Minute      DHI      DNI       WS       RH        T
# count  52560.0  52560.0  52560.0  52560.0  52560.0  52560.0  52560.0  52560.0
# mean      -0.0     -0.0      0.0      0.0      0.0     -0.0      0.0     -0.0
# std        1.0      1.0      1.0      1.0      1.0      1.0      1.0      1.0
# min       -2.0     -2.0     -1.0     -1.0     -1.0     -2.0     -2.0     -3.0
# 25%       -1.0     -1.0     -1.0     -1.0     -1.0     -1.0     -1.0     -1.0
# 50%        0.0      0.0      0.0     -1.0     -1.0     -0.0      0.0     -0.0
# 75%        1.0      1.0      1.0      0.0      1.0      1.0      1.0      1.0
# max        2.0      2.0      1.0      4.0      2.0      7.0      2.0      3.0


#Isolation Forest 적용
#####Isolation Forest는 Regression Tree 기반으로 데이터를 계속 Split하여 데이터 관측치를 고립시키는 방법

# 다차원의 데이터셋에서 효율적으로 작동하는 이상치 제거 방법
# 의사결정트리 기반의 이상치 탐지 기법
# 의사결정트리의 샘플 분포를 여러개 뽑아 앙상블 모델로 변환하여 안정적인 이상치 탐지 및 제거 가능
# 군집기반 이상탐지 알고리즘에 비해 월등한 실행 성능
# 전체 데이터가 아닌 데이터를 샘플링해서 사용하기 때문에 ‘Swamping’과 ‘Masking’ 문제를 예방할 수 있다.
# ‘Swamping’ : 정상치가 이상치에 가까운 경우 정상치를 이상치로 잘못 분류하는 현상

# ‘Masking’ : 이상치가 군집화되어 있으면 정상치로 잘못 분류하는 현상

from sklearn.ensemble import IsolationForest
import collections
# n_estimators : 노드 수 (50 - 100사이의 숫자면 적당하다.)
# max_samples : 샘플링 수
# contamination : 이상치 비율
# max_features : 사용하고자 하는 독립변수 수 (1이면 전부 사용)
# random_state : seed를 일정하게 유지시켜줌(if None, the random number generator is the RandomState instance used by np.random)
# n_jobs : CPU 병렬처리 유뮤(1이면 안하는 것으로 디버깅에 유리. -1을 넣으면 multilple CPU를 사용하게 되어 메모리 사용량이 급격히 늘어날 수 있다.)

model = IsolationForest(n_estimators=100,
                      max_samples='auto',
                      contamination=0.1,
                      max_features=1,
                      bootstrap=False,
                      n_jobs=1,
                      random_state=33,
                      verbose=0)

model.fit(data)
y_pred_outliers = model.predict(data)
print(y_pred_outliers)
print(y_pred_outliers.shape) #(52560,)
print(collections.Counter(y_pred_outliers)) 
#Counter({1: 47304, -1: 5256})

data['out']=y_pred_outliers
outliers=data.loc[data['out']== -1]
outlier_index=list(outliers.index)
print(outlier_index)
print('value count : ',data['out'].value_counts())  # 이상치 갯수 
#PCA로 Isolation Forest 결과 확인

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)
scaler_reduce = pca.fit_transform(scaler_data)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.set_zlabel("x_composite_3")
ax.scatter(scaler_reduce[:, 0], scaler_reduce[:, 1], zs=scaler_reduce[:, 2], s=4, lw=1, label="inliers",c="green")
# Plot x's for the ground truth outliers
ax.scatter(scaler_reduce[outlier_index,0],scaler_reduce[outlier_index,1], scaler_reduce[outlier_index,2],
           lw=2, s=60, marker="x", c="red", label="outliers")
ax.legend()
plt.show()


'''

