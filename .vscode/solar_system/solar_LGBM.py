import pandas as pd
import numpy as np
import os
import glob
import sys
from tqdm import tqdm
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

train = pd.read_csv("C:/data/csv/solar/train/train.csv")
# train.tail()
print(train.shape )
# train.shape #(52560, 9)

submission = pd.read_csv("C:/data/csv/solar/sample_submission.csv")
# submission.tail()
print(submission.shape )
# submission.shape #(7776, 10)

def preprocess_data(data, is_train=True):
    
    temp = data.copy()
    temp = temp[['Hour', 'Minute', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    
    temp['Time_sin'] = np.sin(( 2*np.pi* (2*temp['Hour']+temp['Minute'].apply(lambda x: bool(x))- 12) / 48 ))
    temp = temp.drop(['Minute'], axis = 1)
    
    if is_train==True:          
    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')
        temp['Target2'] = temp['TARGET'].shift(-96).fillna(method='ffill')
        temp = temp.dropna()
        
        temp_t1 = temp['Target1']
        temp = temp.drop(['Target1'], axis = 1)
        temp['Target1'] = temp_t1
        temp_t2 = temp['Target2']
        temp = temp.drop(['Target2'], axis = 1)
        temp['Target2'] = temp_t1
        
        return temp.iloc[:-96]
    
    elif is_train==False:        
        
        return temp.iloc[-48:, :]


df_train = preprocess_data(train)
print(df_train.columns) #Index(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Time_sin', 'Target1', 'Target2']
print(df_train.shape) #(52464, 10)

df_test = [] # 리스트

for i in range(81):
    file_path = 'C:/data/csv/solar/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path) # 데이터 프레임 shape = (336, 9)
    temp = preprocess_data(temp, is_train=False) 
    df_test.append(temp) # 리스트에 전처리된 데이터 프레임 append

X_test = pd.concat(df_test) # 전처리된 데이터 프레임들 세로 병합
print(X_test.shape)
# X_test.shape # (3888, 8)
print(X_test.columns) #(['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T', 'Time_sin']
print(X_test.duplicated().sum()) ### = 0

#train, test 분리
from sklearn.model_selection import train_test_split 
X_train_1, X_valid_1, Y_train_1, Y_valid_1 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -2], test_size=0.3, random_state=0)
X_train_2, X_valid_2, Y_train_2, Y_valid_2 = train_test_split(df_train.iloc[:, :-2], df_train.iloc[:, -1], test_size=0.3, random_state=0)

print(X_train_1.shape)
print(X_valid_1.shape)
print(Y_train_1.shape)
print(Y_valid_1.shape)
# X_train_1.shape # (36724, 8)
# X_valid_1.shape # (15740, 8)
# Y_train_1.shape # (36724,)
# Y_valid_1.shape # (15740,)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

from lightgbm import LGBMRegressor

def LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test):
    
    # (a) Modeling  
    model = LGBMRegressor(objective='quantile', alpha=q,
                         n_estimators=10000, bagging_fraction=0.7, learning_rate=0.01)                   
                         
                         
    model.fit(X_train, Y_train, eval_metric = ['quantile'], 
          eval_set=[(X_valid, Y_valid)], early_stopping_rounds=700, verbose=1000)

    pred = pd.Series(model.predict(X_test).round(2))
    return pred, model


def train_data(X_train, Y_train, X_valid, Y_valid, X_test):

    LGBM_models=[]
    LGBM_actual_pred = pd.DataFrame()

    for q in quantiles:
        print("##############내일  q_{} 훈련 시작!!".format(q))
        pred , model = LGBM(q, X_train, Y_train, X_valid, Y_valid, X_test)
        # 모델 생성 시마다 모델 저장. 
        LGBM_models.append(model)
        # 예측할 때마다 가로 병합
        LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

    LGBM_actual_pred.columns=quantiles
    print(LGBM_actual_pred.shape)   # (3888, 9)
    return LGBM_models, LGBM_actual_pred

# 내일
models_1, results_1 = train_data(X_train_1, Y_train_1, X_valid_1, Y_valid_1, X_test)
# 모레
models_2, results_2 = train_data(X_train_2, Y_train_2, X_valid_2, Y_valid_2, X_test)
print(results_1.shape, results_2.shape)#(3888, 9) (3888, 9)

submission.loc[submission.id.str.contains("Day7"), "q_0.1":] = results_1.sort_index().values
submission.loc[submission.id.str.contains("Day8"), "q_0.1":] = results_2.sort_index().values
print(submission)
submission.to_csv('C:/data/csv/solar/submission_LGBM.csv', index=False)
