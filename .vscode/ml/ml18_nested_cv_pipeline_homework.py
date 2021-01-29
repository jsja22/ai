import numpy as np
from sklearn.datasets import load_diabetes

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # 둘 중에 하나 사용
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# 모델마다 나오는 결과 값을 비교한다.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')
import pandas as pd 

###########################################################

#1. DATA
dataset = load_diabetes()
x = dataset.data 
y = dataset.target 
# print(x.shape, y.shape)

# preprocessing >>  K-Fold 
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
kfold = KFold(n_splits=5, shuffle=True)

#2. Modeling
# pipline : 파라미터튜닝에 전처리까지 합친다. >> 전처리와 모델을 합친다.

# # [1] Pipeline
parameters=[
    {'model__n_estimators' : [100, 200, 300], 'model__max_depth' : [6, 8, 10, 12]},
    {'model__max_depth' : [6, 8, 10, 12], 'model__min_samples_leaf' : [3, 7, 10]},
    {'model__min_samples_split' : [2, 3, 5, 9], 'model__n_jobs' : [-1, 2, 4]}
]

# # [2] make_pipeline
# parameters=[
#     {'randomforestregressor__n_estimators' : [100, 200], 'randomforestregressor__max_depth' : [6, 8, 10, 12]},
#     {'randomforestregressor__max_depth' : [6, 8, 10, 12], 'randomforestregressor__min_samples_leaf' : [3, 7, 10]},
#     {'randomforestregressor__min_samples_split' : [2, 3, 5, 9], 'randomforestregressor__n_jobs' : [-1, 2, 4]}
# ]

for train_index, test_index in kfold.split(x) :
    
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]


    pipe = Pipeline([("scaler", MinMaxScaler()), ('model', RandomForestRegressor())])
    model = RandomizedSearchCV(pipe, parameters, cv=kfold)
    score = cross_val_score(model, x_train, y_train, cv=kfold)

    print('교차검증점수 : ', score)
#교차검증점수 :  [0.49343478 0.39451586 0.41655    0.27115781 0.48780861]#
#교차검증점수 :  [0.38140229 0.50903437 0.54251937 0.36559658 0.55947704]
#교차검증점수 :  [0.41007952 0.48489965 0.42226497 0.46489715 0.40627864]
#교차검증점수 :  [0.40554586 0.41556927 0.4889002  0.45039208 0.51302813]
#교차검증점수 :  [0.40434898 0.4776804  0.32088912 0.45260302 0.3504861 ]