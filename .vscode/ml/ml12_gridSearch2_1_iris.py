# RandomForestClassifier

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt
import datetime
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

dataset = pd.read_csv('C:/data/csv/iris_sklearn.csv',index_col=None,header=0)

x= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

date_now_before = datetime.datetime.now()
date_time_before = date_now_before.strftime('%H%M%S')
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
kfold = KFold(n_splits=5, shuffle=True) 
parameters = [
    {'n_estimators' : [100,200],'max_depth' : [6,8,10,12],'min_samples_leaf': [3,5,7,10],'min_samples_split': [2,3,5,10]},
    {'max_depth' : [6,8,10,12],'min_samples_leaf': [3,5,7,10],'min_samples_split': [2,3,5,10],'n_jobs': [-1,2,4]},
    {'min_samples_leaf': [3,5,7,10],'min_samples_split': [2,3,5,10],'n_jobs': [-1,2,4]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs': [-1,2,4]}
]
#2.model
model =RandomizedSearchCV(RandomForestClassifier(),parameters,cv=kfold)  #SVC모델을 그리드써치로 싸버리겠다는 

#3.compile,trainning
model.fit(x_train,y_train)

#4.evaluate,predict
print("최적의 매개변수:",model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률:',accuracy_score(y_test,y_pred))

#최적의 매개변수: RandomForestClassifier(max_depth=6, min_samples_leaf=10, min_samples_split=10,
#                       n_jobs=-1)
#최종정답률: 0.9666666666666667

date_now_after = datetime.datetime.now()
date_time_after = date_now_after.strftime('%H%M%S')
Duration_of_time = int(date_time_after) - int(date_time_before)
print( "소요시간 :",Duration_of_time,"seconds")

#최적의 매개변수: RandomForestClassifier(max_depth=8, min_samples_leaf=7, min_samples_split=5)
#최종정답률: 1.0