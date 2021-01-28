# RandomForestClassifier

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris  ,load_breast_cancer,load_wine ,load_boston#다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score,r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
dataset = load_boston()

x = dataset.data
y= dataset.target



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)


kfold = KFold(n_splits=5, shuffle=True) 
parameters = [
    {'n_estimators' : [100,200],'max_depth' : [6,8,10,12],'min_samples_leaf': [3,5,7,10],'min_samples_split': [2,3,5,10]},
    {'max_depth' : [6,8,10,12],'min_samples_leaf': [3,5,7,10],'min_samples_split': [2,3,5,10]},
    {'min_samples_leaf': [3,5,7,10],'min_samples_split': [2,3,5,10]},
    {'min_samples_split': [2,3,5,10]},
    {'n_jobs': [-1,2,4]}
]

#2.model
model =GridSearchCV(RandomForestRegressor(),parameters,cv=kfold)  #SVC모델을 그리드써치로 싸버리겠다는 

#3.compile,trainning
model.fit(x_train,y_train)

#4.evaluate,predict
print("최적의 매개변수:",model.best_estimator_)
y_pred = model.predict(x_test)
print('최종정답률:',r2_score(y_test,y_pred))

#최적의 매개변수: RandomForestRegressor(n_jobs=4)
#최종정답률: 0.9219351516969564

