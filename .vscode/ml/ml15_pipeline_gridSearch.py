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
from sklearn.pipeline import Pipeline, make_pipeline
import matplotlib.pyplot as plt
import datetime
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

dataset = load_iris()

x = dataset.data
y= dataset.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)
# scalar = MinMaxScaler()
# scalar.fit(x_train)
# x_train=scalar.transform(x_train)
# x_test = scalar.transform(x_test)
parameters = [
    {"mal__C":[1,10,100,1000],"mal__kernel":["linear"]},
    {"mal__C":[1,10,100],"mal__kernel":["rbf"],"mal__gamma":[0.001,0.0001]},
    {"mal__C":[1,10,100,1000],"mal__kernel":["sigmoid"],"mal__gamma":[0.001,0.0001]}
] #4+6+8 총 18번 돈다 cross_val 총 5번이기때문에 18x5 90번돈다

#2. model
pipe = Pipeline([("scaler",MinMaxScaler()),('mal',SVC())]) #svc 란 모델과 민맥스란 전처리 하나하나 합친것
#model = make_pipeline(MinMaxScaler(),SVC())
#pipe = make_pipeline(StandardScaler(),SVC())

model = GridSearchCV(pipe,parameters, cv=5)
model.fit(x_train,y_train)
results = model.score(x_test,y_test)
print(results)
#MinMaxScaler()
#1.0
#StandardScaler()
#0.9333333333333333

#parameters ('mal')
#1.0

#Minmax하고 Gs나 Rs 하면 train,val 전체가 전처리가 되버린다. 통상적으로 train만 전처리한게 성능이 좋음으로 그것을 해결하고자 pipeline이 나왔다. 
#pipeline은 cv수만큼 전처리를 해주기때문에 과적합 문제를 해결할 수 있다