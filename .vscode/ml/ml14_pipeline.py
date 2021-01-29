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

#2. model
#model = Pipeline([("scaler",MinMaxScaler()),('malddong',SVC())]) #svc 란 모델과 민맥스란 전처리 하나하나 합친것
#model = make_pipeline(MinMaxScaler(),SVC())
model = make_pipeline(StandardScaler(),SVC())

model.fit(x_train,y_train)
results = model.score(x_test,y_test)
print(results)
#MinMaxScaler()
#1.0
#StandardScaler()
#0.9333333333333333