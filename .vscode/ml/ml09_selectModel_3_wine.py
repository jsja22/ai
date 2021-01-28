from sklearn.utils.testing import all_estimators
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris  ,load_breast_cancer,load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y= dataset.target
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

scalar = StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)

allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 : ', accuracy_score(y_test,y_pred))
    except:
        #continue
        print(name, '은 없는 놈!')


import sklearn
print(sklearn.__version__) #0.23.2


# dll
# AdaBoostClassifier 의 정답률 :  0.8888888888888888
# BaggingClassifier 의 정답률 :  1.0
# BernoulliNB 의 정답률 :  0.9444444444444444
# CalibratedClassifierCV 의 정답률 :  0.9722222222222222
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :  0.3888888888888889
# ClassifierChain 은 없는 놈!
# ComplementNB 은 없는 놈!
# DecisionTreeClassifier 의 정답률 :  0.9166666666666666
# DummyClassifier 의 정답률 :  0.4444444444444444
# ExtraTreeClassifier 의 정답률 :  0.9166666666666666
# ExtraTreesClassifier 의 정답률 :  1.0
# GaussianNB 의 정답률 :  1.0
# GaussianProcessClassifier 의 정답률 :  1.0
# GradientBoostingClassifier 의 정답률 :  0.9722222222222222
# HistGradientBoostingClassifier 의 정답률 :  0.9722222222222222
# KNeighborsClassifier 의 정답률 :  1.0
# LabelPropagation 의 정답률 :  1.0
# LabelSpreading 의 정답률 :  1.0
# LinearDiscriminantAnalysis 의 정답률 :  1.0
# LinearSVC 의 정답률 :  0.9722222222222222
# LogisticRegression 의 정답률 :  0.9722222222222222
# LogisticRegressionCV 의 정답률 :  0.9722222222222222
# MLPClassifier 의 정답률 :  0.9722222222222222
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 은 없는 놈!
# NearestCentroid 의 정답률 :  1.0
# NuSVC 의 정답률 :  1.0
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :  0.9722222222222222
# Perceptron 의 정답률 :  0.9722222222222222
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9722222222222222
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :  1.0
# RidgeClassifier 의 정답률 :  0.9722222222222222
# RidgeClassifierCV 의 정답률 :  0.9722222222222222
# SGDClassifier 의 정답률 :  0.9722222222222222
# SVC 의 정답률 :  1.0
# StackingClassifier 은 없는 놈!
# VotingClassifier 은 없는 놈!

## acc->1 가장높음
#dense 모델 0.002636339981108904 1.0
#lstm
#loss,acc: 0.0026363399811089041.0
#conv1d
#loss :  0.00032411928987130523
#acc :  1.0

#성능은 동일 머신러닝과 딥러닝!
#conv1d 
#cancer, wine, boston, diabets, selectModel 실습 돌려보기