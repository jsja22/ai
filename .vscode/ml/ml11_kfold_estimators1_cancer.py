from sklearn.utils.testing import all_estimators
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris  ,load_breast_cancer #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()

x = dataset.data
y= dataset.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

kfold = KFold(n_splits=5,shuffle=True)
allAlgorithms = all_estimators(type_filter='classifier')

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()

        score = cross_val_score(model,x_train,y_train,cv=kfold)

        model.fit(x_train,y_train)
        # y_pred = model.predict(x_test)
        print(name,'의 정답률 : ',score)
    except:
        #continue
        print(name, '은 없는 놈!')

# AdaBoostClassifier 의 정답률 :  0.9473684210526315
# BaggingClassifier 의 정답률 :  0.9473684210526315
# BernoulliNB 의 정답률 :  0.6403508771929824
# CalibratedClassifierCV 의 정답률 :  0.8859649122807017
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :  0.35964912280701755
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :  0.868421052631579
# DecisionTreeClassifier 의 정답률 :  0.9122807017543859
# DummyClassifier 의 정답률 :  0.4824561403508772
# ExtraTreeClassifier 의 정답률 :  0.9035087719298246
# ExtraTreesClassifier 의 정답률 :  0.9736842105263158
# GaussianNB 의 정답률 :  0.9385964912280702
# GaussianProcessClassifier 의 정답률 :  0.8771929824561403
# GradientBoostingClassifier 의 정답률 :  0.956140350877193
# HistGradientBoostingClassifier 의 정답률 :  0.9736842105263158
# KNeighborsClassifier 의 정답률 :  0.9210526315789473
# LabelPropagation 의 정답률 :  0.3684210526315789
# LabelSpreading 의 정답률 :  0.3684210526315789
# LinearDiscriminantAnalysis 의 정답률 :  0.9473684210526315
# LinearSVC 의 정답률 :  0.8596491228070176
# LogisticRegression 의 정답률 :  0.9385964912280702
# LogisticRegressionCV 의 정답률 :  0.956140350877193
# MLPClassifier 의 정답률 :  0.9385964912280702
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :  0.8596491228070176
# NearestCentroid 의 정답률 :  0.868421052631579
# NuSVC 의 정답률 :  0.8596491228070176
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :  0.8421052631578947
# Perceptron 의 정답률 :  0.8947368421052632
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9385964912280702
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :  0.9649122807017544
# RidgeClassifier 의 정답률 :  0.956140350877193
# RidgeClassifierCV 의 정답률 :  0.9473684210526315
# SGDClassifier 의 정답률 :  0.9122807017543859
# SVC 의 정답률 :  0.8947368421052632



# AdaBoostClassifier 의 정답률 :  [0.95604396 0.97802198 0.94505495 0.95604396 0.98901099]
# BaggingClassifier 의 정답률 :  [0.9010989  0.94505495 0.96703297 0.95604396 0.93406593]
# BernoulliNB 의 정답률 :  [0.61538462 0.56043956 0.62637363 0.68131868 0.63736264]
# CalibratedClassifierCV 의 정답률 :  [0.95604396 0.92307692 0.92307692 0.91208791 0.92307692]
# CategoricalNB 은 없는 놈!
# CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :  [0.95604396 0.91208791 0.84615385 0.91208791 0.87912088]
# DecisionTreeClassifier 의 정답률 :  [0.95604396 0.96703297 0.93406593 0.92307692 0.91208791]
# DummyClassifier 의 정답률 :  [0.46153846 0.51648352 0.52747253 0.49450549 0.58241758]
# ExtraTreeClassifier 의 정답률 :  [0.92307692 0.86813187 0.94505495 0.93406593 0.93406593]
# ExtraTreesClassifier 의 정답률 :  [0.96703297 0.95604396 0.98901099 0.97802198 0.95604396]
# GaussianNB 의 정답률 :  [0.94505495 0.9010989  0.96703297 0.93406593 0.93406593]
# GaussianProcessClassifier 의 정답률 :  [0.89010989 0.95604396 0.93406593 0.97802198 0.92307692]
# GradientBoostingClassifier 의 정답률 :  [0.97802198 0.92307692 0.97802198 0.94505495 0.93406593]
# HistGradientBoostingClassifier 의 정답률 :  [0.97802198 0.89010989 0.96703297 0.97802198 0.97802198]
# KNeighborsClassifier 의 정답률 :  [0.95604396 0.93406593 0.91208791 0.95604396 0.89010989]
# LabelPropagation 의 정답률 :  [0.34065934 0.3956044  0.46153846 0.35164835 0.3956044 ]
# LabelSpreading 의 정답률 :  [0.42857143 0.42857143 0.36263736 0.38461538 0.40659341]
# LinearDiscriminantAnalysis 의 정답률 :  [0.94505495 0.97802198 0.92307692 0.95604396 0.95604396]
# LinearSVC 의 정답률 :  [0.95604396 0.93406593 0.91208791 0.84615385 0.95604396]
# LogisticRegression 의 정답률 :  [0.94505495 0.97802198 0.91208791 0.96703297 0.93406593]
# LogisticRegressionCV 의 정답률 :  [0.95604396 0.98901099 0.95604396 0.93406593 0.93406593]
# MLPClassifier 의 정답률 :  [0.91208791 0.94505495 0.96703297 0.95604396 0.94505495]
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :  [0.89010989 0.92307692 0.94505495 0.9010989  0.85714286]
# NearestCentroid 의 정답률 :  [0.9010989  0.9010989  0.87912088 0.92307692 0.86813187]
# NuSVC 의 정답률 :  [0.85714286 0.83516484 0.92307692 0.87912088 0.93406593]
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :  [0.53846154 0.87912088 0.83516484 0.85714286 0.92307692]
# Perceptron 의 정답률 :  [0.86813187 0.81318681 0.59340659 0.91208791 0.93406593]
# QuadraticDiscriminantAnalysis 의 정답률 :  [0.94505495 0.98901099 0.95604396 0.96703297 0.94505495]
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :  [0.97802198 0.97802198 0.95604396 0.95604396 0.97802198]
# RidgeClassifier 의 정답률 :  [0.96703297 0.95604396 0.94505495 0.96703297 0.92307692]
# RidgeClassifierCV 의 정답률 :  [0.94505495 0.96703297 0.98901099 0.92307692 0.94505495]
# SGDClassifier 의 정답률 :  [0.83516484 0.93406593 0.89010989 0.91208791 0.82417582]
# SVC 의 정답률 :  [0.92307692 0.91208791 0.95604396 0.9010989  0.91208791]


###############cross_val_score 추가한게 더 잘나옴