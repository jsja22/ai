from sklearn.utils.testing import all_estimators
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
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

dataset = load_iris()

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

#AdaBoostClassifier 의 정답률 :  0.6333333333333333
#BaggingClassifier 의 정답률 :  0.9333333333333333
#BernoulliNB 의 정답률 :  0.3
#CalibratedClassifierCV 의 정답률 :  0.9
#CategoricalNB 의 정답률 :  0.9
#CheckingClassifier 의 정답률 :  0.3
import sklearn
print(sklearn.__version__) #0.23.2

# AdaBoostClassifier 의 정답률 :  [0.91666667 0.875      0.79166667 0.875      1.        ]
# BaggingClassifier 의 정답률 :  [0.875      1.         0.91666667 0.95833333 0.91666667]
# BernoulliNB 의 정답률 :  [0.25       0.33333333 0.29166667 0.29166667 0.29166667]
# CalibratedClassifierCV 의 정답률 :  [0.95833333 0.79166667 0.91666667 0.91666667 0.91666667]
# CategoricalNB 의 정답률 :  [1.         0.95833333 0.875      0.95833333 0.91666667]
# CheckingClassifier 의 정답률 :  [0. 0. 0. 0. 0.]
# ClassifierChain 은 없는 놈!
# ComplementNB 의 정답률 :  [0.70833333 0.625      0.66666667 0.58333333 0.75      ]
# DecisionTreeClassifier 의 정답률 :  [0.95833333 0.91666667 0.875      0.91666667 0.95833333]
# DummyClassifier 의 정답률 :  [0.25       0.375      0.16666667 0.20833333 0.25      ]
# ExtraTreeClassifier 의 정답률 :  [1.         0.91666667 1.         0.91666667 0.91666667]
# ExtraTreesClassifier 의 정답률 :  [0.91666667 0.91666667 0.95833333 1.         0.95833333]
# GaussianNB 의 정답률 :  [0.95833333 1.         0.91666667 0.875      1.        ]
# GaussianProcessClassifier 의 정답률 :  [0.95833333 1.         0.95833333 0.91666667 0.95833333]
# GradientBoostingClassifier 의 정답률 :  [0.91666667 0.83333333 1.         1.         0.95833333]
# HistGradientBoostingClassifier 의 정답률 :  [0.95833333 1.         0.95833333 0.95833333 0.91666667]
# KNeighborsClassifier 의 정답률 :  [0.875 1.    1.    1.    1.   ]
# LabelPropagation 의 정답률 :  [0.95833333 0.875      0.95833333 1.         0.95833333]
# LabelSpreading 의 정답률 :  [0.95833333 0.95833333 1.         1.         0.95833333]
# LinearDiscriminantAnalysis 의 정답률 :  [0.95833333 0.95833333 1.         0.95833333 1.        ]
# LinearSVC 의 정답률 :  [0.91666667 0.91666667 0.95833333 0.95833333 1.        ]
# LogisticRegression 의 정답률 :  [0.91666667 1.         1.         0.83333333 0.95833333]
# LogisticRegressionCV 의 정답률 :  [0.91666667 0.95833333 1.         0.875      1.        ]
# MLPClassifier 의 정답률 :  [1.         1.         0.91666667 0.95833333 0.95833333]
# MultiOutputClassifier 은 없는 놈!
# MultinomialNB 의 정답률 :  [0.875      0.875      0.91666667 0.79166667 0.875     ]
# NearestCentroid 의 정답률 :  [0.83333333 0.95833333 1.         0.95833333 0.83333333]
# NuSVC 의 정답률 :  [0.95833333 1.         1.         0.875      0.95833333]
# OneVsOneClassifier 은 없는 놈!
# OneVsRestClassifier 은 없는 놈!
# OutputCodeClassifier 은 없는 놈!
# PassiveAggressiveClassifier 의 정답률 :  [0.66666667 0.91666667 0.875      0.79166667 0.75      ]
# Perceptron 의 정답률 :  [0.625      0.75       0.70833333 0.625      0.625     ]
# QuadraticDiscriminantAnalysis 의 정답률 :  [0.95833333 1.         0.95833333 1.         0.91666667]
# RadiusNeighborsClassifier 은 없는 놈!
# RandomForestClassifier 의 정답률 :  [0.95833333 0.95833333 0.91666667 0.875      0.95833333]
# RidgeClassifier 의 정답률 :  [0.95833333 0.95833333 0.75       0.79166667 0.70833333]
# RidgeClassifierCV 의 정답률 :  [0.79166667 0.79166667 0.91666667 0.91666667 0.83333333]
# SGDClassifier 의 정답률 :  [0.875      1.         0.91666667 0.79166667 0.875     ]
# SVC 의 정답률 :  [0.91666667 0.91666667 1.         0.91666667 1.        ]
# StackingClassifier 은 없는 놈!


#cross_val_score 추가한게 더 잘나옴