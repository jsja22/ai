from sklearn.utils.testing import all_estimators
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris,load_boston   #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()

x = dataset.data
y= dataset.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 : ', r2_score(y_test,y_pred))
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

#QuadraticDiscriminantAnalysis 의 정답률 :  1.0
#LogisticRegressionCV 의 정답률 :  1.0
#MLPClassifier 의 정답률 :  1.0
#->정답률 1인녀석들

# #ARDRegression 의 정답률 :  0.8012569266997974
# AdaBoostRegressor 의 정답률 :  0.9033493312537739
# BaggingRegressor 의 정답률 :  0.895950336896333
# BayesianRidge 의 정답률 :  0.7937918622384766
# CCA 의 정답률 :  0.7913477184424629
# DecisionTreeRegressor 의 정답률 :  0.8039284593569378
# DummyRegressor 의 정답률 :  -0.0005370164400797517
# ElasticNet 의 정답률 :  0.7338335519267194
# ElasticNetCV 의 정답률 :  0.7167760356856181
# ExtraTreeRegressor 의 정답률 :  0.6891088431688024
# ExtraTreesRegressor 의 정답률 :  0.9370712650486971
# GammaRegressor 의 정답률 :  -0.0005370164400797517
# GaussianProcessRegressor 의 정답률 :  -6.073105259620457
# GeneralizedLinearRegressor 의 정답률 :  0.7461438417572277
# GradientBoostingRegressor 의 정답률 :  0.9451901842989139
# HistGradientBoostingRegressor 의 정답률 :  0.9323597806119726
# HuberRegressor 의 정답률 :  0.7472151075110175
# IsotonicRegression 은 없는 놈!
# KNeighborsRegressor 의 정답률 :  0.5900872726222293
# KernelRidge 의 정답률 :  0.8333325493719488
# Lars 의 정답률 :  0.7746736096721595
# LarsCV 의 정답률 :  0.7981576314184016
# Lasso 의 정답률 :  0.7240751024070102
# LassoCV 의 정답률 :  0.7517507753137198
# LassoLars 의 정답률 :  -0.0005370164400797517
# LassoLarsCV 의 정답률 :  0.8127604328474287
# LassoLarsIC 의 정답률 :  0.8131423868817642
# LinearRegression 의 정답률 :  0.8111288663608656
# LinearSVR 의 정답률 :  0.701029257629088
# MLPRegressor 의 정답률 :  0.5570630481251425
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 은 없는 놈!
# MultiTaskElasticNetCV 은 없는 놈!
# MultiTaskLasso 은 없는 놈!
# MultiTaskLassoCV 은 없는 놈!
# NuSVR 의 정답률 :  0.2594558622083819
# OrthogonalMatchingPursuit 의 정답률 :  0.5827617571381449
# OrthogonalMatchingPursuitCV 의 정답률 :  0.78617447738729
# PLSCanonical 의 정답률 :  -2.2317079741425756
# PLSRegression 의 정답률 :  0.8027313142007887
# PassiveAggressiveRegressor 의 정답률 :  0.03300363143512641
# PoissonRegressor 의 정답률 :  0.8575650836250985
# RANSACRegressor 의 정답률 :  0.4998281609487739
# RadiusNeighborsRegressor 은 없는 놈!
# RandomForestRegressor 의 정답률 :  0.9166862360890715
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :  0.8098487632912241
# RidgeCV 의 정답률 :  0.8112529186351158
# SGDRegressor 의 정답률 :  -2.1203999397959445e+26
# SVR 의 정답률 :  0.2347467755572229
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :  0.78620026338839
# TransformedTargetRegressor 의 정답률 :  0.8111288663608656
# TweedieRegressor 의 정답률 :  0.7461438417572277
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 은 없는 놈!


# ExtraTreesRegressor 의 정답률 :  0.9370712650486971 가장높음
#tensorflow keras모델과 비교해보자
#dense model
#전처리 후 (x=x/711.)
#loss: 11.227853775024414
#mae: 2.465669870376587
#RMSE :  3.350798737628819
#R2 : 0.8656681310980472

#lstm
#r2->0.89

