from sklearn.utils.testing import all_estimators
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris  ,load_breast_cancer,load_wine,load_boston,load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

dataset = load_diabetes()
x = dataset.data
y= dataset.target
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

scalar = StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)
kfold = KFold(n_splits=5,shuffle=True)

allAlgorithms = all_estimators(type_filter='regressor')

for (name, algorithms) in allAlgorithms:
    try:
        model = algorithms()
        score = cross_val_score(model,x_train,y_train,cv=kfold)

        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        print(name,'의 정답률 : ', score)
    except:
        #continue
        print(name, '은 없는 놈!')

# ARDRegression 의 정답률 :  0.4987483503692143
# AdaBoostRegressor 의 정답률 :  0.36797624994694866
# BaggingRegressor 의 정답률 :  0.2710704123377842
# BayesianRidge 의 정답률 :  0.5008218932350129
# CCA 의 정답률 :  0.48696409064967594
# DecisionTreeRegressor 의 정답률 :  -0.20461499766279756
# DummyRegressor 의 정답률 :  -0.00015425885559339214
# ElasticNet 의 정답률 :  0.008101269711286885
# ElasticNetCV 의 정답률 :  0.43071557917754755
# ExtraTreeRegressor 의 정답률 :  -0.17596608751116816
# ExtraTreesRegressor 의 정답률 :  0.3986290286399231
# GammaRegressor 의 정답률 :  0.005812599388535289
# GaussianProcessRegressor 의 정답률 :  -5.636096407912189
# GeneralizedLinearRegressor 의 정답률 :  0.005855247171688949
# GradientBoostingRegressor 의 정답률 :  0.3884127026380264
# HistGradientBoostingRegressor 의 정답률 :  0.28899497703380905
# HuberRegressor 의 정답률 :  0.5033459728718326
# IsotonicRegression 은 없는 놈!
# KNeighborsRegressor 의 정답률 :  0.3968391279034368
# KernelRidge 의 정답률 :  -3.3847644323549924
# Lars 의 정답률 :  0.49198665214641635
# LarsCV 의 정답률 :  0.5010892359535759
# Lasso 의 정답률 :  0.3431557382027084
# LassoCV 의 정답률 :  0.49757816595208426
# LassoLars 의 정답률 :  0.36543887418957965
# LassoLarsCV 의 정답률 :  0.495194279067827
# LassoLarsIC 의 정답률 :  0.4994051517531072
# LinearRegression 의 정답률 :  0.5063891053505036
# LinearSVR 의 정답률 :  -0.33470258280275034
# MLPRegressor 의 정답률 :  -2.799710348720772
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 은 없는 놈!
# MultiTaskElasticNetCV 은 없는 놈!
# MultiTaskLasso 은 없는 놈!
# MultiTaskLassoCV 은 없는 놈!
# NuSVR 의 정답률 :  0.14471275169122277
# OrthogonalMatchingPursuit 의 정답률 :  0.3293449115305741
# OrthogonalMatchingPursuitCV 의 정답률 :  0.44354253337919747
# PLSCanonical 의 정답률 :  -0.975079227792292
# PLSRegression 의 정답률 :  0.4766139460349792
# PassiveAggressiveRegressor 의 정답률 :  0.4493811350774273
# PoissonRegressor 의 정답률 :  0.32989738735884344
# RANSACRegressor 의 정답률 :  0.1503945880051919
# RadiusNeighborsRegressor 의 정답률 :  -0.00015425885559339214
# RandomForestRegressor 의 정답률 :  0.35985508655430587
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :  0.40936668956159705
# RidgeCV 의 정답률 :  0.49525463889305044
# SGDRegressor 의 정답률 :  0.3933157471542975
# SVR 의 정답률 :  0.14331518075345895
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :  0.5176398294434736
# TransformedTargetRegressor 의 정답률 :  0.5063891053505036
# TweedieRegressor 의 정답률 :  0.005855247171688949
# VotingRegressor 은 없는 놈!
# _SigmoidCalibration 은 없는 놈!

# # TheilSenRegressor 의 정답률 :  0.5176398294434736 가장높음


# ARDRegression 의 정답률 :  [0.52259729 0.39476157 0.61802916 0.5024108  0.40131635]
# AdaBoostRegressor 의 정답률 :  [0.46040631 0.56844882 0.4233991  0.5149664  0.37379524]
# BaggingRegressor 의 정답률 :  [0.30538629 0.44075317 0.48085204 0.28752205 0.45802549]
# BayesianRidge 의 정답률 :  [0.52588756 0.5141248  0.50113468 0.35321753 0.48758982]
# CCA 의 정답률 :  [ 0.50213084  0.48619861  0.56913319  0.53696863 -0.20563114]
# DecisionTreeRegressor 의 정답률 :  [ 0.15585289 -0.22674221 -0.13687204 -0.08091356  0.00561268]
# DummyRegressor 의 정답률 :  [-0.01331601 -0.0034444  -0.01360788 -0.00470655 -0.05244349]
# ElasticNet 의 정답률 :  [0.39291839 0.5143684  0.45225337 0.50416976 0.47100634]
# ElasticNetCV 의 정답률 :  [0.39904171 0.46231469 0.54984845 0.49347429 0.51227458]
# ExtraTreeRegressor 의 정답률 :  [-0.30802288 -0.19283546  0.19185758 -0.16787453 -0.00603025]
# ExtraTreesRegressor 의 정답률 :  [0.57630502 0.40366355 0.36617579 0.38266972 0.50895451]
# GammaRegressor 의 정답률 :  [0.40153861 0.37571053 0.3332353  0.38874418 0.39597578]
# GaussianProcessRegressor 의 정답률 :  [-0.74194931 -0.57760696 -1.04895839 -0.80247045 -0.63264805]
# GeneralizedLinearRegressor 의 정답률 :  [0.46265585 0.4410229  0.32449549 0.42550879 0.48326912]
# GradientBoostingRegressor 의 정답률 :  [0.35684638 0.43790471 0.45051079 0.46945294 0.36980183]
# HistGradientBoostingRegressor 의 정답률 :  [0.28671595 0.52511122 0.38722756 0.16132228 0.54270158]
# HuberRegressor 의 정답률 :  [0.50708532 0.57435234 0.32204653 0.51271122 0.47964716]
# IsotonicRegression 은 없는 놈!
# KNeighborsRegressor 의 정답률 :  [0.52908011 0.3080537  0.45757902 0.40336458 0.3621261 ]
# KernelRidge 의 정답률 :  [-4.2962644  -3.48540749 -4.06842188 -3.74824339 -3.42787445]
# Lars 의 정답률 :  [ 0.51272758 -0.67863236  0.36076794  0.45399922  0.70400241]
# LarsCV 의 정답률 :  [0.53438213 0.50957206 0.40062181 0.3985594  0.49778348]
# Lasso 의 정답률 :  [0.49664669 0.44485722 0.49174664 0.50428304 0.51080712]
# LassoCV 의 정답률 :  [0.43401706 0.47441599 0.50054912 0.54143536 0.46653152]
# LassoLars 의 정답률 :  [0.42630415 0.4033287  0.43345182 0.36108631 0.35365182]
# LassoLarsCV 의 정답률 :  [0.31998399 0.43064724 0.59153509 0.52204053 0.50679252]
# LassoLarsIC 의 정답률 :  [0.39093134 0.43619531 0.50958789 0.58245674 0.51996893]
# LinearRegression 의 정답률 :  [0.5355053  0.29602787 0.48785613 0.42969223 0.57527634]
# LinearSVR 의 정답률 :  [0.24823065 0.33445991 0.23291051 0.18080249 0.03306757]
# MLPRegressor 의 정답률 :  [-0.65899242 -1.31322675 -1.58901704 -0.96826381 -1.00733652]
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 은 없는 놈!
# MultiTaskElasticNetCV 은 없는 놈!
# MultiTaskLasso 은 없는 놈!
# MultiTaskLassoCV 은 없는 놈!
# NuSVR 의 정답률 :  [0.10700057 0.16304514 0.06492039 0.14900331 0.1073332 ]
# OrthogonalMatchingPursuit 의 정답률 :  [0.09346763 0.28842935 0.27431738 0.37322222 0.22723416]
# OrthogonalMatchingPursuitCV 의 정답률 :  [0.3876118  0.41348553 0.58291873 0.39287373 0.37461769]
# PLSCanonical 의 정답률 :  [-1.39090866 -0.78426453 -2.06842682 -1.51048585 -0.77923831]
# PLSRegression 의 정답률 :  [0.48216076 0.49810701 0.62105844 0.39848222 0.40135464]
# PassiveAggressiveRegressor 의 정답률 :  [0.43723839 0.49671713 0.3398751  0.47569242 0.44843275]
# PoissonRegressor 의 정답률 :  [0.54618582 0.47474141 0.35639064 0.54032173 0.41026741]
# RANSACRegressor 의 정답률 :  [ 0.32739236  0.26332155 -0.08823333  0.28220376  0.42473383]
# RadiusNeighborsRegressor 은 없는 놈!
# RandomForestRegressor 의 정답률 :  [0.47685433 0.5029674  0.51636155 0.33688511 0.57397832]
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :  [0.55950325 0.25077324 0.57538524 0.45072309 0.48732258]
# RidgeCV 의 정답률 :  [0.39398498 0.60486969 0.48798754 0.41251172 0.51277307]
# SGDRegressor 의 정답률 :  [0.39131855 0.5309712  0.56291656 0.44646462 0.43217072]
# SVR 의 정답률 :  [0.15286586 0.18940379 0.14812088 0.06903969 0.11497652]
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :  [0.54682189 0.35661671 0.37918116 0.61847101 0.41809359]
# TransformedTargetRegressor 의 정답률 :  [0.35413494 0.45591528 0.50027404 0.45438739 0.58987234]
# TweedieRegressor 의 정답률 :  [0.4892711  0.48491872 0.40381702 0.43526298 0.34927749]
# VotingRegressor 은 없는 놈!


###########cross_val_score 적용한게 더 잘나옴!