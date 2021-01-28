from sklearn.utils.testing import all_estimators
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris  ,load_breast_cancer,load_wine,load_boston
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

dataset = load_boston()
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


# ARDRegression 의 정답률 :  [0.65490329 0.73876112 0.66273679 0.76567735 0.69473176]
# AdaBoostRegressor 의 정답률 :  [0.55837642 0.78979597 0.54214441 0.88858732 0.89129101]
# BaggingRegressor 의 정답률 :  [0.84262597 0.77404342 0.87649973 0.80919156 0.60378749]
# BayesianRidge 의 정답률 :  [0.77129985 0.7651884  0.72002104 0.60277378 0.59684132]
# CCA 의 정답률 :  [0.2994297  0.70475657 0.64259898 0.75127542 0.65528115]
# DecisionTreeRegressor 의 정답률 :  [0.63205179 0.73845454 0.75625328 0.73425722 0.71695151]
# DummyRegressor 의 정답률 :  [-1.83728203e-03 -3.41863196e-02 -4.63104302e-06 -2.02272254e-04
#  -3.10248746e-02]
# ElasticNet 의 정답률 :  [0.64940096 0.55853482 0.54355909 0.69935303 0.54265972]
# ElasticNetCV 의 정답률 :  [0.693441   0.64008941 0.64145694 0.78809215 0.69303325]
# ExtraTreeRegressor 의 정답률 :  [0.60101469 0.75245998 0.65131632 0.66665169 0.68108191]
# ExtraTreesRegressor 의 정답률 :  [0.90469057 0.92538923 0.87675765 0.862137   0.77813966]
# GammaRegressor 의 정답률 :  [0.65847596 0.62566756 0.6572704  0.52940989 0.67378003]
# GaussianProcessRegressor 의 정답률 :  [-0.11423477  0.40436508  0.13815734  0.37761152  0.29941773]
# GeneralizedLinearRegressor 의 정답률 :  [0.6375679  0.61207402 0.48025667 0.61425336 0.67179943]
# GradientBoostingRegressor 의 정답률 :  [0.91042673 0.80936824 0.75241029 0.84622439 0.86785035]
# HistGradientBoostingRegressor 의 정답률 :  [0.87787975 0.75847233 0.77778498 0.83458179 0.88380036]
# HuberRegressor 의 정답률 :  [0.64302033 0.79861894 0.79671405 0.58452869 0.61066835]
# IsotonicRegression 은 없는 놈!
# KNeighborsRegressor 의 정답률 :  [0.74431285 0.76000725 0.6695045  0.74037142 0.59492053]
# KernelRidge 의 정답률 :  [-6.67315512 -4.5885377  -4.87295127 -6.2333707  -8.11639834]
# Lars 의 정답률 :  [0.7165282  0.74697765 0.58873815 0.67409254 0.66364685]
# LarsCV 의 정답률 :  [0.77663597 0.46486369 0.7648548  0.79936289 0.54113995]
# Lasso 의 정답률 :  [0.63223561 0.60062095 0.53395324 0.69117031 0.61621842]
# LassoCV 의 정답률 :  [0.76749225 0.77601375 0.64744843 0.66246529 0.67681367]
# LassoLars 의 정답률 :  [-0.00910919 -0.00171382 -0.00622047 -0.01150655 -0.00182027]
# LassoLarsCV 의 정답률 :  [0.75717964 0.67674047 0.51534182 0.66686067 0.72998908]
# LassoLarsIC 의 정답률 :  [0.75662255 0.72979258 0.52425963 0.73701587 0.66746481]
# LinearRegression 의 정답률 :  [0.67286069 0.63466648 0.78537081 0.72093122 0.63543078]
# LinearSVR 의 정답률 :  [0.62948867 0.74484542 0.52424413 0.74876316 0.57183924]
# MLPRegressor 의 정답률 :  [0.39172797 0.67020084 0.65182383 0.55229977 0.70491452]
# MultiOutputRegressor 은 없는 놈!
# MultiTaskElasticNet 은 없는 놈!
# MultiTaskElasticNetCV 은 없는 놈!
# MultiTaskLasso 은 없는 놈!
# MultiTaskLassoCV 은 없는 놈!
# NuSVR 의 정답률 :  [0.52285429 0.64608146 0.64819664 0.51693102 0.60611609]
# OrthogonalMatchingPursuit 의 정답률 :  [0.55588518 0.3887034  0.43962461 0.58692116 0.55840625]
# OrthogonalMatchingPursuitCV 의 정답률 :  [0.51854316 0.58563255 0.46925406 0.65626685 0.79651158]
# PLSCanonical 의 정답률 :  [-0.59184026 -1.95885468 -2.47625373 -2.75801899 -4.24203883]
# PLSRegression 의 정답률 :  [0.48685775 0.79289934 0.64227956 0.535073   0.77144414]
# PassiveAggressiveRegressor 의 정답률 :  [0.4465379  0.59954627 0.57046079 0.21975886 0.06479956]
# PoissonRegressor 의 정답률 :  [0.69529416 0.7789077  0.70596821 0.80226081 0.77729359]
# RANSACRegressor 의 정답률 :  [0.3447118  0.15026028 0.48800173 0.4209576  0.3313222 ]
# RadiusNeighborsRegressor 은 없는 놈!
# RandomForestRegressor 의 정답률 :  [0.87538637 0.77421257 0.84693553 0.80931677 0.91505666]
# RegressorChain 은 없는 놈!
# Ridge 의 정답률 :  [0.68279915 0.76457513 0.61970561 0.73010496 0.66121995]
# RidgeCV 의 정답률 :  [0.67437585 0.7299511  0.51264695 0.68531403 0.79454393]
# SGDRegressor 의 정답률 :  [0.73577106 0.67400043 0.61308546 0.78561214 0.55150843]
# SVR 의 정답률 :  [0.56474106 0.66184902 0.59686458 0.57596892 0.55439769]
# StackingRegressor 은 없는 놈!
# TheilSenRegressor 의 정답률 :  [0.53189052 0.43891085 0.47304227 0.69599909 0.56069799]
# TransformedTargetRegressor 의 정답률 :  [0.72659081 0.65080714 0.59970952 0.66183502 0.78140944]
# TweedieRegressor 의 정답률 :  [0.73766837 0.5939869  0.53173102 0.51544878 0.64094115]
# VotingRegressor 은 없는 놈!
###########cross_val_score 적용안한게 더 잘나옴!