from sklearn.utils.testing import all_estimators
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris   #다중 분류모델
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

dataset = load_iris()

x = dataset.data
y= dataset.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

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

#tensorflow 모델에서도 acc1이 나왔기에 결과는 동일
#cancer, wine, boston, diabets, selectModel 실습 돌려보기