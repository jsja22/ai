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
dataset = load_iris()

x = dataset.data
y= dataset.target

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=77, shuffle =True, train_size = 0.8)


kfold = KFold(n_splits=5, shuffle=True)

model =LinearSVC()
score = cross_val_score(model,x_train,y_train,cv=kfold)  #model and data concatenate

print('score :',score) #fit하고 model.score까지 포함
#score : [1.         0.95833333 1.         0.95833333 0.95833333]  
# train 에서 val 5등분한게 분리안해주고 할때보다 1이 한개 더생김