import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris ,load_breast_cancer  #다중 분류모델
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
dataset = load_breast_cancer()

x = dataset.data
y= dataset.target


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

scalar = StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)

model = SVC()

from tensorflow.keras.callbacks import EarlyStopping

model.fit(x_train,y_train)

result = model.score(x_test,y_test)
print("model.score:",result) #(loss,acuracy)

y_pred =model.predict(x_test)
print("test predict: {}".format(y_pred)) 

acc = accuracy_score(y_test,y_pred)
print("acuuracy_score: ",acc)

# LinearSVC()
#model.score: 0.9736842105263158
#test predict: [1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 1 1 1 1 1 0 0 0 1 0
# 0 1 1 1 1 0 1 0 1 1 1 0 0 0 1 0 1 0 0 1 0 1 1 1 1 1 1 0 1 1 1 1 0 1 1 1 1
# 0 0 1 0 1 1 1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0
# 1 1 1]
#acuuracy_score:  0.9736842105263158

#SVC
#model.score: 0.9649122807017544
#test predict: [1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 0 0 1 0 0 1 0 1 0 0 1 0 1 1 1 1 1 0 1 0 1 0
# 0 1 1 1 1 0 1 0 1 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 1 0 1 1 1 1
# 0 0 1 0 1 1 1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0
# 1 1 1]
#acuuracy_score:  0.9649122807017544

## KNeighborsClassifier
#model.score: 0.956140350877193
#test predict: [1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 1 1 1 1 1 0 1 0 1 0
# 0 1 1 1 1 0 1 0 1 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 0 1 0 1 0 1 1 1 1
# 0 1 1 0 0 1 1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0
# 1 1 1]
#acuuracy_score:  0.956140350877193

#DecisionTreeClassifier(max_depth=7, random_state=1)
#model.score: 0.9473684210526315
#test predict: [1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 1 1 1 1 0 1 0 1 0
# 0 1 1 1 1 0 1 0 1 1 1 0 0 0 1 0 1 0 0 0 0 1 1 0 1 1 1 0 0 1 0 1 0 1 1 1 1
# 0 0 1 0 0 1 1 0 1 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0
# 1 1 1]
#acuuracy_score:  0.9473684210526315

#RandomForestClassifier()
#model.score: 0.956140350877193
#test predict: [1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 0 1 0 1 1 1 1 1 0 1 0 1 0
# 0 1 1 1 1 0 1 0 1 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 1 1 0 1 1 0 1 0 1 1 1 1
# 0 1 1 0 1 1 1 0 1 0 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 0 1 1 1 0 1 0
# 1 1 1]
#acuuracy_score:  0.956140350877193

#tensorflow
#Dense model acc->0.9824561476707458
#LSTM model -> acc: 

#전반적으로 텐서플로우에서 덴스모델이 머신러닝으로 돌린 모델보다 성능우수!