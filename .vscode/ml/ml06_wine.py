import numpy as np
from sklearn.datasets import load_wine  #다중분류
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression  #회기가 아니라 분류이다
import matplotlib.pyplot as plt

dataset = load_wine()

x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True)

scalar = MinMaxScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)

models = [LinearSVC(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred,y_test)
    print('accuracy_score : ', accuracy)


####MinmAX 가 더잘나옴
#SVC()
#model.score: 1.0
#test predict: [1 0 2 1 0 0 0 2 0 0 2 2 0 1 0 1 1 0 1 1 1 1 0 2 0 1 1 1 1 1 0 0 2 2 0 1]
#acuuracy_score:  1.0

# LinearSVC
#model.score: 1.0
#test predict: [1 0 2 1 0 2 1 2 1 1 0 1 0 0 1 1 0 0 2 2 0 0 2 1 2 1 2 2 0 1 0 0 2 2 2 2]
#acuuracy_score:  1.0

### KNeighborsClassifier
#model.score: 1.0
#test predict: [0 1 2 0 2 0 0 0 1 1 1 2 0 1 2 0 0 0 1 2 1 0 0 1 1 0 0 0 1 0 1 0 0 1 0 1]
#acuuracy_score:  1.0


###DecisionTreeClassifier()
#model.score: 0.9166666666666666
#test predict: [0 0 0 1 1 0 2 1 0 2 2 2 2 1 2 1 2 2 2 0 1 2 2 1 2 0 1 0 0 1 0 1 1 0 2 2]
#acuuracy_score:  0.9166666666666666

#DecisionTreeClassifier(max_depth=7, random_state=1)
#model.score: 1.0
#test predict: [2 1 2 1 1 0 0 1 1 0 1 0 1 2 1 1 1 1 2 0 0 0 0 2 2 1 0 2 0 1 1 2 0 0 2 2]
#acuuracy_score:  1.0

#RandomForestClassifier()
#model.score: 0.9722222222222222
#test predict: [2 2 1 2 1 0 2 0 1 2 1 2 2 0 2 1 0 2 0 1 0 2 2 0 1 2 2 1 2 1 1 1 2 1 1 2]
#acuuracy_score:  0.9722222222222222


#tensorflow
#Dense model acc->0.9824561476707458
#LSTM model -> acc: 

#전반적으로 텐서플로우에서 덴스모델이 머신러닝으로 돌린 모델보다 성능우수!