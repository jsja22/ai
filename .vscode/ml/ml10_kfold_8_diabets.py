import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris ,load_breast_cancer,load_boston, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
dataset = load_diabetes()

x = dataset.data
y= dataset.target


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True,random_state=66)

scalar = StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test = scalar.transform(x_test)

kfold = KFold(n_splits=5, shuffle=True)


models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
for i in models:
    model = i
    score = cross_val_score(model,x_train,y_train,cv=kfold)  #model and data concatenate

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    print('r2_score : ', r2)
    print('score :',score)
# LinearRegression()
# model_score :  0.5063891053505039
# r2_score :  0.5063891053505039
# score : [0.46667918 0.4713362  0.37028633 0.60924617 0.4835923 ]

# KNeighborsRegressor()
# model_score :  0.38626977834604637
# r2_score :  0.38626977834604637
# score : [0.52210728 0.37888683 0.34175598 0.4114938  0.30779638]

# DecisionTreeRegressor()
# model_score :  -0.17582931787725609
# r2_score :  -0.17582931787725609
# score : [ 0.10458273  0.10699988 -0.30182267 -0.27815861 -0.04855048]

# RandomForestRegressor()
# model_score :  0.3497387235644682
# r2_score :  0.3497387235644682
# score : [0.49507718 0.29610166 0.48940065 0.48782522 0.47768087]