import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris ,load_breast_cancer,load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
dataset = load_boston()

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
# model_score :  0.8111288663608667
# r2_score :  0.8111288663608667
# score : [0.5546244  0.6512012  0.77282217 0.7329422  0.67999801]

# KNeighborsRegressor()
# model_score :  0.8404010032786686
# r2_score :  0.8404010032786686
# score : [0.62967742 0.77320992 0.7510784  0.71257016 0.64664014]

# DecisionTreeRegressor()
# model_score :  0.7875457648836091
# r2_score :  0.7875457648836091
# score : [0.38523385 0.60647734 0.86042396 0.79491717 0.68359838]

# RandomForestRegressor()
# model_score :  0.9243326917640274
# r2_score :  0.9243326917640274
# score : [0.88835889 0.79268249 0.91191501 0.87288337 0.77582715]