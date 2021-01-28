import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris ,load_breast_cancer,load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
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

models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
for i in models:
    model = i

    #3. compile fit
    model.fit(x_train,y_train)

    #4. evaluation, prediction
    print(f'\n{i}')
    result = model.score(x_test,y_test)
    print('model_score : ', result)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    print('r2_score : ', r2)


# LinearRegression()
# model_score :  0.8111288663608667
# r2_score :  0.8111288663608667

# KNeighborsRegressor()
# model_score :  0.8404010032786686
# r2_score :  0.8404010032786686

# DecisionTreeRegressor()
# model_score :  0.7353433006797355
# r2_score :  0.7353433006797355

# RandomForestRegressor()
# model_score :  0.9234872770495681
# r2_score :  0.9234872770495681

#성능은 Randomforest가 가장 우수