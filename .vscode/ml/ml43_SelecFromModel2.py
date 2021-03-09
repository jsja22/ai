import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

dataset = load_boston()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
        train_size=0.8,shuffle=True, random_state=66)

xgb = XGBRegressor(n_estimators=1000, learning_rate=0.1)

parameters = {
    'n_estimators': [100, 200, 400],
    'learning_rate' : [0.01, 0.03, 0.05, 0.1],
    'colsample_bytree': [0.5, 0.7, 0.8, 0.9],
    'colsample_bylevel':[0.5, 0.7, 0.8, 0.9],
    'max_depth': [4, 5, 6]
}

model = RandomizedSearchCV(xgb,parameters, cv=5, n_jobs=-1 )

model.fit(x_train, y_train)# verbose=True,  eval_metric=["logloss","rmse"],
                #eval_set=[(x_train, y_train), (x_test, y_test)],
                #early_stopping_rounds=20)

score = model.score(x_test,y_test)
print("r2score: ", score)
thresholds = np.sort(model.best_estimator_.feature_importances_)
print(thresholds)

# from matplotlib import pyplot
# pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
# pyplot.show()

tmp = 0
tmp2 = [0,0]
for thresh in thresholds:
    selection = SelectFromModel(model.best_estimator_, threshold=thresh, prefit=True)  #predit ??

    selection_x_train = selection.transform(x_train)
    print(selection_x_train.shape)

    selection_model = XGBRegressor(n_jobs = 8)
    selection_model.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    y_pred = selection_model.predict(selection_x_test)

    score = r2_score(y_test,y_pred)
    if score > tmp :
        tmp = score
        tmp2[0] = thresh
        tmp2[1] = selection_x_train.shape[1]

    print('Thresh=%.3f, n=%d, R2: %.2f%%' %(thresh, selection_x_train.shape[1], score*100))
    print(f'Best Score so far : {tmp*100}%')
    print('Best Threshold : ', tmp2[0])

print('=========================================================================================')
print(f'Best Threshold : {tmp2[0]}, n = {tmp2[1]}')

selection = SelectFromModel(model.best_estimator_, threshold = tmp2[0], prefit = True)

selection_x_train = selection.transform(x_train)

selection_model = RandomizedSearchCV(xgb, parameters, cv =5)
selection_model.fit(selection_x_train, y_train)

selection_x_test = selection.transform(x_test)
y_predict = selection_model.predict(selection_x_test)

score = r2_score(y_test, y_predict)

print('=========================================================================================')
print(f'최종 R2 score : {score*100}%, n = {tmp2[1]}일때!!')
print('=========================================================================================')
print(f'1번 점수 : {tmp*100}%\n2번 점수 : {score*100}%')
print('=========================================================================================')



# =========================================================================================
# Best Threshold : 0.024063946679234505, n = 8
# =========================================================================================
# 최종 R2 score : 92.58032761093799%, n = 8일때!!
# =========================================================================================
# 1번 점수 : 93.5166310089852%
# 2번 점수 : 92.58032761093799%

