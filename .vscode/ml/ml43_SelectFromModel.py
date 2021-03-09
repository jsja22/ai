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
        train_size=0.8, random_state=66)

model = XGBRegressor(n_estimators=1000, learning_rate=0.1)

model.fit(x_train, y_train, verbose=True,  eval_metric=["logloss","rmse"],
                eval_set=[(x_train, y_train), (x_test, y_test)],
                early_stopping_rounds=20)
result = model.evals_result()
print(result)

y_pred = model.predict(x_test)

r2 = r2_score(y_pred, y_test)
print("R2 : ", r2_score)

thresholds = np.sort(model.feature_importances_)
print(thresholds)

for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)  #predit ??

    parameters = {
        'n_estimators': [100, 200, 400],
        'learning_rate' : [0.01, 0.03, 0.05, 0.1],
        'colsample_bytree': [0.5, 0.7, 0.8, 0.9],
        'colsample_bylevel':[0.5, 0.7, 0.8, 0.9],
        'max_depth': [4, 5, 6]
    }
    search = RandomizedSearchCV(XGBRegressor(), parameters, cv=5, n_jobs=-1)

    selection_x_train = selection.transform(x_train)
    print(selection_x_train.shape)

    search.fit(selection_x_train, y_train)

    selection_x_test = selection.transform(x_test)
    x_pred = search.predict(selection_x_test)

    score = r2_score(y_test,x_pred)
    print('best_parameter: ', search.best_estimator_)
    print("Thresh =%.3f, n=%d, R2: %.2f%%" %(thresh, selection_x_train.shape[1], score*100))
    
# print(model.coef_)
# print(model.intercept_)
# Coefficients are not defined for Booster type None

#최적의 랜덤서치 와 r2값 피처임포턴스는
# (404, 12)
# best_parameter:  XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
#              colsample_bynode=1, colsample_bytree=0.7, gamma=0, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.03, max_delta_step=0, max_depth=6,
#              min_child_weight=1, missing=nan, monotone_constraints='()',
#              n_estimators=200, n_jobs=8, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
# Thresh =0.003, n=12, R2: 93.79%


