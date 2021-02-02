import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('C:/data/computer vision/mnist_data/train.csv',index_col=None, header=0)
test = pd.read_csv('C:/data/computer vision/mnist_data/test.csv',index_col=None, header=0)
sub = pd.read_csv('C:/data/computer vision/mnist_data/submission.csv',index_col=None, header=0)
submission = pd.read_csv('C:/data/computer vision/sample_submission.csv',index_col=None, header=0)
print(train.shape, test.shape) #(2048, 787) (20480, 786)
print(submission.shape) #[5000 rows x 27 columns] #5000개의 이미지 데이터에서 각각 a-z사이의 알파벳이 존재하면 1로 출력하면 된다.
print(submission.shape) #[20480 rows x 2 columns]
print(train,test,submission,sub) 

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255
y1 = temp.iloc[:,1]
y2 = temp.iloc[:,2]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y1 = y1.to_numpy()
y2 = y2.to_numpy()

x_pred = x_test.to_numpy()
# pca = PCA()
# x = pca.fit(x)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# print("cumsum :",cumsum)

# d = np.argmax(cumsum >=0.965)+1
# print("cumsum >=0.965 :", cumsum>=0.965)
# print("d :", d) # 131
pca = PCA(n_components=131)
x = pca.fit_transform(x)

#x_train,x_test,y1_train,y1_test,y2_train,y2_test = train_test_split(x,y1,y2, train_size=0.8,shuffle=True, random_state=66)
x_train,x_test,y1_train,y1_test = train_test_split(x,y1, train_size=0.8,shuffle=True, random_state=66)

model = XGBClassifier(n_jobs=8, eval_metrics='merror')

#model.fit(x_train,[y1_train,y2_train], verbose=True, eval_set=[(x_train,[y1_train,y2_train]),(x_test,[y1_test,y2_test])])
model.fit(x_train,y1_train, verbose=True, eval_set=[(x_train,y1_train),(x_test,y1_test)])

#accuracy = model.score(x_test, [y1_test,y2_test])
accuracy = model.score(x_test, y1_test)

print("acc :", accuracy) #acc : 0.275609756097561