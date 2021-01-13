import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

x= dataset.data
#x= dataset['data']
y= dataset.target
#y= dataset['target']  #dict는  이렇게도 표현 가능해서 원하는 방식으로 쓰면 됨. 

df= pd.DataFrame(x, columns=dataset['feature_names'])  #(data, dataframe의 header)  #출력했을때 가장 왼쪽에 나와있는 0,1,2 ...~ 값들은 index임

df.columns = ['sepal_length','sepal_width','petal_length','petal_width']  #clounm명 변경 

df['Target'] =y

df.to_csv('C:/data/csv/iris_sklearn.csv',sep=',')  #sep 콤마로 구분
