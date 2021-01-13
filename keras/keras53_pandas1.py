import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()

print(dataset.keys()) #dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.values())

print(dataset.target_names)  #['setosa' 'versicolor' 'virginica'] =>(0,1,2)로 표현

x= dataset.data
#x= dataset['data']
y= dataset.target
#y= dataset['target']  #dict는  이렇게도 표현 가능해서 원하는 방식으로 쓰면 됨. 

print(x)
print(y)
print(x.shape,y.shape)  #(150, 4) (150,)
print(type(x),type(y))  #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

df= pd.DataFrame(x, columns=dataset['feature_names'])  #(data, dataframe의 header)  #출력했을때 가장 왼쪽에 나와있는 0,1,2 ...~ 값들은 index임
print(df)
print(df.shape) #(150, 4) #참고 : list에선 shape가 안먹힘 
print(df.columns)#Index(['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],  dtype='object') 
df.columns = ['sepal_length','sepal_width','petal_length','petal_width']  #clounm명 변경 
print(df.columns)       
print(df.index)  #RangeIndex(start=0, stop=150, step=1)  총 150개 index가 들어가있음. index는 명시안해주면 자동 indexing 된다.
print(df.head()) #df[:5]
print(df.tail()) #df[-5:]
print(df.info()) 
print(df.describe()) #count,mean, std, min, 25%,... max  다 볼수있음

#y column  추가
print(df['sepal_length'])
df['Target']= dataset.target
print(df.head)
print(df.shape)  #(150, 5)
print(df.columns)
print(df.index)
print(df.tail())  #target값은 2 shuffle안되어있기때문에 0~49 = 0 50~99 =1 100~149 =2  =>iris데이터

print(df.info())
print(df.isnull())
print(df.isnull().sum)
print(df.describe())
print(df['Target'].value_counts()) #2    50
                                   #1    50
                                   #0    50

#상관계수 히트맵
print(df.corr()) # 1에가까울수록 상관관계가 높음

import matplotlib.pyplot as plt
import seaborn as sns

# sns.set(font_scale=1.2)  #글자크기
# sns.heatmap(data=df.corr(),square=True, annot=True, cbar=True)  #heatmap 사각형, square 사각형 형태 annot 글씨를 넣어주는거 cbar 오른쪽 옆에 바 표시 #색이 찐할수록 안좋음
# plt.show()

#도수 분포표
plt.figure(figsize=(10,6))

plt.subplot(2,2,1)
plt.hist(x='sepal_length',data=df)  #여기서 hist는 histogram fit의 hist는 history 둘은 다른것이다!
plt.title('sepal_length')

plt.subplot(2,2,2)
plt.hist(x='sepal_width',data=df)
plt.title('sepal_width')

plt.subplot(2,2,3)
plt.hist(x='petal_length',data=df)
plt.title('petal_length')

plt.subplot(2,2,4)
plt.hist(x='petal_width',data=df)
plt.title('petal_width')

plt.show()


