import numpy as np
import pandas as pd

df = pd.read_csv('C:/data/csv/iris_sklearn.csv',index_col=0,header=0) #index col은 data가 아니라는것을 명시  0번쨰가 header라는것을 인식시켜줌 

print(df)

print(df.shape) #(150, 5)
print(df.info())

#pandas를 넘파이로 바꾸는 것을 찾아라

#print(df.values) #print(df.to_numpy())랑 동일한결과 
aa=df.to_numpy()
print(aa)
print(type(df.to_numpy())) #<class 'numpy.ndarray'> 변환됨
#-> target값이 float형태로 바뀜 numpy는 한가지 형태로만 사용하기 때문임.

np.save('C:/data/npy/iris_sklearn.npy', arr=aa)

#과제
#판다스의 loc  iloc에 대해 정리



