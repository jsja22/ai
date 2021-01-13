import numpy as np
import pandas as pd

df = pd.read_csv('C:/data/csv/iris_sklearn.csv',index_col=0,header=0) #index col은 data가 아니라는것을 명시  0번쨰가 header라는것을 인식시켜줌 

print(df)