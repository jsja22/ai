import numpy as np
import pandas as pd
import re


df = pd.read_csv('C:/data/csv/samsung.csv',index_col=0,header=0, encoding='cp949')

df.columns = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
df.replace(',','',inplace=True, regex=True)
df = df.astype('float32')
df = df.sort_values(by=['일자'],axis=0)

print(df)

df_x = df.loc['2018-05-04': '2021-01-12']
df_y = df.loc['2018-05-08' : '2021-01-13']
df_x1 = df.loc['2021-01-13': ]

print(df_x.shape)   # 661, 14
print(df_y.shape)   # 661, 14

df_x = df_x.iloc[:,[0,1,2,9,12]]
df_x1 = df_x1.iloc[:,[0,1,2,9,12]]
print(df_x)

df_y = df_y.iloc[:,[3]]

print(df_y)
print(df_x)
x= df_x.to_numpy()
y = df_y.to_numpy()
x_pred = df_x1.to_numpy()

print(x)
print(y)
print(x.shape)
print(y.shape)