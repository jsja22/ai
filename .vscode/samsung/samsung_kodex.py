import numpy as np
import pandas as pd

df = pd.read_csv('C:/data/csv/kodex.csv', index_col=0, header=0, encoding='cp949',thousands=',')

print(df.shape) #(1088, 16)
print(df.columns) #Index(['시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량', '금액(백만)',
                    # '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],

#결측치 떄문에 데이터 맞춰주기
datasets_1 = df.iloc[:664,:]
datasets_2 = df.iloc[667:,:]
df = pd.concat([datasets_1,datasets_2])


df = df.drop(['전일비','Unnamed: 6'], axis=1)
df = df.drop(['거래량','신용비','외인비','개인','기관','외인(수량)','외국계','프로그램'], axis=1)
df = df[::-1]

print(df.shape) #(1085, 6)
print(df.isnull().sum())

df2 = df.values
np.save('C:/data/npy/kodex.npy', arr=df2)