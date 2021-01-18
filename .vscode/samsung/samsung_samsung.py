import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Lambda,Conv1D
from sklearn.metrics import mean_squared_error,r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


dataset= pd.read_csv('C:/data/csv/samsung.csv', index_col=0, header=0, encoding='cp949')

print(dataset.columns) #Index(['시가', '고가', '저가', '종가', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관',
                  # '외인(수량)', '외국계', '프로그램', '외인비'],

dataset['시가'] = dataset.loc[:,['시가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['고가'] = dataset.loc[:,['고가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['저가'] = dataset.loc[:,['저가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['종가'] = dataset.loc[:,['종가']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['거래량'] = dataset.loc[:,['거래량']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['금액(백만)'] = dataset.loc[:,['금액(백만)']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['개인'] = dataset.loc[:,['개인']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['기관'] = dataset.loc[:,['기관']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['외인(수량)'] = dataset.loc[:,['외인(수량)']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['외국계'] = dataset.loc[:,['외국계']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)
dataset['프로그램'] = dataset.loc[:,['프로그램']].apply(lambda x: x.str.replace(',', '').astype(float), axis=1)

print(type(dataset.iloc[0, 1])) #<class 'numpy.float64'>

#결측치 있는 3개행 제거
datasets_1 = dataset.iloc[:662,:]
datasets_2 = dataset.iloc[665:,:]

dataset = pd.concat([datasets_1,datasets_2])

#추가 데이터 samsung2

dataset2 = pd.read_csv('C:/data/csv/samsung2.csv', encoding='cp949', index_col=0, header=0, thousands=',')
dataset2 = dataset2.dropna()
dataset2 = dataset2.drop(['전일비','Unnamed: 6'], axis=1)
dataset2 = dataset2.drop(['2021-01-13'])
print(dataset2)

dataset3 = pd.read_csv('C:/data/csv/samsung3.csv', encoding='cp949', index_col=0, header=0, thousands=',')
dataset3 = dataset3.iloc[:1,:]
dataset3 = dataset3.drop(['전일비','Unnamed: 6'], axis=1)

#2018-05-03까지 데이터 액면분할

dataset = dataset[::-1]  
dataset.loc[:'2018-05-04','시가':'종가'] = dataset.loc[:'2018-05-04','시가':'종가']/50.
dataset.loc[:'2018-05-04','거래량'] = dataset.loc[:'2018-05-04','거래량']*50.

print(dataset.head())

#samsung1,samsung2 합치기
df = pd.concat([dataset,dataset2], axis=0)
print(df.tail())
#합친거에 samsung3 합치기
df2 = pd.concat([df,dataset3], axis=0)
print(df2.tail())


df3 = df2.drop(['거래량','신용비','외인비','개인','기관','외인(수량)','외국계','프로그램'], axis=1)
#df = df.drop(['개인','기관','외인(수량)','외국계','프로그램'], axis=1)

print(df3.tail())

#상관 계수 확인하기

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(font_scale=0.5, font='Malgun Gothic', rc={'axes.unicode_minus':False})
# sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True) 
# plt.show()

df3 = df3.iloc[1314:,:]  ## 2016-08-10월까지 kodex데이터랑 맞춰주기
print(df3)

print(df3.isnull().sum())
#pandas를 numpy 형태로 변환
df4 = df3.values
print(df4)


np.save('C:/data/npy/samsung_2.npy', arr=df4)