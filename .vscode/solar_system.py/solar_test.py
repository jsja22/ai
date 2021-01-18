import pandas as pd
import glob
import xlrd
import openpyxl
 
#우선 모든 자료들을 raw data folder에 저장
all_data = pd.DataFrame()
all_data1 = pd.DataFrame()

for f in glob.glob("C:/data/csv/*.csv"): #csv 불러옴
    df=pd.read_csv(f,index_col=None, header=0, encoding='cp949',thousands=',')
    all_data1 = pd.concat([df], ignore_index=True)
    all_data=all_data.append(all_data1,ignore_index=True)


all_data.to_csv('C:/data/csv/sum_test.csv',index_col=None, header=0,encoding='cp949',thousands=',') 

df1 = pd.read_csv('C:/data/csv/sum_test.csv',index_col=None, header=0)
print(df1)
print(df1.shape)