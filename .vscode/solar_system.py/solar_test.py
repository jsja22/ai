import pandas as pd
import glob
import xlrd
import openpyxl
 

for i in range(81) :
    filepath = 'C:/data/csv/test/{}.csv'.format(i)
    globals()['test{}'.format(i)] = pd.read_csv(filepath,index_col=False)
    globals()['test_{}'.format(i)] = globals()['test{}'.format(i)].iloc[:,3:]
    all_data1 = pd.concat(['test%d','test%d'], ignore_index=True)
    all_data = all_data.append(all_data1,ignore_index=True)


all_data = all_data.values
print(all_data)

# #우선 모든 자료들을 raw data folder에 저장
# all_data = pd.DataFrame()
# all_data1 = pd.DataFrame()

# for f in glob.glob("C:/data/csv/*.csv"): #csv 불러옴
#     df=pd.read_csv(f,index_col=None, header=0, encoding='')
#     all_data1 = pd.concat([df], ignore_index=True)
#     all_data=all_data.append(all_data1,ignore_index=True)


# all_data.to_csv('C:/data/csv/test/sum_test.csv')
# #df1 = pd.read_csv('C:/data/csv/test/sum_test1.csv',index_col=None, header=0)
# #print(df1)
# #print(df1.shape)