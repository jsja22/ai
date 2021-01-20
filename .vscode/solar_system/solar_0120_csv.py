import pandas as pd
import glob
import os

input_file = 'C:/data/csv/predict1/'

output_file = 'C:/data/csv/predict1/result2.csv'

allFile_list = glob.glob(os.path.join(input_file, '0120predict3_*'))

print(allFile_list)
allData = []

for file in allFile_list:
    df = pd.read_csv(file) # for구문으로 csv파일들을 읽어 들인다
    df = df.iloc[:,-1]
    allData.append(df) # 빈 리스트에 읽어 들인 내용을 추가한다

dataCombine = pd.concat(allData, axis=1, ignore_index=True)

dataCombine.to_csv(output_file, index=False,header=0)