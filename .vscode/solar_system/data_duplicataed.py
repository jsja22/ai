import pandas as pd
import numpy as np

#data 준비
train_df = pd.read_csv('C:/data/csv/solar/train/train.csv')
print(train_df .shape)     #(52560, 9)
print(train_df .tail())
sample = pd.read_csv('C:/data/csv/solar/sample_submission.csv')

df = pd.DataFrame(train_df)

print(df.duplicated())
print(df.duplicated().sum())

def preprocess_data (data, is_train=True) :
    temp = data.copy()
    temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
    if is_train == True :    
        temp['Target1'] = temp['TARGET'].shift(-48).fillna(method='ffill')   # 다음날 TARGET을 붙인다.
        temp['Target2'] = temp['TARGET'].shift(-48*2).fillna(method='ffill') # 다다음날 TARGET을 붙인다.
        temp = temp.dropna()    # 결측값 제거
        return temp.iloc[:-96]  # 이틀치 데이터만 빼고 전체
    elif is_train == False :         
        # Day, Minute 컬럼 제거
        temp = temp[['Hour', 'TARGET', 'DHI', 'DNI', 'WS', 'RH', 'T']]
        return temp.iloc[-48:, :] # 마지막 하루치 데이터


# test_data = []
# for i in range(81):
#     file_path = 'C:/data/csv/solar/test/%d.csv'%i
#     temp = pd.read_csv(file_path)
#     temp = preprocess_data(temp, is_train=False)
#     test_data.append(temp)

# x_test = pd.concat(test_data)
# print(x_test.shape) #(3888, 8) # 81day 48 hour 8 columns
# test_dataset = x_test.to_numpy()

# print(x_test.duplicated())
# print(x_test.duplicated().sum()) #43개 가 곂침 ...
# #x_pred = test_dataset.reshape(81, 48, 7) 
# #print(x_pred.shape)
# #print(x_pred.duplicated())
# #print(x_pred.duplicated().sum())

# df_test = [] # 리스트

# for i in range(81):
#     file_path = 'C:/data/csv/solar/test/' + str(i) + '.csv'
#     temp = pd.read_csv(file_path) # 데이터 프레임 shape = (336, 9)
#     temp = preprocess_data(temp, is_train=False) 
#     df_test.append(temp) # 리스트에 전처리된 데이터 프레임 append

# X_test = pd.concat(df_test) # 전처리된 데이터 프레임들 세로 병합
# print(X_test.shape)

# print(X_test.duplicated())
# print(X_test.duplicated().sum())

def preprocess_data(data):
	temp = data.copy()
	return temp.iloc[-48:, :]

df_test = []

for i in range(81):
  file_path = 'C:/data/csv/solar/test/' + str(i) + '.csv'
  temp = pd.read_csv(file_path)
  temp = preprocess_data(temp)
  df_test.append(temp)

X_test = pd.concat(df_test)
#Attach padding dummy time series
X_test = X_test.append(X_test[-96:])
print(X_test.shape)

print(X_test.duplicated())
print(X_test.duplicated().sum())
print(X_test.tail())