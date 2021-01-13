import numpy as np
import pandas as pd

sdf = pd.read_csv('C:/data/csv/samsung.csv',index_col=0,header=0,encoding='cp949') #index col은 data가 아니라는것을 명시  0번쨰가 header라는것을 인식시켜줌 

print(sdf)

#print(sdf.shape) #(2400, 14)
#print(sdf.info()) #<class 'pandas.core.frame.DataFrame'>

sdf_num=sdf.to_numpy()
np.save('C:/data/npy/samsung1.npy', arr=sdf_num)

sdf= sdf.dropna(how='all') #모든 row가 NaN 일 때만 row를 제거

for i in range(len(sdf[:662])):
    sdf.iloc[i,0] = str(sdf.iloc[i,0])
    
print(type(sdf))
#str -> int 변환
for i in range(len(sdf[:662])):
     sdf.iloc[i,0] = int(sdf.iloc[i,0].replace(',',''))
    
print(sdf) 
sdf = sdf.sort_values(['일자'], ascending = [True])
print(sdf)

sdf= sdf.values
print(sdf)

np.save('C:/data/npy/samsung2.npy',arr=sdf)



