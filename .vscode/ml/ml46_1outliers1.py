#이상치 처리
#1. 0 처리
#2. nan 처리후 보간
#3. 3,4,5~~~

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
aaa = np.array([1,2,-200,4,6,7,90,100,200,300])
def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    #사분위는 데이터값이 높은순으로 정렬 1/4
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : "), quartile_3
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr *1.5)
    upper_bound = quartile_3 + (iqr *1.5)

  
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outlier_loc = outliers(aaa)
print("이상치의 위치: ", outlier_loc)
# 이상치의 위치:  (array([8, 9], dtype=int64),)
plt.figure(figsize=(12,8))
sns.boxplot(data=aaa,color='red')
plt.show()