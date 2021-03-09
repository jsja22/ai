import numpy as np
import pandas as pd

def outliers(data_out):
    out = []
    if str(type(data_out))== str("<class 'numpy.ndarray'>"):
        print("numpyarray")
        for col in range(data_out.shape[1]):
            data = data_out[:,col]
            print(data)

            quartile_1, quartile_3 = np.percentile(data,[25,75])
            print("1사분위 : ",quartile_1)
            print("3사분위 : ",quartile_3)
            iqr = quartile_3 - quartile_1
            lower_bound = quartile_1 - (iqr*1.5)
            upper_bound = quartile_3 + (iqr*1.5)
            out_col = np.where((data>upper_bound)|(data<lower_bound))
            print(out_col)
            data = data[out_col]
            print(f"{col+1}번째 행렬의 이상치 값: ", data)
            out.append(out_col)

    return out


aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],[1000,2000,3,4000,5000,6000,-7000,8,9000,10000]])
aaa = aaa.transpose()
print("type aaa",type(aaa))
outlier_loc = outliers(aaa)

print("이상치의 위치 : ",outlier_loc)
