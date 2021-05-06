import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from tqdm import tqdm
from scipy import stats
x = []
for i in range(1,9):
    df = pd.read_csv(f'C:/data/kaggle/csv/sub{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

print(x)

x = np.array(x)
print(x.shape) #(3, 72000, 1)

a= []
df = pd.read_csv(f'C:/data/kaggle/csv/sub{i}.csv', index_col=0, header=0)
for i in range(100000):
    for j in range(1):
        b = []
        for k in range(8):
            b.append(x[k,i,j].astype('int'))
        a.append(stats.mode(b)[0])  #stats.mode 가장 보편적인 값을 찾음

sub = pd.read_csv('C:/data/kaggle/sample_submission.csv')
sub['Survived'] = np.array(a)
sub.to_csv('C:/data/kaggle/csv/answer_concat4.csv',index=False)