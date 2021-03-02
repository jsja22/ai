import numpy as np
import pandas as pd
import tensorflow.keras.backend as K

x = []
for i in range(1,4):
    df = pd.read_csv(f'C:/data/computer_vision2/dirty_mnist_2/csv/best{i}.csv', index_col=0, header=0)
    data = df.to_numpy()
    x.append(data)

x = np.array(x)

df = pd.read_csv(f'C:/data/computer_vision2/dirty_mnist_2/csv/best{i}.csv', index_col=0, header=0)
for i in range(5000):
    for j in range(26):
        a = []
        for k in range(3):
            a.append(x[k,i,j].astype('float32'))
        a = np.array(a)
        df.iloc[[i],[j]] = (pd.DataFrame(a).astype('float32').quantile(0.5,axis = 0)[0]).astype('float32')
        
y = pd.DataFrame(df, index = None, columns = None)
y.to_csv('C:/data/computer_vision2/dirty_mnist_2/last_submission.csv')  