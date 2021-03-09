import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data  = pd.read_csv('C:/data/csv/winequality-white.csv',index_col =None, header=0, sep=';')

count_data = data.groupby('quality')['quality'].count()
print(count_data)

import matplotlib.pyplot as plt
count_data.plot()
plt.show()

