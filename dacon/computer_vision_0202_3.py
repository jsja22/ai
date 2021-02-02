import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('C:/data/computer vision/mnist_data/train.csv',index_col=None, header=0)
test = pd.read_csv('C:/data/computer vision/mnist_data/test.csv',index_col=None, header=0)
sub = pd.read_csv('C:/data/computer vision/mnist_data/submission.csv',index_col=None, header=0)

print(train.shape, test.shape) #(2048, 787) (20480, 786)
print(sub.shape) #[20480 rows x 2 columns]
print(train,test,sub) 

temp = pd.DataFrame(train)
test_df = pd.DataFrame(test)

x = temp.iloc[:,3:]/255
y = temp.iloc[:,1]
x_test = test_df.iloc[:,2:]/255

x = x.to_numpy()
y = y.to_numpy()

x_train = x.reshape(-1,28,28,1)
print(x_train.shape)