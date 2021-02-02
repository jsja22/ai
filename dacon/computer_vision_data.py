import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('C:/data/computer vision/mnist_data/train.csv')
test = pd.read_csv('C:/data/computer vision/mnist_data/test.csv')
print(train.shape, test.shape) #(2048, 787) (20480, 786)

