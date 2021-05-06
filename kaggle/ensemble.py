import numpy as np 
import pandas as pd 

sub = pd.read_csv('C:/data/kaggle/sample_submission.csv')

sub1 = pd.read_csv('C:/data/kaggle/csv/sub2.csv') 
sub2 = pd.read_csv('C:/data/kaggle/csv/sub9.csv') 
sub3 = pd.read_csv('C:/data/kaggle/csv/sub8.csv') 
sub4 = pd.read_csv('C:/data/kaggle/csv/sub4.csv')

res = (2*sub1['Survived'] + sub2['Survived'] + sub3['Survived'] + 2*sub4['Survived'])/6
sub.Survived = np.where(res > 0.5, 1, 0).astype(int)

sub.to_csv("C:/data/kaggle/csv/last.csv", index = False)
sub['Survived'].mean()

