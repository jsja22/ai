import os
import numpy as np
import struct
from struct import unpack
from tqdm import tqdm
path = 'C:/Users/bitcamp/Desktop/hyeon/'

def read(dataset="training", datatype='images'):
  if dataset is "training":
    fname_img = os.path.join(path, 'train-images.idx3-ubyte')
    fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
  elif dataset is "testing":
    fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
    fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')

  with open(fname_lbl,'rb') as flbl:
    magic, num = struct.unpack(">II", flbl.read(8))
    lbl = np.fromfile(flbl,dtype=np.int8)
    print(lbl.shape)
  with open(fname_img,'rb') as fimg:
    magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = np.fromfile(fimg,dtype=np.uint8).reshape(len(lbl),rows,cols)
  if(datatype == 'images'):
    get_data = lambda idx:img[idx]
  elif(datatype == 'labels'):
    get_data = lambda idx:lbl[idx]

  for i in range(len(lbl)):
    yield get_data(i)

trainData=np.array(list(read('training', 'images')))
trainLabels=np.array(list(read('training', 'labels')))
testData=np.array(list(read('testing','images')))
testLabels=np.array(list(read('testing','labels')))    

print(trainData.shape)
print(trainLabels.shape)
print(testData.shape)
print(testLabels.shape)

# (60000, 28, 28)
# (60000,)
# (10000, 28, 28)
# (10000,)

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self,X,Y):
    self.Xtr = X
    self.Ytr = Y

  def predict(self, X, k):
    testNumber = X.shape[0]
    Ypred = np.zeros(testNumber,dtype=self.Ytr.dtype)
    distance_array = np.array([])

    for i in tqdm(range(testNumber)):
      for j in range(self.Xtr.shape[0]):
        L1_distance = np.sum(np.abs(self.Xtr[j] - X[i]))
        #distance2 = np.sqrt(np.sum(np.power(self.Xtr[j] - X[i],2)))
        distance_array = np.append(distance_array, L1_distance)
      index = np.argsort(np.sort(distance_array))
      wantedIndex = index[:k]
      Ypred[i] = self.Ytr[wantedIndex]
    
      # minIndex = np.argmin(distance1)
      # Ypred[i] = self.Ytr[minIndex]
      
    return Ypred

nn = NearestNeighbor()
nn.train(trainData[:5000], trainLabels[:5000])
Y_predict = nn.predict(testData[:1000],k=1)
print('accuracy : %f'%(100*np.mean(Y_predict == testLabels)))


