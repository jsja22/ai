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

def L1(img1,img2):
    return sum(abs(img1-img2)).sum()

def L2(img1,img2):
    return np.sqrt(np.sum((img1-img2)**2))

def decision(arr):
    cnt_list = [0]*10
    same=[]
    for i in arr:
        cnt_list[i] += 1
    frcnt = max(cnt_list)

    for i in range(10):
        if cnt_list[i] == frcnt:
            same.append(i)

    if len(same) != 1:
        for nearest in arr:
            for num in same:
                if num == nearest:
                    return num
    else:
        return cnt_list.index(frcnt)

def accuracy(pred,label):
    correct = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            correct = correct + 1
    return correct / float(len(pred))

class NearestNeighbor(object):
  def __init__(self, k = 3, L = L1):
    #pass
    self.K=k
    self.distance_type = L

  def train(self,X,Y):
    self.Xtr = X
    self.Ytr = Y

  def predict(self, X):
      k=self.K
      Ypred = []
      for i in tqdm(range(len(X))):
          distance_list = np.array([])
          for j in range(len(self.Xtr)):
              distance_list = np.append(distance_list,(self.distance_type(self.Xtr[j],X[i])))

          idx = np.argsort(distance_list)
          neighbors = self.Ytr[(idx[0:k])]
          pred = decision(neighbors)
          Ypred.append(pred)
      return Ypred



#     testNumber = X.shape[0]
#     Ypred = np.zeros(testNumber,dtype=self.Ytr.dtype)
#     for i in tqdm(range(testNumber)):
#       distance1 = np.sum(np.abs(self.Xtr - X[i,:]),axis = 1)
#       print(distance1.shape)
#       #distance2 = np.sqrt(np.sum(np.power(self.Xtr - X[i,:],2),axis = 1))
#       minIndex = np.argmin(distance1)
#       Ypred[i] = self.Ytr[minIndex]
#     return Ypred
#   def accuracy(self,pred,test):
#     correct = 0
#     for i in range(len(pred)):
#         if pred[i] == test[i]:
#             correct = correct + 1
#     return correct / float(len(pred))


nn = NearestNeighbor(k=3,L=L2)
nn.train(trainData[:], trainLabels[:])
Y_predict = nn.predict(testData[:])

testLabels = testLabels.tolist()
accuracy(Y_predict,testLabels)



