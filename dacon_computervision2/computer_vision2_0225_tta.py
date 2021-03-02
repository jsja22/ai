#tta
# Applying different transformations to test images and averaging will improve accuracy.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import imutils
import zipfile
import os
from PIL import Image
from typing import Tuple, Sequence, Callable
import glob
from sklearn.model_selection import KFold
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 디바이스 설정

labels_df = pd.read_csv('C:/data/computer_vision2/dirty_mnist_2nd_answer.csv')[:]
imgs_dir = np.array(sorted(glob.glob('C:/data/computer_vision2/dirty_mnist_2/dirty_mnist_2nd/*')))[:]
labels = np.array(labels_df.values[:,1:])
test_imgs_dir = np.array(sorted(glob.glob('C:/data/computer_vision2/dirty_mnist_2/test_dirty_mnist/*')))

print(labels.shape) #(50000, 26)
class MnistDataset_v1(Dataset):
    def __init__(self, imgs_dir=None, labels=None, transform=None, train=True):
        self.imgs_dir = imgs_dir
        self.labels = labels
        self.transform = transform
        self.train = train
        pass
    
    def __len__(self):
        # 데이터 총 샘플 수
        return len(self.imgs_dir)
    
    def __getitem__(self, idx):
        # 1개 샘플 get
        img = cv2.imread(self.imgs_dir[idx], cv2.IMREAD_COLOR)
        img = self.transform(img)
        if self.train==True:
            label = self.labels[idx]
            return img, label
        else:
            return img
        
        pass
resnext = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)

class Resnext(nn.Module):
    def __init__(self):
        super(Resnext, self).__init__()
        self.resnext = resnext 
        self.FC = nn.Linear(1000, 26)  #fully connected layer 정의  input node 1000 output node 26
        nn.init.xavier_normal_(self.FC.weight) #Xavier 초깃값 

        

    def forward(self, x):
        x = self.resnext(x)
        x = torch.sigmoid(self.FC(x))
        return x

model = Resnext()
print(model)

kf = KFold(n_splits=5, shuffle=True)
folds=[]
for train_idx, valid_idx in kf.split(labels_df):
    folds.append((train_idx, valid_idx))  #dirty_mnist_2nd_answer.csv -> kfold

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True

#seed_everything(42)
for fold in range(1):
    model = Resnext().to(device)
#     model = nn.DataParallel(model)
    train_idx = folds[fold][0]
    valid_idx = folds[fold][1]



    train_transform = transforms.Compose([                                
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
        ])
    valid_transform = transforms.Compose([                                 
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
        ])


    epochs=30
    batch_size=64        
    
    
    
    # Data Loader
    train_dataset = MnistDataset_v1(imgs_dir=imgs_dir[train_idx], labels=labels[train_idx], transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    valid_dataset = MnistDataset_v1(imgs_dir=imgs_dir[valid_idx], labels=labels[valid_idx], transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=16, shuffle=False)  
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones= [23,29], gamma=0.5)

    criterion = torch.nn.BCELoss()
    
    
    epoch_accuracy = []
    valid_accuracy = []
    valid_losses=[]
    valid_best_accuracy=0

    best_models=[]

    for epoch in range(epochs):
      with tqdm(train_loader,total=train_loader.__len__(),unit='batch') as train_bar:
        model.train()
        batch_accuracy_list = []
        batch_loss_list = []
        start=time.time()
        for n, (X, y) in enumerate((train_bar)):
            train_bar.set_description(f"Train Epoch {epoch}")
            X = torch.tensor(X, device=device, dtype=torch.float32)
            y = torch.tensor(y, device=device, dtype=torch.float32)
            y_hat = model(X)
            
            
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()


            
            y_hat  = y_hat.cpu().detach().numpy()
            y_hat = y_hat>0.5
            y = y.cpu().detach().numpy()

            batch_accuracy = (y_hat == y).mean()
            batch_accuracy_list.append(batch_accuracy)
            batch_loss_list.append(loss.item())
            train_acc = np.mean(batch_accuracy_list)
            
            train_bar.set_postfix(train_loss= loss.item(),train_acc = train_acc)

        model.eval()# eval()모드를 사용하면 모델 내부의 모든 layer가 evaluation 모드가 됩니다. evaluation 모드에서는 batchnorm, dropout과 같은 기능들이 사용되지 않습니다.
        valid_batch_accuracy=[]
        valid_batch_loss = []

        with torch.no_grad():
          with tqdm(valid_loader,total=valid_loader.__len__(),unit="batch") as valid_bar:
            for n_valid, (X_valid, y_valid) in enumerate((valid_bar)):
                valid_bar.set_description(f"Valid Epoch {epoch}")
                X_valid = torch.tensor(X_valid, device=device)#, dtype=torch.float32)
                y_valid = torch.tensor(y_valid, device=device, dtype=torch.float32)
                y_valid_hat = model(X_valid)
                
                valid_loss = criterion(y_valid_hat, y_valid).item()
                
                y_valid_hat = y_valid_hat.cpu().detach().numpy()>0.5
                
                
                valid_batch_loss.append(valid_loss)
                valid_batch_accuracy.append((y_valid_hat == y_valid.cpu().detach().numpy()).mean())
                val_acc=np.mean(valid_batch_accuracy)
                valid_bar.set_postfix(valid_loss = valid_loss,valid_acc = val_acc)
                
            valid_losses.append(np.mean(valid_batch_loss))
            valid_accuracy.append(np.mean(valid_batch_accuracy))

        scheduler.step()

        if np.mean(valid_batch_accuracy) > 0.87:
            path = "C:/data/computer_vision2/models/"
            MODEL = "resnext"
            torch.save(model, f'{path}_{MODEL}_{valid_loss:2.4f}_epoch_{epoch}.pth')

        if np.mean(valid_batch_accuracy)>valid_best_accuracy:
            best_model=model
            valid_best_accuracy = np.mean(valid_batch_accuracy)

    best_models.append(best_model)


best_modelss=[]
model1 = torch.load('C:/data/computer_vision2/models/_resnext_0.2439_epoch_27.pth')
best_modelss.append(model1)
print(best_modelss)
test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
        ])

submission = pd.read_csv("C:/data/computer_vision2/sample_submission.csv")

for model in best_modelss:
    with torch.no_grad():
        model.eval()

        test_dataset = MnistDataset_v1(imgs_dir=test_imgs_dir, transform=test_transform, train=False)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

        for n, X_test in enumerate(tqdm(test_loader)):
            X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
            with torch.no_grad():
                model.eval()  
                pred_test = model(X_test).cpu().detach().numpy()
                submission.iloc[n*32:(n+1)*32,1:] += pred_test

submission.iloc[:,1:] = np.where(submission.values[:,1:]>=0.5, 1,0)


submission.to_csv('C:/data/computer_vision2/submission/02212270.csv', index=False)
'''
pred1=pd.read_csv('C:/data/computer_vision2/EfficientNetB0-fold0_0223_02.csv')
pred2=pd.read_csv('C:/data/computer_vision2/EfficientNetB2-fold0_0225_02.csv')
pred3=pd.read_csv('C:/data/computer_vision2/submission/02212270.csv')
pred4=pd.read_csv('C:/data/computer_vision2/submission/0227_1.csv')
final=((pred1 + pred2 + pred3+pred4)/4 > 0.5)*1
final['index']=pred1['index']

final.to_csv('C:/data/computer_vision2/submission/ensemble4.csv', index=False)