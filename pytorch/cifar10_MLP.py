import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets


#딥러닝 모델 설계할 때 장비 확인
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print('Using pytorch version:',torch.__version__,'Device:',DEVICE) #Using pytorch version: 1.7.1 Device: cuda

BATCH_SIZE = 32
EPOCHS = 10


#data load

train_dataset = datasets.CIFAR10(root = "../data/FashionMNIST",train=True, download=True,transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root = "../data/FashionMNIST",train=False, download=True,transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE, shuffle=False)

for(X_train, Y_train) in train_loader:
    print('x_train: ',X_train.size(), 'type: ', X_train.type())
    print('y_train: ',Y_train.size(), 'type: ', X_train.type())
    break

pltsize = 1
plt.figure(figsize=(10*pltsize,pltsize))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(np.transpose(X_train[i],(1,2,0)))
    plt.title('Class: ' + str(Y_train[i].item()))
#plt.show()

################MLP Model#####################3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32*3,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,10)
    def forward(self,x):
        x = x.view(-1,32*32*3)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
#optimizer, objective function setting

model = Net().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print(model)

# x_train:  torch.Size([32, 3, 32, 32]) type:  torch.FloatTensor
# y_train:  torch.Size([32]) type:  torch.FloatTensor
# Net(
#   (fc1): Linear(in_features=3072, out_features=512, bias=True)
#   (fc2): Linear(in_features=512, out_features=256, bias=True)
#   (fc3): Linear(in_features=256, out_features=10, bias=True)
# )

def train(model, train_loader,optimizer,log_interval):
    model.train()
    for batch_idx,(image,label) in enumerate(train_loader):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print("Train Epoch: {} [{}/{}({:.0f}%)]\tTrain Loss: {:.6f}".format(Epoch,batch_idx*len(image),len(train_loader.dataset),100.*batch_idx/len(train_loader), loss.item()))

#evaluate

def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad(): #gradient 흐름 억제 #그라디언트를 통해 파라미터 값이 없데이트 되는 현상을 방지
        for image, label in test_loader:
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)

            test_loss += criterion(output,label).item()
            prediction = output.max(1,keepdim = True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct /len(test_loader.dataset)

    return test_loss, test_accuracy

#Loss, acc 

for Epoch in range(1, EPOCHS+1):
    train(model, train_loader, optimizer, log_interval=200)
    test_loss, test_accuracy = evaluate(model, test_loader)
    print("\n[EPOCH: {}], \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(Epoch, test_loss,test_accuracy))
#  Test Loss: 0.0455,      Test Accuracy: 48.52 %
