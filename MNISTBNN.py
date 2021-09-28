import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import torch.distributions as dist
import torchbnn as bnn


batch_size=100
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
T=10

class BNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x),dim=1)

def step(model,lr):
    for p in model.parameters():
        print(p)
        d_p=p.grad.data


def train(model):
    traindata,testdata=getData(100)
    loss_fn=torch.nn.NLLLoss(reduction='sum')
    for data,target in traindata:
        # data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss=loss_fn(output,target)-1
        # 이러면 loss가 실질적으로 U(theta)의 역할을 한다
        loss.backward()
        step(model)

def getData(batch_size):
    mnist_train = dsets.MNIST(root='MNIST_data/',
                        train=True,
                        transform=transforms.ToTensor(),
                        download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)
    



    data_loader = DataLoader(dataset=mnist_train,
                                        batch_size=batch_size, # 배치 크기는 100
                                        shuffle=True,
                                        drop_last=True)

    test_loader = DataLoader(dataset=mnist_test,
                                        batch_size=10000,
                                        shuffle=True,
                                        drop_last=True)
    return data_loader,test_loader

if __name__=="__main__":
    M=BNNmodel()
    train(M)
