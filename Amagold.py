import torch
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import random
import torch.distributions as dist
import torchbnn as bnn
import numpy as np
from copy import deepcopy

datasize=60000
batch_size=2000
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
T=10

class BNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500,256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x),dim=1)



def step(model,lr,beta,momentum_buf,traindata,mh_train_data,mh_correction=False):
    model_old=deepcopy(model)
    rho=0
    momentum_buf_old=deepcopy(momentum_buf)
    model,momentum_buf,rho=AMAGOLD(model,momentum_buf,lr,beta,traindata)
    if mh_correction:
        for data, target in mh_train_data:
                data, target = data.to(device), target.to(device)
                model.zero_grad()
                output = model(data)
                U_new = F.nll_loss(output, target)
                model_old.zero_grad()
                output = model_old(data)
                U_old = F.nll_loss(output, target)
                break
        # MH correction step
        a=(U_old-U_new)*datasize+rho
        u=torch.log(torch.rand(1))
        if u.item()<a.data.item():
            return model,momentum_buf
        else:
            return model_old,-momentum_buf_old
    return model,momentum_buf


def AMAGOLD(model,momentum_buf,lr,beta,traindata):
    rho=0
    for t in range(T):
        for data, target in traindata:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output=model(data)
            U=F.nll_loss(output,target)*datasize
            U.backward()
            i=0
            
            for p in model.parameters():
                momentum_buf_old=deepcopy(momentum_buf[i])
                if t==0:
                    p.data+=0.5*momentum_buf[i]
                else:
                    p.data+=momentum_buf[i]
                U_grad=p.grad.data
                eta=torch.randn(p.size()).to(device)
                eta*=2*((lr*beta)**.5)
                momentum_buf[i]=((1-beta)*momentum_buf[i]-lr*U_grad+eta)/(1+beta)
                rho += torch.sum(U_grad * (momentum_buf_old + momentum_buf[i])) 
                if t==T-1:
                    p.data += 0.5*momentum_buf[i]
                i+=1
            break
    return model,momentum_buf,rho
    
    

    

def train(model,traindata,mh_train_data,test_data):
    lr=0.0005/datasize
    beta=5e-6
    momentum_buf=[]
    for p in model.parameters():
        momentum_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))
    for i in range(20):
        if i%1==0:
            test(model,test_data,i)
        model,momentum_buf=step(model,lr,beta,momentum_buf,traindata,mh_train_data)

def test(model,testdata,epoch):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in testdata:
            data, target = data.to(device), target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output,target,reduction='sum')
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testdata.dataset)
    print('\nIter: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch,
        test_loss, correct, len(testdata.dataset),
        100. * correct / len(testdata.dataset)))

def getData(batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    data_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    mh_train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=datasize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=1000, shuffle=False, **kwargs)
    return data_loader,test_loader,mh_train_loader

if __name__=="__main__":
    M=BNNmodel()
    M.to(device)
    traindata,testdata,mh_train_data=getData(batch_size)
    train(M, traindata,mh_train_data,testdata)