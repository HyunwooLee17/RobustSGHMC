import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
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
batch_size=1000
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
T=10

class BNNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x),dim=1)



def step(model,lr,beta,p_buf,mh_train_data):
    model_old=deepcopy(model)
    rho=0
    i=0
    p_old_neg_buf=[]
    for p in model.parameters():
        p_old_neg_buf.append(-p_buf[i])
        p.data+=0.5*p_buf[i]
        AMAGOLD(p,lr,beta,rho,p_buf,i)
        p.data+=0.5*p_buf[i]
        i+=1
    # for data, target in mh_train_data:
    #     data, target = data.to(device), target.to(device)
    #     model.zero_grad()
    #     output = model(data)
    #     U_new = F.nll_loss(output, target)
    #     model_old.zero_grad()
    #     output = model_old(data)
    #     U_old = F.nll_loss(output, target)
    #     break
    
    # a=(U_old-U_new)*datasize+rho
    
    u=torch.log(torch.rand(1))
    # if u.item()<=a.data.item():
    #     print('success!')
    #     return model,p_buf
    # else:
    #     return model_old,p_old_neg_buf
    return model,p_buf

def AMAGOLD(p,lr,beta,rho,p_buf,i):
    for t in range(T):
        if t!=0:
            p.data+=p_buf[i]
        p_buf_old=deepcopy(p_buf[i])
        eta=torch.normal(mean=0,std=torch.ones_like(p)*np.sqrt(4*lr*beta)).to(device)
        U_p=p.grad.data
        U_p.add_(5e-4,p.data)
        p_buf[i]=((1-lr*beta)*p_buf_old-lr*U_p+eta)/(1+lr*beta)
        rho += torch.sum(U_p * (p_buf_old + p_buf[i]))*0.5
    
    

    

def train(model,epoch,traindata,mh_train_data):
    model.train()
    # traindata,testdata=getData(batch_size)
    # loss_fn=torch.nn.NLLLoss(reduction='mea)
    lr=0.01/datasize
    p_buf=[]
    for p in model.parameters():
        p_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))
    for batch_idx, (data,target) in enumerate(traindata):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        # print(output.size())
        # input()
        loss = F.nll_loss(output, target)*datasize
        # print(loss,len(data),datasize)
        # print(output,target)
        # loss=loss/len(data)*datasize
        # input()
        # 이러면 loss가 실질적으로 U(theta)의 역할을 한다
        loss.backward()
        model,p_buf=step(model,lr,0.5,p_buf,mh_train_data)
        if batch_idx%100==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(traindata.dataset),
                100. * batch_idx / len(traindata), loss.data.item()))

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


    mh_train_loader = DataLoader(dataset=mnist_train,
                                        batch_size=datasize, # 배치 크기는 100
                                        shuffle=True,
                                        drop_last=True)

    test_loader = DataLoader(dataset=mnist_test,
                                        batch_size=10000,
                                        shuffle=True,
                                        drop_last=True)
    return data_loader,test_loader,mh_train_loader

if __name__=="__main__":
    M=BNNmodel()
    M.to(device)
    traindata,testdata,mh_train_data=getData(batch_size)
    test(M,testdata,-1)
    for epoch in range(10):
        train(M, epoch,traindata,mh_train_data)
        test(M,testdata,epoch)