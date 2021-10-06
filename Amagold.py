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
weight_decay=5e-4
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



def step(model,lr,beta,momentum_buf,traindata,mh_train_data,mh_correction=True):
    model_old=deepcopy(model)
    rho=0.0
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
        # print(u.item(),a.data.item())
        if u.item()<a.data.item():
            # print("accept")
            return model,momentum_buf
        else:
            # print(len(momentum_buf_old),len(momentum_buf))
            return model_old,[-1*m for m in momentum_buf_old]
    return model,momentum_buf


def AMAGOLD(model,momentum_buf,lr,beta,traindata):
    rho=0.0
    t=-1
    while True:
        for data, target in traindata:
            data, target = data.to(device), target.to(device)
            if t>=0:
                model.zero_grad()
                output=model(data)
                U=F.nll_loss(output,target)*datasize
                U.backward()
            i=0
            
            for p in model.parameters():
                if t==-1:
                    p.data+=0.5*momentum_buf[i]
                else:
                    p.data+=momentum_buf[i]
                    U_grad=p.grad.data
                    U_grad.add_(weight_decay, p.data)
                    eta=torch.randn(p.size()).to(device)
                    eta*=2*((lr*beta)**.5)
                    momentum_buf_old=deepcopy(momentum_buf[i])
                    momentum_buf[i]=((1-beta)*momentum_buf[i]-lr*U_grad+eta)/(1+beta)
                    rho += torch.sum(U_grad * (momentum_buf_old + momentum_buf[i]))
                    # print(rho)
                    if t==T-1:
                        p.data += 0.5*momentum_buf[i]
                i+=1
            t=t+1
            if t==T:
                break
        if t==T:
            break
    return model,momentum_buf,0.5*rho
    
    

    

def train(model,traindata,mh_train_data,test_data):
    lr=0.0005/datasize
    beta=5e-6
    momentum_buf=[]
    model_vec=[model]
    output_vec=torch.zeros(1000,10).to(device)
    for p in model.parameters():
        momentum_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))
    for i in range(200):
        if i%10==0:
            test(model_vec,test_data,i,output_vec)
            model_vec=[]
        model,momentum_buf=step(model,lr,beta,momentum_buf,traindata,mh_train_data)
        model_vec.append(model)
    test(model_vec,test_data,i,output_vec)

def test(model_vec,testdata,epoch,output_vec):
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in testdata:
            data, target = data.to(device), target.to(device)
            for model in model_vec:
                model.eval()
                output=model(data)
                output_vec += output
            # print(output_vec)
            output_vec/=(epoch+1)
            # output_tensor=torch.stack(output_vec)
            # print(output_tensor)
            # output=torch.mean(output_tensor,2,True)
            # print(output)
            test_loss+=F.nll_loss(output_vec,target,reduction='sum')
            pred = output_vec.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testdata.dataset)
    print('\nIter: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch,
        test_loss, correct, len(testdata.dataset),
        100. * correct / len(testdata.dataset)))
    return output_vec

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
    M.load_state_dict(torch.load('./checkpoints/sgd_init_epoch3.pt'))
    torch.manual_seed(11)
    traindata,testdata,mh_train_data=getData(batch_size)
    train(M, traindata,mh_train_data,testdata)