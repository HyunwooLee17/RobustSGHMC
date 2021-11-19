import torch
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import random
import torch.distributions as dist
import numpy as np
from copy import deepcopy
from torch.linalg import matrix_norm

data_size=10000
test_size=5000
batch_size=2000
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
# T=10
weight_decay=5e-4
# temparature=1
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



def step(model,lr,beta,momentum_buf,traindata,mh_train_data,temparature,mh_correction,T):
    model_old=deepcopy(model)
    rho=0.0
    sig=0
    momentum_buf_old=deepcopy(momentum_buf)
    model,momentum_buf,rho=AMAGOLD(model,momentum_buf,lr,beta,traindata,temparature,T)
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
        a=(U_old-U_new)*data_size/temparature+rho
        u=torch.log(torch.rand(1))
        # print(u.item(),a.data.item(),rho)
        if u.item()<a.data.item():
            # print("accept")
            sig=1
            return sig,model,momentum_buf
        else:
            # print(len(momentum_buf_old),len(momentum_buf))
            return sig,model_old,[-1*m for m in momentum_buf_old]
    return sig,model,momentum_buf


def AMAGOLD(model,momentum_buf,lr,beta,traindata,temparature,T):
    rho=0.0
    t=-1
    while True:
        for data, target in traindata:
            data, target = data.to(device), target.to(device)
            if t>=0:
                model.zero_grad()
                output=model(data)
                U=F.nll_loss(output,target)*data_size
            
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
                    eta*=2*((lr*beta*temparature)**.5)
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
    
    

    

def train(model,traindata,mh_train_data,test_data,temparature,mh_correction,T=10):
    print('Train! temp={:.4f}, MH={}, LF step={} '.format(temparature,mh_correction,T))
    lr=0.0005/data_size

    beta=5e-6
    momentum_buf=[]
    model_vec=[deepcopy(model)]
    epoch=100
    s=0
    if T==1:
        epoch=2000
    for p in model.parameters():
        momentum_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))
    for i in range(epoch):
        if i%(20)==0:
            # print(s/(epoch/10))
            test(model_vec,test_data,i)
        sig,model,momentum_buf=step(model,lr,beta,momentum_buf,traindata,mh_train_data,temparature,mh_correction,T)
        s+=sig
        model_vec.append(deepcopy(model))
        # if i==50:
            # torch.save(model.state_dict(),'./burn_in.pt')
    test(model_vec,test_data,i)



def test(model_vec,testdata,epoch):
    test_loss=0
    correct=0
    F_norm=0
    with torch.no_grad():
        for data,target in testdata:
            output_vec=torch.zeros(1000,10).to(device)        
            data,target=data.to(device),target.to(device)    
            for model in model_vec:
                model.eval()
                output=model(data)
                output_vec += output
            output_vec/=len(model_vec)
            test_loss+=F.nll_loss(output_vec,target,reduction='sum')
            pred = output_vec.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    model=model_vec[-1]
    F_norm+=torch.norm(model.fc1.weight)**2
    F_norm+=torch.norm(model.fc2.weight)**2
    F_norm+=torch.norm(model.fc3.weight)**2
    test_loss /= len(testdata.dataset)
    L2_norm=0
    L2_norm+=matrix_norm(model.fc1.weight,ord=2)
    L2_norm+=matrix_norm(model.fc2.weight,ord=2)
    L2_norm+=matrix_norm(model.fc3.weight,ord=2)
    # print('\nIter: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch,
    #     test_loss, correct, len(testdata.dataset),
    #     100. * correct / len(testdata.dataset)))
    # print('And L2 norm is',L2_norm.data**(1/2))
    print('{} {:.4f} {:.2f} {:.4f} {:.4f}'.format(epoch,test_loss,100. * correct / len(testdata.dataset),F_norm.data**(1/2), L2_norm))
    # return output_vec

def getData(batch_size,MH_batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    data_loader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(data_size)),
        batch_size=batch_size, shuffle=True, **kwargs)
    mh_train_loader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(data_size)),
        batch_size=MH_batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(test_size)),
        batch_size=1000, shuffle=False, **kwargs)
    return data_loader,test_loader,mh_train_loader

if __name__=="__main__":
    USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
    device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
    torch.manual_seed(11)
    M=BNNmodel()
    M.to(device)
    traindata,testdata,mh_train_data=getData(batch_size,data_size)

    M.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    train(M, traindata,mh_train_data,testdata,1,True)

    M.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    train(M, traindata,mh_train_data,testdata,0.01,True)

    M.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    train(M, traindata,mh_train_data,testdata,1/50000**(1/2),True)

    M.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    train(M, traindata,mh_train_data,testdata,0.0001,True)
