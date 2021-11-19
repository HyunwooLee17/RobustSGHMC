from copy import deepcopy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import random
import torch.distributions as dist
import torchbnn as bnn
import numpy as np

data_size=10000
test_size=5000
batch_size=1000
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

def step(model,lr,weight_decay,alpha,momentum_buf,temparature=1):
    i=0
    for p in model.parameters():
        # if not hasattr(p,'buf'):
            # p.buf = torch.randn(p.size()).to(device)*np.sqrt(lr)
        d_p = p.grad.data
        d_p.add_(weight_decay, p.data)
        eps = torch.randn(p.size()).to(device)
        buf_new = (1-alpha)*momentum_buf[i] - lr*d_p + (2.0*lr*alpha*temparature)**.5*eps
        p.data.add_(momentum_buf[i])
        momentum_buf[i] = buf_new
        i+=1
    return model,momentum_buf


def train(model,traindata,mh_data,temparature,momentum_buf):
    model.train()
    lr=0.1
    rho=0.0
    model_old=deepcopy(model)
    momentum_buf_old=deepcopy(momentum_buf)
    for batch_idx, (data,target) in enumerate(traindata):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)*data_size
        # 이러면 loss가 실질적으로 U(theta)의 역할을 한다
        loss.backward()
        model,momentum_buf=step(model,lr=0.1/data_size,weight_decay=5e-4,alpha=0.01,momentum_buf=momentum_buf,temparature=temparature)
    model_new=deepcopy(model)
    momentum_buf_new=deepcopy(momentum_buf)
    a=MHstep(model_old,model_new,mh_data,momentum_buf_old,momentum_buf_new,temparature)
    u=torch.log(torch.rand(1))
    if u.item()<a.data.item():
        print("accept")
        return model,momentum_buf
    else:

        return model_old,[-1*m for m in momentum_buf_old]

def MHstep(model_old,model_new,mh_data,temparature):
    for data, target in mh_data:
        data, target = data.to(device), target.to(device)
        model_new.zero_grad()
        output = model_new(data)
        U_new = F.nll_loss(output, target)/temparature*data_size
        model_old.zero_grad()
        output = model_old(data)
        U_old = F.nll_loss(output, target)/temparature*data_size
        break
    H_new=U_new+
    return (U_old-U_new)*data_size


def test(model_vec,testdata,epoch):
    # model.eval()
    test_loss=0
    correct=0
    L2_norm=0
    with torch.no_grad():
        for data,target in testdata:
            data, target = data.to(device), target.to(device)
            output_vec=torch.zeros(1000,10).to(device)       
            for model in model_vec:
                model.eval()
                output=model(data)
                output_vec += output
            output_vec/=len(model_vec)
            test_loss+=F.nll_loss(output_vec,target,reduction='sum')
            pred = output_vec.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(testdata.dataset)
    model=model_vec[-1]
    L2_norm+=torch.norm(model.fc1.weight)**2
    L2_norm+=torch.norm(model.fc2.weight)**2
    L2_norm+=torch.norm(model.fc3.weight)**2
    print('\nIter: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch,
        test_loss, correct, len(testdata.dataset),
        100. * correct / len(testdata.dataset)))
    # print('And L2 norm is',L2_norm.data)
    print('{} {:.4f} {:.2f} {:.4f}\n'.format(epoch,test_loss,100. * correct / len(testdata.dataset),L2_norm.data**(1/2)))

def getData(batch_size):
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
        batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(test_size)),
        batch_size=1000, shuffle=False, **kwargs)
    return data_loader,test_loader,mh_train_loader

def run(temparature):
    M=BNNmodel()
    M.to(device)
    traindata,testdata,mh_data=getData(batch_size)
    Model_vec=[M]
    momentum_buf=[torch.randn(p.size()).to(device)*np.sqrt(0.1) for p in M.parameters()]
    for epoch in range(100):
        M_temp=deepcopy(M)
        Model_vec.append(M_temp)
        if epoch%10==0:
            test(Model_vec,testdata,epoch)
        M,momentum_buf=train(M,traindata,mh_data,temparature,momentum_buf)



if __name__=="__main__":
    torch.manual_seed(11)
    run(1)