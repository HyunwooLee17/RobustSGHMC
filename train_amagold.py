from __future__ import print_function
import argparse
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from copy import deepcopy
import sys
import pickle
from torch.utils.data import Subset
from torch.linalg import matrix_norm
from Amagold import AMAGOLD

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x),dim=1)

def test(model_vec, it):
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output_vec=torch.zeros(1000,10).to(device)
            data, target = data.to(device), target.to(device)
            for model in model_vec:
                model.eval()
                output=model(data)
                output_vec+=output
            # output = model(data)
            output_vec/=len(model_vec)
            test_loss += F.nll_loss(output, target, reduction='sum').data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    F_norm=0
    model=model_vec[-1]
    F_norm+=torch.norm(model.fc1.weight)**2
    F_norm+=torch.norm(model.fc2.weight)**2
    F_norm+=torch.norm(model.fc3.weight)**2
    test_loss /= len(test_loader.dataset)
    L2_norm=0
    L2_norm+=matrix_norm(model.fc1.weight,ord=2)
    L2_norm+=matrix_norm(model.fc2.weight,ord=2)
    L2_norm+=matrix_norm(model.fc3.weight,ord=2)
    # print('\nIter: {}, Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(it,
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))

    print('{} {:.4f} {:.2f} {:.4f} {:.4f}'.format(it,test_loss,100. * correct / len(test_loader.dataset),F_norm.data**(1/2), L2_norm))


def run(optimizer,lr):
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    succ=0
    p_buf = []
    for p in model.parameters():
        p_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))
    model_vec=[model]
    test(model_vec, 0)
    for it in range(1, 101):
        sig,model,p_buf = optimizer.outer_loop(model,p_buf,lr)
        succ += sig
        model_vec.append(model)
        if it%20==0:
            test(model_vec, it)
            # print(1.0*succ/it)
    print('=================================================================')
    
def run_langevin(optimizer,lr):
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    succ=0
    p_buf = []
    for p in model.parameters():
        p_buf.append(torch.zeros(p.size()).to(device)*np.sqrt(lr))
    model_vec=[model]
    test(model_vec, 0)
    for it in range(1, 101):
        sig,model,p_buf = optimizer.outer_loop(model,p_buf,lr)
        succ += sig
        model_vec.append(model)
        if it%20==0:
            test(model_vec, it)
            # print(1.0*succ/it)
    print('=================================================================')
    

def run_noBias(optimizer,lr):
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    model.fc1.bias.data.fill_(0.0)
    model.fc2.bias.data.fill_(0.0)
    model.fc3.bias.data.fill_(0.0)
    succ=0
    p_buf = []
    for p in model.parameters():
        p_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))
    model_vec=[model]
    test(model_vec, 0)
    for it in range(1, 101):
        sig,model,p_buf = optimizer.outer_loop(model,p_buf,lr)
        model.fc1.bias.data.fill_(0.0)
        model.fc2.bias.data.fill_(0.0)
        model.fc3.bias.data.fill_(0.0)
        model_vec.append(model)
        if it%20==0:
            test(model_vec, it)
            # print(1.0*succ/it)
    print('=================================================================')

def run_norm(optimizer,lr):
    model = Net()
    model.to(device)
    model.load_state_dict(torch.load('./checkpoints/sgd_init_epoch10.pt'))
    succ=0
    p_buf = []
    for p in model.parameters():
        p_buf.append(torch.randn(p.size()).to(device)*np.sqrt(lr))
    model_vec=[model]
    test(model_vec, 0)
    for it in range(1, 101):
        sig,model,p_buf = optimizer.outer_loop(model,p_buf,lr)
        succ += sig
        model_vec.append(model)
        if it%20==0:
            test(model_vec, it)
            # print(1.0*succ/it)
    print('=================================================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=50, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--beta', type=float, default=5e-6, metavar='M',
                        help='beta')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    print('lr',args.lr,'beta',args.beta)
    datasize = 10000
    test_size = 10000
    batch_size= 2000
    torch.manual_seed(args.seed)
    kwargs = {'num_workers': 4, 'pin_memory': True}
    trainloader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(datasize)),
        batch_size=batch_size, shuffle=True, **kwargs)
    mh_trainloader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(datasize)),
        batch_size=datasize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(test_size)),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    weight_decay = 5e-4
    T = 10
    lr = args.lr/datasize
    criterion = nn.NLLLoss()
    
    
    print('\n Adjusting data size\n')
    datasize = 10000
    test_size = 10000
    batch_size= 2000
    trainloader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(datasize)),
        batch_size=batch_size, shuffle=True, **kwargs)
    mh_trainloader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(datasize)),
        batch_size=datasize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        Subset(datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),range(test_size)),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    print('-----------------')
    print('langevin dynamics')
    print('------------------')
    mh=False
    lr=0.1/datasize
    opt_1 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, 1, criterion,1,mh)
    opt_2 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, 1, criterion,0.01,mh)
    opt_3 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, 1, criterion,0.0001,mh)
    opt_4 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, 1, criterion,1e-6,mh)
    run_langevin(opt_1,lr)
    run_langevin(opt_2,lr)
    run_langevin(opt_3,lr)
    run_langevin(opt_4,lr)
    # print('-----------------')
    # print('with full mh correction')
    # print('------------------')
    # mh=True
    # opt_1 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,1,mh)
    # opt_2 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,0.01,mh)
    # opt_3 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,0.0001,mh)
    # opt_4 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,1e-6,mh)
    # run(opt_1)
    # run(opt_2)
    # run(opt_3)
    # run(opt_4)
    print('-----------------')
    print('with stochastic mh correction')
    print('------------------')
    # mh=True
    # mh_trainloader = torch.utils.data.DataLoader(
    #     Subset(datasets.MNIST('../data', train=True, download=True,
    #                 transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ])),range(datasize)),
    #     batch_size=batch_size, shuffle=True, **kwargs)
    # opt_1 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,1,mh)
    # opt_2 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,0.01,mh)
    # opt_3 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,0.0001,mh)
    # opt_4 = AMAGOLD(trainloader, mh_trainloader, datasize, T, weight_decay, args.beta, criterion,1e-6,mh)
    # run(opt_1)
    # run(opt_2)
    # run(opt_3)
    # run(opt_4)