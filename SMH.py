import numpy as np
# import numpy.random.Generator as npgen
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.stats as stats
import math
import os

def effFMH(kernel,pi,psi,theta,theta_new,m):
    P=np.sum([psi(i) for i in range(m)])
    tau=[psi(i)/P for i in range(m)]

    N=np.random.poisson(pi(theta,theta_new)*P)
    for i in range(N):
        X=np.random.multinomial(m,tau)
        B=np.random.binomial(1,-np.log(kernel(theta,theta_new))/(pi(theta,theta_new)*psi(X)))
        if B==1:
            return theta
    return theta_new


def getData():
    data=pd.read_csv("diabetes2.csv",na_values = ['?', '??', 'N/A', 'NA', 'nan', 'NaN', '-nan', '-NaN', 'null'])
    x=data.iloc[:100,:-1].values
    y=data.iloc[:100,-1].values
    return x,y

def torch_regression(x,y):
    x_tensor=torch.from_numpy(x).float()
    y_tensor=torch.from_numpy(y).float()
    # print(x_tensor,y_tensor)
    model = nn.Sequential(
    nn.Linear(x_tensor.shape[1],1), # input_dim = 2, output_dim = 1
    nn.Sigmoid() # 출력은 시그모이드 함수를 거친다
    )

    # W=torch.rand((x_tensor.shape[1],1), requires_grad=True , dtype=x_tensor.dtype)
    # H=torch.sigmoid(x_tensor.matmul(W))
    # loss=F.binary_cross_entropy(H,y_tensor)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    nb_epochs = 10000
    for epoch in range(nb_epochs + 1):

        # Cost 계산
        H=model(x_tensor).squeeze(-1)
        loss=F.binary_cross_entropy(H,y_tensor)

        # cost로 H(x) 개선
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 100번마다 로그 출력
        if epoch % 100 == 0:
            prediction = H >= torch.FloatTensor([0.5]) # 예측값이 0.5를 넘으면 True로 간주
            correct_prediction = prediction.float() == y_tensor # 실제값과 일치하는 경우만 True로 간주
            accuracy = correct_prediction.sum().item() / len(correct_prediction) # 정확도를 계산
            print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format( # 각 에포크마다 정확도를 출력
                epoch, nb_epochs, loss.item(), accuracy * 100,
        ))
    for param in model.parameters():
        print(param)
    # print(model.parameters())

def Bayesian_regression_SMH(x,y,k):
    m=x.shape[0]
    num_param=x.shape[1]
    # print(m,num_param)
    theta=np.zeros(num_param,dtype=np.float64)

    # 1. Find theta_hat

    if os.path.isfile('thetaHat.pt'):
        theta_hat=torch.load('thetaHat.pt')
    else:
        theta_hat=torch.zeros(num_param,dtype=torch.float64,requires_grad=True)
        # print(theta_hat)
        def U(theta):
            res=torch.tensor(0.)
            for i in range(m):
                res+= torch.exp(theta.matmul(torch.from_numpy(x[i])))
                res+= torch.log(1+torch.exp(-1*theta.matmul(torch.from_numpy(x[i]))))
                res-= y[i]*theta.matmul(torch.from_numpy(x[i]))
            return res
            # print(torch.log(torch.tensor(2.)))
        optimizer = optim.Adam([theta_hat], lr=0.001)
        nb_epochs = 300
        for epoch in range(nb_epochs + 1):
            loss=U(theta_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(epoch,loss.item())
                print("theta^=",theta_hat)
        torch.save(theta_hat,'thetaHat.pt')
    
    print(theta_hat)

    # 2. define Kernel
    

    phi=lambda theta,theta_new: np.sum(np.abs(theta-theta_hat)**(k+1)+np.sum(np.abs(theta_new-theta_hat)**(k+1)))
    
    def U_bar(k,i):
        if k==1:
            return np.max(np.abs(x[i]))
        elif k==2:
            return np.max(np.abs(x[i])**2)/4
        elif k==3:
            return np.max(np.abs(x[i])**3)/(6*np.sqrt(3))

    psi=[U_bar(k+1,i)/math.factorial(k+1) for i in range(m)]
    

    
def Bayesian_regression(x,y):
    m=x.shape[0]
    num_param=x.shape[1]
    # print(m,num_param)
    theta=np.zeros(num_param,dtype=np.float64)
    print(theta)
    for epoch in range(1000):
        theta_new=np.random.default_rng().multivariate_normal(mean=theta,cov=np.identity(num_param,dtype=np.float64))
        res=0.
        for x_i,y_i in zip(x,y):
            res+=( -1*np.dot(theta_new,x_i) -np.log(1+np.exp(-1*np.dot(theta_new,x_i)))+y_i*np.dot(theta_new,x_i))
            res-=(-1*np.log(1+np.exp(np.dot(theta,x_i)))+y_i*np.dot(theta,x_i))
        # print(res)
        u=np.random.default_rng().uniform(0,1)
        if u<res:
            theta=theta_new

        if epoch%100==0 and epoch>200:
            print(theta)
    # print(theta_new)

if __name__=="__main__":
    # 1. Refine dataset
    x,y=getData()
    # torch_regression(x,y)
    Bayesian_regression(x,y)
    # Bayesian_regression_SMH(x,y,2)