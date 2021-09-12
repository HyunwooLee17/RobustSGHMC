import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.distributions as dist
T=10
D=100


def U(theta,x,y,minibatch=False):
    if minibatch:
        idx=np.random.choice(x.shape[0],D,replace=False)
    else:
        idx=list(range(x.shape[0]))
    res=torch.tensor(0.,requires_grad=True)
    for i in idx:
        res.data+=U_i(theta,i,x,y)
    res.data*=x.shape[0]/len(idx)
    return res
    

def U_i(theta,i,x,y):
    # if i==3:
        # print("24",x[i])
        # print("25",y[i])
        # print("26",theta.matmul(x[i]))
    # if theta.matmul(x[i])>100:
    #     return theta.matmul(x[i])-y[i]*theta.matmul(x[i])
        
    return (theta.matmul(x[i]))-y[i]*theta.matmul(x[i])

def U_grad(theta,x,y,minibatch=False):
    if minibatch:
        idx=np.random.choice(x.shape[0],D,replace=False)
    else:
        idx=list(range(x.shape[0]))
    my_grad=torch.zeros(x.shape[1],dtype=torch.double)
    for i in idx:
        res=U_i(theta,i,x,y)
        res.backward()
        my_grad+=theta.grad
        # if i==8:
        # print("41",res)
        # print("42",theta.grad,i)
        # print("37",my_grad,i)
        theta.grad.zero_()
    return my_grad

def U_grad_manual(theta,x,y,minibatch=False):
    if minibatch:
        idx=np.random.choice(x.shape[0],D,replace=False)
    else:
        idx=list(range(x.shape[0]))
    res=torch.zeros(x.shape[1],dtype=torch.double)
    for i in idx:
        for j in range(x.shape[1]):
            res[j]+=x[i][j]-y[i]*x[i][j]
    return res

def AMAGOLD(theta,eps,sigma,friction,dim,x,y,r=None):
    if r==None:
        r=dist.MultivariateNormal(torch.zeros(dim),sigma*torch.eye(dim)).sample().double()
    r_old=r
    rho=0

    theta_new=theta+0.5*eps*r_old/sigma
    for t in range(T):
        # print("r_old:",r_old)
        if t!=0:
            theta_new=theta_new+eps*r_old/sigma
        eta=dist.MultivariateNormal(torch.zeros(dim),4*eps*friction*sigma*torch.eye(dim)).sample().double()
        # theta_new.requires_grad_(True)
        # theta_new.retain_grad()
        U_tilde_grad=U_grad_manual(theta_new,x,y,minibatch=True)
        # theta_new.grad.zero_()
        # print("U_grad:",U_tilde_grad)
        r_new=((1-eps*friction)*r_old - eps*U_tilde_grad+eta)/(1+eps*friction)
        rho=rho+1/2*eps*U_tilde_grad.matmul(r_old+r_new)/sigma
        r_old=r_new
    theta_new=theta_new+0.5*eps*r_old/sigma
    a_top=U(theta,x,y)
    a_bot=U(theta_new,x,y)
    alpha=(a_top-a_bot+rho)
    print(a_top,a_bot,rho)
    # print("alpha:",alpha)
    u=torch.log(torch.rand(1))
    if u.data<alpha.data:
        return theta_new,r_new
    else:
        return theta,r

def getData():
    data=pd.read_csv("diabetes2.csv",sep=',',na_values = ['?', '??', 'N/A', 'NA', 'nan', 'NaN', '-nan', '-NaN', 'null'])
    x=torch.from_numpy(data.iloc[:,:-1].to_numpy(dtype=np.float64))
    y=torch.from_numpy(data.iloc[:,-1].to_numpy(dtype=np.float64))
    return x,y

def Bayesian_regression(x,y):
    # print(x.shape[1])
    dim=x.shape[1]
    # theta=dist.MultivariateNormal(torch.zeros(dim),torch.eye(dim)).sample().double()
    # theta.requires_grad_(True)
    theta=torch.tensor([ 0.2647,  1.1312, -0.4927, -0.7948,  1.5214, -0.7275, -0.6821,  0.1458],dtype=torch.float64, requires_grad=True)
    # print(theta)
    
    
    r=dist.MultivariateNormal(torch.zeros(dim),torch.eye(dim)).sample().double()
    for epoch in range(1000):
        theta,r=AMAGOLD(theta,0.25,1,0.25,x.shape[1],x,y,r)
        print(epoch,theta)
    # print(U_grad(theta,x,y))
    # print(U_grad_manual(theta,x,y))
    # print(theta)


if __name__=="__main__":
    # 1. Refine dataset
    x,y=getData()
    # torch_regression(x,y)
    Bayesian_regression(x,y)
    # Bayesian_regression_SMH(x,y,2)