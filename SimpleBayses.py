import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
def getData():
    data=pd.read_csv("diabetes2.csv",sep=',',na_values = ['?', '??', 'N/A', 'NA', 'nan', 'NaN', '-nan', '-NaN', 'null'])
    x=torch.from_numpy(data.iloc[:,:-1].to_numpy())
    y=torch.from_numpy(data.iloc[:,-1].to_numpy())
    return x,y


def model(theta,x):
    return nn.sigmoid(theta.matmul(x))


def Bayesian_regression(x,y):
    m=x.shape[0]
    num_param=x.shape[1]
    # print(m,num_param)
    theta=torch.zeros(num_param)
    # print(theta)
    M=dist.multivariate_normal.MultivariateNormal(torch.zeros(num_param),torch.eye(num_param))
    num_epochs=100
    for _ in range(num_epochs):
        theta_new=M.sample()
        print(theta_new)

if __name__=="__main__":
    # 1. Refine dataset
    x,y=getData()
    # torch_regression(x,y)
    Bayesian_regression(x,y)
    # Bayesian_regression_SMH(x,y,2)