import numpy as np
import pandas as pd
import scipy.stats as stats
T=10
D=100


def U(theta,x,y,minibatch=False):
    if minibatch:
        idx=np.random.choice(x.shape[0],D,replace=False)
        batch=zip(x[idx],y[idx])
    else:
        batch=zip(x,y)
    res=-1*np.sum([stats.multivariate_normal.logpdf(y_item,mean=np.dot(theta,x_item)) for (x_item,y_item) in batch])
    res*=x.shape[0]
    res/=D
    return res

def U_grad(theta,x,y,minibatch=False):
    if minibatch:
        idx=np.random.choice(x.shape[0],D,replace=False)
        batch=zip(x[idx],y[idx])
    else:
        batch=zip(x,y)
    res=[1/stats.multivariate_normal.logpdf(y_item,mean=np.dot(theta,x_item)) for (x_item,y_item) in batch]
    res=np.multiply(res,-1*x.shape[0]/D)
    return res



def AMAGOLD(U,theta,eps,sigma,friction,dim,x,y,U_grad):
    r=np.random.default_rng().multivariate_normal(mean=np.zeros(dim),cov=sigma*np.identity(dim))
    
    rho =0
    r_old=r

    theta_new=theta+0.5*eps/sigma*r_old
    for t in range(T):
        if t!=0:
            theta_new=theta_new+eps*r_old/sigma
        
        eta=np.random.default_rng().multivariate_normal(mean=np.zeros(dim),cov=4*eps*friction*sigma*np.identity(dim))
        U_tilde_grad=U_grad(theta_new,x,y,minibatch=True)
        print(r_old,U_tilde_grad)
        r_new=((1-eps*friction)*r_old-eps*U_tilde_grad+eta)/(1+eps*friction)
        rho=rho+0.5*eps/sigma*np.dot(U_tilde_grad,(r_old+r_new))
        r_old=r_new
    
    theta_new=theta_new+0.5*eps/sigma*r_old

    a=U(theta,x,y)-U(theta_new,x,y)+rho
    # print(U(theta,x,y),U(theta_new,x,y),rho)
    if np.random.default_rng().uniform(0,1)<np.exp(a):
        return (theta_new,r_new)
    else:
        return (theta,-1*r)

def getData():
    data=pd.read_csv("winequality-red.csv",sep=';',na_values = ['?', '??', 'N/A', 'NA', 'nan', 'NaN', '-nan', '-NaN', 'null'])
    x=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values
    return x,y

def Bayesian_regression(x,y):
    theta=np.random.default_rng().multivariate_normal(mean=np.zeros(x.shape[1]),cov=np.identity(x.shape[1]))
    # print(x.shape[1])
    # print(theta)
    for epoch in range(1000):
        theta=AMAGOLD(U,theta,0.25,1,0.25,x.shape[1],x,y,U_grad)




if __name__=="__main__":
    # 1. Refine dataset
    x,y=getData()
    # torch_regression(x,y)
    Bayesian_regression(x,y)
    # Bayesian_regression_SMH(x,y,2)