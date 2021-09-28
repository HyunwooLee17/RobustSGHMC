import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.distributions as dist
import math
T=10
D=100




def getData():
    data=pd.read_csv("diabetes2.csv",sep=',',na_values = ['?', '??', 'N/A', 'NA', 'nan', 'NaN', '-nan', '-NaN', 'null'])
    x=torch.from_numpy(data.iloc[:,:-1].to_numpy(dtype=np.float32))
    y=torch.from_numpy(data.iloc[:,-1].to_numpy(dtype=np.float32))
    return x,y