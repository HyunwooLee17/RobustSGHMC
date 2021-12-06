import torch.nn as nn
import torch.nn.functional as F
import torch
import hamiltorch
import matplotlib.pyplot as plt
import torchvision.datasets as dsets
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.linalg import matrix_norm

hamiltorch.set_random_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return F.log_softmax(self.fc1(x), dim=1)

net = Net()


mnist_trainset = datasets.MNIST(root='../../data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='../../data', train=False, download=True, transform=None)
# plt.imshow(mnist_trainset.train_data[0].reshape((28,28)))
# plt.show()
D = 28*28
N_tr = 100
N_val = 1000


x_train = mnist_trainset.train_data[:N_tr].float()
x_train = x_train[:,None]
y_train = mnist_trainset.train_labels[:N_tr].reshape((-1,1)).float()
x_val = mnist_trainset.train_data[N_tr:N_tr+N_val].float()
x_val = x_val[:,None]
y_val = mnist_trainset.train_labels[N_tr:N_tr+N_val].reshape((-1,1)).float()

x_train = x_train.to(device)
y_train = y_train.to(device)
x_val = x_val.to(device)
y_val = y_val.to(device)

tau_list = []
tau = 10.#./100. # 1/50
for w in net.parameters():
    print(w.nelement())
    # tau_list.append(tau/w.nelement())
    tau_list.append(tau)
tau_list = torch.tensor(tau_list).to(device)

hamiltorch.set_random_seed(123)
net = Net()
params_init = hamiltorch.util.flatten(net).to(device).clone()
# print(params_init.shape)

step_size = 0.001#0.01# 0.003#0.002
num_samples = 100#2000 # 3000
L = 3 #3
tau_out = 1.
normalizing_const = 1.
burn =30 #GPU: 3000

params_hmc = hamiltorch.sample_model(net, x_train, y_train, params_init=params_init, model_loss='multi_class_linear_output', num_samples=num_samples, burn = burn,
                            step_size=step_size, num_steps_per_sample=L,tau_out=tau_out, tau_list=tau_list, normalizing_const=normalizing_const)


for p in params_hmc[burn:]:
    m=p[:28*28*10].reshape(28*28,10)
    # print(m)
    print(matrix_norm(m,ord=2))
