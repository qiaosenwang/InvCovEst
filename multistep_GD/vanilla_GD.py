import torch
import numpy as np
import torch.nn as nn
from .My_optim import GD
from .Gaussian_generator import Gaussian_Distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from plot_heatmap import heatmap

def L1_penal(x):
    dim = list(x.size())[0]
    out = 0
    #y = torch.zeros_like(x)
    #z = x.where(abs(x)>1e-5, y)
    for i in range(1, dim):
        out += torch.sum(torch.abs(torch.diag(x, i))) + torch.sum(torch.abs(torch.diag(x, -i)))
    return out


class net(nn.Module):
    def __init__(self, dim, diag, beta):
        super(net, self).__init__()
        v = torch.normal(torch.zeros(dim), 0.1)
        self.X = torch.matmul(v.t(), v) + diag * torch.eye(dim, dtype=torch.float32)
        self.X.requires_grad = True
        self.beta = beta

    def forward(self, S):
        return -torch.logdet(self.X) + torch.sum(S * self.X) + self.beta * L1_penal(self.X)
'''
def sample_cov(sample_size, distribution):
    samples = []
    dim = list(distribution.sample().size())[0]
    mean = torch.zeros(1, dim, dtype=torch.float32)
    S = torch.zeros(dim, dim, dtype=torch.float32)
    for i in range(sample_size):
        x = distribution.sample()
        samples.append(x)
        mean += x / sample_size
    for i in range(sample_size):
        S += torch.matmul((samples[i] - mean).t(), samples[i] - mean) / sample_size
    return S
'''

def main():

    mean = torch.tensor(np.ones(32), dtype=torch.float32)
    diag = torch.tensor(np.ones(32), dtype=torch.float32)

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=0.25, type='chain', slash=1)
    #truth = torch.inverse(population.cov)
    truth = population.invcov
    #print(x.sample())
    n = 1000
    d = population.dim
    eps = 1e-4

    print(truth)
    _, _, _, S = population.generate(n)
    print(S)

    R = torch.inverse(S)
    print(R)
    x = torch.ones(3,3)
    x.requires_grad = True
    l = L1_penal((x))
    l.backward()
    print(x.grad)



    NLogL = net(dim=d, diag=1.0, beta=0.05)

    param_groups = []
    param_groups.append({'params': [NLogL.X]})
    optimizer = GD(params=param_groups, lr=0.5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 120, 300, 600], gamma=0.1)

    performance = dict()
    performance['loss'] = []
    performance['iteration'] = []

    max_iter = 800
    for i in range(1, max_iter+1):
        out = NLogL.forward(S)
        optimizer.zero_grad()
        out.backward()
        optimizer.step()
        scheduler.step()

        if i % 1 == 0:
            loss = torch.norm(NLogL.X.clone().detach()-truth).item()
            performance['loss'].append(loss)
            performance['iteration'].append(i)
            if i % 100 == 0:
                print("iteration={}".format(i))
                print("Frobenius loss={}".format(loss))
                #print("-log_likelihood={}".format(loss.item()))
                #print("gradient={}".format(NLogL.X.grad))

    NLogL.X.requires_grad = False
    zero = torch.zeros(d, d)
    Y = NLogL.X
    Y_c = Y.where(abs(Y) > eps, zero)
    R_c = R.where(abs(R) > eps, zero)
    print("estimation={}".format(Y))

    print("censored={}".format(Y_c))
    print("true_precision={}".format(truth))
    print(population.cov)
    print("sample_cov={}".format(S))
    print("direct_inverse={}".format(R))

    #plt.plot(performance['iteration'], performance['loss'])
    #plt.show()
    font_setting = {'fontsize': 5}

    sns.heatmap(torch.abs(truth).numpy(), annot=truth.numpy(), annot_kws=font_setting, cmap="YlGn", vmin=0, vmax=0.5, square=True)
    plt.show()
    sns.heatmap(torch.abs(Y_c).numpy(), annot=Y_c.numpy(), annot_kws=font_setting, cmap="YlGn", vmin=0, vmax=0.5, square=True)
    plt.show()
    sns.heatmap(torch.abs(R_c).numpy(), annot=R_c.numpy(), annot_kws=font_setting, cmap="YlGn", vmin=0, vmax=0.5, square=True)
    plt.show()
    heatmap(Y_c.numpy())

if __name__ == '__main__':
    main()