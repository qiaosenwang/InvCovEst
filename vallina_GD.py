import torch
import numpy as np
import torch.nn as nn
from My_optim import GD
import matplotlib.pyplot as plt

def L1_penal(x):
    dim = list(x.size())[0]
    out = 0
    for i in range(1, dim):
        out += torch.sum(torch.abs(torch.diag(x, i))) + torch.sum(torch.abs(torch.diag(x, -i)))
    return out


class net(nn.Module):
    def __init__(self, dim, diag, beta):
        super(net, self).__init__()
        v = torch.normal(torch.zeros(dim), 0.1)
        self.X = torch.matmul(v.t(), v) + diag * torch.eye(dim,dtype=torch.float32)
        self.X.requires_grad = True
        self.beta = beta

    def forward(self, S):
        return -torch.logdet(self.X) + torch.sum(S * self.X) + self.beta * L1_penal(self.X)

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


def main():

    mean = torch.tensor([1,2, 3], dtype=torch.float32)
    cov = torch.tensor([[2, 0, 1],[0, 1, 0],[1, 0, 4]], dtype=torch.float32)
    truth = torch.inverse(cov)
    x = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)
    #print(x.sample())
    n = 10000
    d = list(x.sample().size())[0]
    eps =0.0001


    S = sample_cov(n, x)
    print(S)
    R = torch.inverse(S)
    print(R)

    NLogL = net(dim=d, diag=0.5, beta=0.05)

    param_groups = []
    param_groups.append({'params':[NLogL.X], 'dimension':d})
    optimizer = GD(params=param_groups, lr=0.1, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.1)

    performance = dict()
    performance['loss'] = []
    performance['iteration'] = []

    max_iter = 2000
    for i in range(1, max_iter+1):
        out = NLogL.forward(S)
        optimizer.zero_grad()
        out.backward()
        optimizer.step()
        scheduler.step()

        if i % 5 == 0:
            loss = torch.norm(NLogL.X.clone().detach()-truth).item()
            performance['loss'].append(loss)
            performance['iteration'].append(i)
            if i % 100 == 0:
                print("iteration={}".format(i))
                print("Frobenius loss={}".format(loss))
                #print("-log_likelihood={}".format(loss.item()))
                #print("gradient={}".format(NLogL.X.grad))

    NLogL.X.requires_grad = False
    Y = NLogL.X
    print("estimation={}".format(Y))
    zero = torch.zeros(d, d)
    print("censored={}".format(Y.where(abs(Y) > eps, zero)))
    print("true_precision={}".format(truth))
    #print("sample_cov={}".format(S))
    print("direct_inverse={}".format(R))

    plt.plot(performance['iteration'], performance['loss'])
    plt.show()

if __name__ == '__main__':
    main()