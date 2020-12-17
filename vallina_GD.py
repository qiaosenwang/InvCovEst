import torch
import numpy as np
import torch.nn as nn
from My_optim import SGD

class net(nn.Module):
    def __init__(self, dim, diag):
        super(net, self).__init__()
        self.X = torch.tensor(np.random.normal(0,1,(1,1)),dtype=torch.float32) * torch.ones(dim,dim, dtype=torch.float32) + diag * torch.eye(dim,dtype=torch.float32)
        self.X.requires_grad =True

    def forward(self, S):
        return -torch.logdet(self.X) + torch.sum(S * self.X)+ 0.001 * torch.sum(torch.abs(self.X))

def main():

    d = 5
    samples = []
    sample_size = 5000
    mean = torch.zeros(d,1)
    S = torch.zeros(d,d)
    for i in range(1, sample_size+1):
        x = torch.tensor(np.random.normal(0, 4, (d, 1)), dtype=torch.float32)
        samples.append(x)
        mean += x / sample_size
    for i in range(1, sample_size+1):
        S += torch.matmul(samples[i-1]-mean, (samples[i-1]-mean).t()) / sample_size
    print(S)

    #S = torch.tensor([[9,2,3],[2,8,1],[3,1,16]], dtype=torch.float32)
    R = torch.inverse(S)

    NLogL = net(dim=d, diag=6)

    param_groups = []
    param_groups.append({'params':[NLogL.X], 'dimension':d})
    optimizer = SGD(params=param_groups, lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000,4000,8000], gamma=0.1)

    print(NLogL.X)

    max_iter = 10000
    for i in range(1, max_iter+1):
        loss = NLogL.forward(S)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % 100 == 0:
            print("iteration={}".format(i))
            print("-log_likelihood={}".format(loss.item()))
            print("gradient={}".format(NLogL.X.grad))
            print("estimation={}".format(NLogL.X))

if __name__ == '__main__':
    main()