'''
    generator
'''
import torch
import numpy as np

class Gaussian_Distribution:

    def __init__(self, type, mean, diag, sub=0, slash=1, cross=0, prec=None):
        self.type = type
        self.mean = mean
        self.dim = list(mean.size())[0]
        self.diag = diag
        self.sub = sub
        self.slash = slash
        self.cross = cross
        if self.type == 'discrete':
            self.invcov = torch.diag_embed(diag)
        elif self.type == 'chain':
            self.invcov = torch.diag_embed(diag)
            sub_tensor = self.sub * torch.ones(self.dim - slash)
            self.invcov += torch.diag(sub_tensor, slash) + torch.diag(sub_tensor, -slash)
        elif self.type == 'star':
            self.invcov = torch.diag_embed(diag)
            for i in range(0, self.dim):
                if i != cross:
                    self.invcov[i][cross] = self.invcov[cross][i] = self.sub
        elif self.type == 'DIY':
            self.invcov = prec
            self.dim = list(prec.size())[0]
            self.mean = torch.zeros(self.dim, dtype=torch.float32)
        self.invcov = torch.FloatTensor(self.invcov)
        self.cov = torch.inverse(self.invcov)



    def generate(self, size, numpy_like=False):
        dist = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, self.cov)
        sample = []
        sample_mean = torch.zeros(1, self.dim, dtype=torch.float32)
        sample_cov = torch.zeros(self.dim, self.dim, dtype=torch.float32)
        for i in range(0, size):
            x = dist.sample()
            sample.append(x)
            sample_mean += x / size
        for i in range(0, size):
            sample_cov += torch.matmul((sample[i]-sample_mean).t(), sample[i]-sample_mean) / size
        if numpy_like:
            sample_np = []
            for t in sample:
                sample_np.append(t.numpy())
            cov_np = (self.cov.numpy() + self.cov.numpy().T) / 2
            return np.random.multivariate_normal(self.mean.numpy(), cov_np), sample_np, sample_mean.numpy(), sample_cov.numpy()
        else:
            return dist, sample, sample_mean, sample_cov


