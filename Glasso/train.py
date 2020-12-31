'''
    Train with My Glasso
'''
import numpy as np
import torch
import sys
from multistep_GD.Gaussian_generator import Gaussian_Distribution
from plot_heatmap import heatmap

from My_Glasso import Glasso


def main():
    mean = torch.tensor(np.ones(16), dtype=torch.float32)
    diag = torch.tensor(np.ones(16), dtype=torch.float32)

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=0.3, type='chain', slash=1)
    truth = population.invcov.numpy()
    n = 1000
    d = population.dim

    print(truth)
    dist, sample, _, emp_cov = population.generate(n, numpy_like=True)

    model = Glasso(alpha=0.05)
    model.fit(emp_cov)

    heatmap(model.prec)
    #heatmap(np.linalg.inv(model.prec))


if __name__ == '__main__':
    main()