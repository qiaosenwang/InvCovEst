'''
    start training
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from multistep_GD.Gaussian_generator import Gaussian_Distribution
from gradient_descent.gd_method import GD

from plot_heatmap import heatmap


def main():
    mean = torch.tensor(np.ones(16), dtype=torch.float32)
    diag = 2 * torch.tensor(np.ones(16), dtype=torch.float32)

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=0.3, type='chain', slash=1)
    truth = population.invcov.numpy()
    n = 1000
    d = population.dim

    dist, sample, _, emp_cov = population.generate(n, numpy_like=True)
    model = GD(lr=0.01, alpha=0.01)
    model.fit(emp_cov)
    heatmap(model.prec)


if __name__ == '__main__':
    main()