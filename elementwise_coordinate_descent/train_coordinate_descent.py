'''
    train
'''

import numpy as np
import torch
from multistep_GD.Gaussian_generator import Gaussian_Distribution
from plot_heatmap import heatmap
from Marjanovic_coordinate_descent import MarjanovicCoordinateDescent

def main():
    mean = torch.tensor(np.ones(16), dtype=torch.float32)
    diag = 2 * torch.tensor(np.ones(16), dtype=torch.float32)

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=0.3, type='chain', slash=1, cross=7)
    truth = population.invcov.numpy()
    n = 1000
    d = population.dim

    print(truth)
    dist, sample, _, emp_cov = population.generate(n, numpy_like=True)
    print(emp_cov)
    for alpha in np.arange(1e-3, 6e-3, 5e-4):
        model = MarjanovicCoordinateDescent(alpha=alpha)
        model.fit(emp_cov)
        print(alpha)
        heatmap(model.prec)

if __name__ == '__main__':
    main()
