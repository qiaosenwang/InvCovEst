'''
    start training
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from multistep_GD.Gaussian_generator import Gaussian_Distribution
from gradient_descent.gd_method import GD
from measurements import *
from plot_heatmap import heatmap
import pandas as pd



def main():
    mean = torch.tensor(np.ones(16), dtype=torch.float32)
    diag = 2 * torch.tensor(np.ones(16), dtype=torch.float32)

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=0.3, type='chain', slash=1)
    truth = population.invcov.numpy()
    n = 1000
    d = population.dim

    data = pd.read_csv("samples.csv")
    sample = data.iloc[1:, 1:].values

    emp_cov = sample_cov(sample)

    #dist, sample, _, emp_cov = population.generate(n, numpy_like=True)
    model = GD()
    model.fit(emp_cov, alpha=0.05, lr=0.01)
    heatmap(emp_cov)
    heatmap(model.prec)



if __name__ == '__main__':
    main()