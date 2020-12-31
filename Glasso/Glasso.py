'''
    Glasso
'''

import numpy as np
from scipy import linalg
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, ledoit_wolf
import torch
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from multistep_GD.Gaussian_generator import  Gaussian_Distribution
from multistep_GD.plot_heatmap import heatmap

from cross_validation import *

def main():
    mean = torch.tensor(np.ones(16), dtype=torch.float32)
    diag = torch.tensor(np.ones(16), dtype=torch.float32)

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=0.3, type='chain', slash=1)
    truth = population.invcov.numpy()
    n = 1000
    d = population.dim

    print(truth)
    dist, sample, _, S = population.generate(n, numpy_like=True)
    #print(S)
    #print(np.array(sample))
    print(sample_mean(np.array(sample)))
    print(sample_cov(np.array(sample)))

    R = np.linalg.inv(S)
    #print(R)
    #print(sample)
    np.random.seed(0)
    model = GraphicalLassoCV()
    model.fit(np.array(sample))
    cov_ = model.covariance_
    prec_ = model.precision_

    heatmap(prec_)

    plt.figure(figsize=(4, 3))
    plt.axes([.2, .15, .75, .7])
    plt.plot(model.cv_alphas_, np.mean(model.grid_scores_, axis=1), 'o-')
    plt.axvline(model.alpha_, color='.5')
    plt.title('Model selection')
    plt.ylabel('Cross-validation score')
    plt.xlabel('alpha')

    plt.show()
    print(model.cv_alphas_, model.grid_scores_)

    model = GraphicalLasso()
    model.fit(sample)
    heatmap(model.precision_, 0.055)

    score = dict()
    score['log_lik'] = []
    score['AIC'] = []
    alpha_list = np.hstack((np.arange(0, 0.1, 0.001), np.arange(0.11, 0.3, 0.01)))
    data = np.array(sample)
    for alpha in alpha_list:
        out_dict = cross_val_score_GLasso(data, alpha=alpha)
        score['log_lik'].append(out_dict['log_lik'])
        score['AIC'].append(out_dict['AIC'])
    plt.plot(alpha_list, score['log_lik'], 'o-')
    plt.show()
    plt.plot(alpha_list, score['AIC'])
    plt.show()



if __name__ == '__main__':
    main()