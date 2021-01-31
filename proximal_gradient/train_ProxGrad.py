'''
    Train with My Glasso
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
from multistep_GD.Gaussian_generator import Gaussian_Distribution
from functions.plot_heatmap import heatmap
from functions.measurements import *
from ProxGrad import *
from cross_val_ProxGrad import cross_val_score_ProxGrad
from sklearn.covariance import GraphicalLassoCV
import pandas as pd
from data_processing import z_score


def main():
    mean = torch.tensor(np.zeros(32), dtype=torch.float32)
    diag = torch.tensor(np.ones(32), dtype=torch.float32)
    X = torch.eye(32, dtype=torch.float32)
    X[5, 10] = X[10, 5] = X[19, 20] = X[20, 19] = X[1, 31] =X[31, 1] = -0.5

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=-0.2, type='DIY', slash=1, prec=X)
    truth = population.invcov.numpy()
    n = 400
    p = population.dim

    print(truth)
    heatmap(truth)




    data = pd.read_csv("exp3.csv")
    sample = data.values[1:, 1:]
    sample = z_score(sample)
    emp_cov = sample_cov(sample)

    score = dict()
    score['log_lik'] = []
    score['AIC'] = []
    score['non_zero'] = []
    alpha_list = np.hstack((np.arange(1e-5, 0.1, 0.002), np.arange(0.1, 0.3, 0.01)))
    data = np.array(sample)
    for alpha in alpha_list:
        out_dict = cross_val_score_ProxGrad(data, alpha=alpha, type='FISTA')
        score['log_lik'].append(out_dict['log_lik'])
        score['AIC'].append(out_dict['AIC'])
        score['non_zero'].append(out_dict['non_zero'])
    plt.plot(alpha_list, score['log_lik'])
    plt.show()
    plt.plot(alpha_list, score['AIC'])
    plt.show()
    plt.plot(alpha_list, score['non_zero'])
    plt.show()

    model = ProxGrad()
    l = len(alpha_list)
    alpha = 0
    log_lik = -1e12
    for i in range(0, l):
        if score['log_lik'][i] > log_lik:
            alpha = alpha_list[i]
            log_lik = score['log_lik'][i]
    print(alpha)
    prec = model.fit_FISTA(emp_cov, alpha)
    heatmap(prec)
    print('nonzero:', L0_penal(prec))

    alpha = 0
    aic = 1e12
    for i in range(0, l):
        if score['AIC'][i] < aic:
            alpha = alpha_list[i]
            aic = score['AIC'][i]
    print(alpha)
    prec = model.fit_FISTA(emp_cov, alpha)
    heatmap(prec)
    print('nonzero:', L0_penal(prec))



    model = GraphicalLassoCV(tol=1e-8)
    model.fit(sample)
    heatmap(model.precision_)
    print('nonzero:', L0_penal(model.precision_))







if __name__ == '__main__':
    main()