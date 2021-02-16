'''
    Train with My Glasso
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
from multistep_GD.Gaussian_generator import Gaussian_Distribution
from functions.plot_heatmap import heatmap
from functions.plot_network import network
from data_processing import z_score
from functions.measurements import *
from ProxGrad import *
from cross_val_ProxGrad import cross_val_score_ProxGrad
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import pandas as pd
import time


def main():

    data = pd.read_csv("protein.csv")
    sample = data.values
    print(sample)
    label = list(data.columns.values)
    print(label)
    precision_list = []

    sample = z_score(sample)
    emp_cov = sample_cov(sample)
    heatmap(emp_cov)
    heatmap(np.linalg.inv(emp_cov))
    '''

    score = dict()
    score['log_lik'] = []
    score['AIC'] = []
    score['non_zero'] = []
    alpha_list = np.hstack((np.arange(1e-5, 0.1, 0.002), np.arange(0.1, 0.5, 0.01)))
    data = np.array(sample)
    for alpha in alpha_list:
        out_dict = cross_val_score_ProxGrad(data, alpha=alpha, type='ISTA')
        score['log_lik'].append(out_dict['log_lik'])
        score['AIC'].append(out_dict['AIC'])
        score['non_zero'].append(out_dict['non_zero'])
    # plt.plot(alpha_list, score['log_lik'])
    # plt.ylabel("cross-validation score")
    # plt.xlabel("$\lambda$")
    # plt.title("Cross-Validation Curve")
    # plt.savefig("fig/temp.png", dpi=600)
    # plt.show()

    model = ProxGrad()
    l = len(alpha_list)
    alpha = 0
    log_lik = -1e12
    for i in range(0, l):
        if score['log_lik'][i] > log_lik:
            alpha = alpha_list[i]
            log_lik = score['log_lik'][i]
    prec = model.fit_ISTA(emp_cov, alpha)
    heatmap(prec)
    precision_list.append(prec.copy())
    '''

    score = dict()
    score['log_lik'] = []
    score['AIC'] = []
    score['non_zero'] = []
    alpha_list = np.hstack((np.arange(1e-5, 0.1, 0.002), np.arange(0.1, 0.5, 0.01)))
    data = np.array(sample)
    for alpha in alpha_list:
        out_dict = cross_val_score_ProxGrad(data, alpha=alpha, type='FISTA')
        score['log_lik'].append(out_dict['log_lik'])
        score['AIC'].append(out_dict['AIC'])
        score['non_zero'].append(out_dict['non_zero'])

    model = ProxGrad()
    l = len(alpha_list)
    alpha = 0
    log_lik = -1e12
    for i in range(0, l):
        if score['log_lik'][i] > log_lik:
            alpha = alpha_list[i]
            log_lik = score['log_lik'][i]

    prec = model.fit_FISTA(emp_cov, alpha)
    heatmap(prec)
    precision_list.append(prec.copy())
    print('nonzero:', L0_penal(prec))

    score = dict()
    score['log_lik'] = []
    score['AIC'] = []
    score['non_zero'] = []
    alpha_list = np.hstack((np.arange(1e-5, 0.1, 0.002), np.arange(0.1, 0.5, 0.01)))
    data = np.array(sample)
    for alpha in alpha_list:
        out_dict = cross_val_score_ProxGrad(data, alpha=alpha, type='MFISTA')
        score['log_lik'].append(out_dict['log_lik'])
        score['AIC'].append(out_dict['AIC'])
        score['non_zero'].append(out_dict['non_zero'])

    model = ProxGrad()
    l = len(alpha_list)
    alpha = 0
    log_lik = -1e12
    for i in range(0, l):
        if score['log_lik'][i] > log_lik:
            alpha = alpha_list[i]
            log_lik = score['log_lik'][i]
    prec = model.fit_MFISTA(emp_cov, alpha)
    heatmap(prec)
    precision_list.append(prec.copy())

    network(label, precision_list)

if __name__ == '__main__':
    main()