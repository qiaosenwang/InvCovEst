'''
    Train with My Glasso
'''
import numpy as np
import torch
import matplotlib.pyplot as plt
from multistep_GD.Gaussian_generator import Gaussian_Distribution
from functions.plot_heatmap import heatmap
from functions.measurements import *
from ISTA import *
from cross_val_ISTA import cross_val_score_ISTA
from sklearn.covariance import GraphicalLassoCV


def main():
    mean = torch.tensor(np.ones(16), dtype=torch.float32)
    diag = torch.tensor(np.ones(16), dtype=torch.float32)

    population = Gaussian_Distribution(mean=mean, diag=diag, sub=-0.3, type='chain', slash=1)
    truth = population.invcov.numpy()
    n = 200
    d = population.dim

    print(truth)
    dist, sample, _, emp_cov = population.generate(n, numpy_like=True)




    '''
    tol = 0.1

    alpha_list = []
    score_list = []
    left = 1e-5
    left_score = cross_val_score_ISTA(sample, left)['AIC']
    alpha_list.append(left)
    score_list.append(left_score)

    right = 1.5
    right_score = cross_val_score_ISTA(sample, right)['AIC']
    alpha_list.append(right)
    score_list.append(right_score)

    
    while right - left > tol:
        mid = (left + right) / 2
        mid_score = cross_val_score_ISTA(sample, mid)['AIC']
        if mid_score > right_score:
            pivot = right
            break
        elif mid_score > left_score:
            pivot = left
            break
        else:
            pivot = mid
            alpha_list.append(mid)
            score_list.append(mid_score)
            left = (left + mid)/2
            right = (right + mid) / 2
        left_score = cross_val_score_ISTA(sample, left)
        right_score = cross_val_score_ISTA(sample, right)
        alpha_list.append(left)
        alpha_list.append(right)
        score_list.append(left)
        score_list.append(right)
    '''

    score = dict()
    score['log_lik'] = []
    score['AIC'] = []
    score['non_zero'] = []
    alpha_list = np.hstack((np.arange(1e-5, 0.1, 0.002), np.arange(0.1, 0.3, 0.01)))
    data = np.array(sample)
    for alpha in alpha_list:
        out_dict = cross_val_score_ISTA(data, alpha=alpha)
        score['log_lik'].append(out_dict['log_lik'])
        score['AIC'].append(out_dict['AIC'])
        score['non_zero'].append(out_dict['non_zero'])
    plt.plot(alpha_list, score['log_lik'])
    plt.show()
    plt.plot(alpha_list, score['AIC'])
    plt.show()
    plt.plot(alpha_list, score['non_zero'])
    plt.show()

    model = ProxG()
    l = len(alpha_list)
    alpha = 0
    log_lik = -1e12
    for i in range(0, l):
        if score['log_lik'][i] > log_lik:
            alpha = alpha_list[i]
            log_lik = score['log_lik'][i]
    print(alpha)
    prec = model.fit_ISTA(emp_cov, alpha)
    heatmap(prec)
    print('nonzero:', L0_penal(prec))

    alpha = 0
    aic = 1e12
    for i in range(0, l):
        if score['AIC'][i] < aic:
            alpha = alpha_list[i]
            aic = score['AIC'][i]
    print(alpha)
    prec = model.fit_ISTA(emp_cov, alpha)
    heatmap(prec)
    print('nonzero:', L0_penal(prec))



    model = GraphicalLassoCV()
    model.fit(sample)
    heatmap(model.precision_)
    print('nonzero:', L0_penal(model.precision_))


    #heatmap(np.linalg.inv(model.prec))
    #return




if __name__ == '__main__':
    main()