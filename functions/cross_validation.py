
'''
    cross validation
'''

import numpy as np
from measurements import *
from sklearn.covariance import GraphicalLasso
from LOCOV.LOCOV import LOCOV
from penalty import *

def cross_val_score_GLasso(data, fold=5, alpha = 0.01):

    n = data.shape[0]
    m = int(n / fold)
    score = {}
    score['log_lik'] = 0
    score['AIC'] = 0
    score['non_zero'] = 0

    for i in range(1, fold + 1):
        test_index = np.arange((i-1) * m, i * m)
        #print(test_index)
        train_index = np.delete(np.arange(0, n), test_index)
        test_data = data[test_index, :]
        train_data = data[train_index, :]
        cov = sample_cov(test_data)
        model = GraphicalLasso(alpha=alpha)
        model.fit(train_data)
        prec = model.precision_

        score['log_lik'] += log_likelihood(cov, prec) / fold
        score['AIC'] += AIC(cov, prec, n-m) / fold
        score['non_zero'] += L0_penal(prec) / fold

    return score


def cross_val_score_LOCOV(data, fold=5, alpha=0.01):

    n = data.shape[0]
    m = int(n / fold)
    score = {}
    score['log_lik'] = 0
    score['AIC'] = 0
    score['non_zero'] = 0

    for i in range(1, fold + 1):
        test_index = np.arange((i-1) * m, i * m)
        #print(test_index)
        train_index = np.delete(np.arange(0, n), test_index)
        test_data = data[test_index, :]
        train_data = data[train_index, :]
        test_cov = sample_cov(test_data)
        train_cov = sample_cov(train_data)
        model = LOCOV(alpha=alpha)
        model.fit(train_cov)
        prec = model.prec

        score['log_lik'] += log_likelihood(test_cov, prec) / fold
        score['AIC'] += AIC(test_cov, prec, n - m) / fold
        score['non_zero'] += L0_penal(prec) / fold

    return score
