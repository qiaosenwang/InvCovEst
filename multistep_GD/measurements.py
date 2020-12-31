'''

'''

import numpy as np
from multistep_GD.penalty import *



def log_likelihood(emp_cov, precision):
    p = precision.shape[0]
    log_likelihood_ = - np.sum(emp_cov * precision) + np.linalg.slogdet(precision)[1]
    log_likelihood_ -= p * np.log(2 * np.pi)
    log_likelihood_ /= 2.
    return log_likelihood_

def df(x):
    p = x.shape[0]
    return((L0_penal(x) + p) / 2)

def AIC(emp_cov, precision):
    return -2 * log_likelihood(emp_cov, precision) + 2 * df(precision)

def sample_mean(data):
    n = data.shape[0]
    p = data.shape[1]

    mean = np.zeros(p, dtype=float)
    for i in range(0, n):
        mean += data[i, :] / n
    return mean

def sample_cov(data):
    n = data.shape[0]
    p = data.shape[1]
    mean = sample_mean(data)

    cov = np.zeros((p, p))
    for i in range(0, n):
        cov += np.dot((data[i, :]-mean).reshape(p, 1), (data[i, :]-mean).reshape(1, p)) / n
    return cov


