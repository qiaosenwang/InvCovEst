'''

'''
import numpy as np
from functions.measurements import *
from functions.penalty import *

def grad_surrogate(x, delta):
    p = x.shape[0]
    grad = np.zeros_like(x, dtype=float)
    for i in range(1, p):
        for j in range(0, i):
            if abs(x[i, j]) > delta:
                grad[i, j] = grad[j, i] = np.sign(x[i, j])
            else:
                grad[i, j] = grad[j, i] = x[i, j] / delta
    return grad


def l1_penal_off_diag(x):

    return np.abs(x).sum() - np.abs(np.diag(x)).sum()


def dual_gap(emp_cov, x, alpha):

    p = x.shape[0]
    return np.sum(emp_cov * x) + alpha * l1_penal_off_diag(x) - p



def grad(x, emp_cov, alpha, delta):

    return emp_cov - np.linalg.inv(x) + alpha * grad_surrogate(x, delta)

class GD:

    def __init__(self):
        pass


    def fit(self, emp_cov, alpha=0.05, delta=1e-8, lr=0.01, tol=1e-8):

        max_iter = 2000
        iteration = 0
        p = emp_cov.shape[0]
        prec = np.linalg.pinv(emp_cov) + 0.1 * np.eye(p)

        while iteration < max_iter:
            prec_former = prec.copy()
            prec -= lr * grad(prec, emp_cov, alpha, delta)
            iteration += 1
            #print(np.linalg.norm(prec_former-prec))
            if dual_gap(emp_cov, prec, alpha) < tol:
                break

        print(iteration)

        self.prec = prec






