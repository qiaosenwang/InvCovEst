'''

'''
import numpy as np
from functions.measurements import *
from functions.penalty import *


def grad(x, emp_cov, alpha):

    return emp_cov - np.linalg.inv(x) + alpha * np.sign(x)

class GD:

    def __init__(self, lr=0.01, alpha=0.01):
        self.lr = lr
        self.alpha = alpha

    def fit(self, emp_cov, tol=1e-15):

        max_iter = 200000
        iteration = 0
        p = emp_cov.shape[0]
        prec = np.linalg.pinv(emp_cov) + np.eye(p)

        while iteration < max_iter:
            prec_former = prec.copy()
            prec -= self.lr * grad(prec, emp_cov, self.alpha)
            iteration += 1
            print(np.linalg.norm(prec_former-prec))
            if np.linalg.norm(prec_former-prec) < tol:
                break

        print(iteration)

        self.prec = prec






