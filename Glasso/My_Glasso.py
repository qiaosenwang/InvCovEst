'''
    My_Glasso
'''

import numpy as np
import scipy

def soft_threshold(x, t):

    if x > t:
        return x - t
    elif x < -t:
        return x + t
    else:
        return 0.


def cd(s, V, alpha):

    d = s.shape[0]
    beta = np.ones([d, 1])
    max_iter = 50
    tol = 1e-20

    iteration = 0
    while iteration < max_iter:
        beta_former = beta.copy()
        for j in range(d):
            sum = 0
            for k in range(d):
                if k != j:
                    sum += V[k, j] * beta[k, 0]
            beta[j, 0] = soft_threshold(s[j]-sum, alpha)
        iteration += 1
        if np.linalg.norm(beta - beta_former) < tol:
            break
    #print(iteration)
    return beta



class Glasso:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, emp_cov):

        max_iter = 50
        tol = 1e-20
        p = emp_cov.shape[0]

        indices = np.arange(p)
        W = emp_cov + self.alpha * np.eye(p)
        V = np.copy(W[:p - 1, :p - 1], order='C')
        prec = np.linalg.pinv(emp_cov) + self.alpha * np.eye(p)

        iteration = 0
        while iteration < max_iter:
            W_former = W.copy()

            for idx in range(p):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    V[di] = W[di][indices != idx]
                    V[:, di] = W[:, di][indices != idx]
                else:
                    V[:] = W[:p - 1, :p - 1]

                u = W[indices != idx, idx].copy().reshape(p - 1, 1)
                s = emp_cov[indices != idx, idx].copy().reshape(p - 1, 1)
                beta = cd(s, V, self.alpha)
                prec[idx, idx] = 1. / (W[idx, idx] - np.dot(u.T, beta))
                prec[indices != idx, idx] = - beta.reshape(p - 1) * prec[idx, idx]
                prec[idx, indices != idx] = - beta.reshape(p - 1) * prec[idx, idx]

                W[indices != idx, idx] = np.dot(V, beta).reshape(p - 1)
                W[idx, indices != idx] = np.dot(V, beta).reshape(p - 1)

                #print(prec)
            print(np.linalg.norm(W - W_former, ord=np.inf))
            iteration += 1
            if np.linalg.norm(W - W_former, ord=np.inf) < tol:
                break
        print(iteration)

        self.prec = prec
