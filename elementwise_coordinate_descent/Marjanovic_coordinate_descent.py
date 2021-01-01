'''
    proposed by Marjanocvic
'''

import numpy as np


def Z(x, i, j, X0):
    p = X0.shape[0]

    if i == j:
        E = np.zeros([p, p])
        E[i, i] = 1
        return X0 + (x - X0[i, i]) * E
    else:
        U = np.zeros([p, p])
        U[i, j] = U[j, i] = 1
        return X0 + (x - X0[i, j]) * U



def phi(x, i, j, X0, S, alpha):

    out = - np.linalg.slogdet(Z(x, i, j, X0))[1] + S[i, j] * x
    if i == j:
        return out
    else:
        out += S[i, j] * x + 2 * alpha * np.sign(abs(x))
        return out


def delta(X, i, j):
    #print('*', X[i, i] * X[j, j] - X[i, j] * X[i, j])
    return X[i, i] * X[j, j] - X[i, j] * X[i, j]




class MarjanovicCoordinateDescent:

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, emp_cov):

        max_iter = 100
        tol = 1e-8
        p = emp_cov.shape[0]

        d = np.diag(emp_cov).copy()
        Y = np.diag(d).copy()
        d = 1 / d
        X = np.diag(d)
        #print(Y)
        #print(X)
        #print(emp_cov)
        #X = np.eye(p)
        #Y = np.eye(p)

        iteration = 0
        while iteration < max_iter:
            X_former = X.copy()
            for i in range(p):
                for j in range(i + 1):
                    if i == j:
                        X[i, i] = X_former[i, i] + (Y[i, i] - emp_cov[i, i]) / (Y[i, i] * emp_cov[i, i])
                    else:
                        m = X_former[i, j] + Y[i, j] / delta(Y, i, j)
                        if emp_cov[i, j] != 0:
                            m += (delta(Y, i, j) - np.sqrt(delta(Y, i, j)**2 + 4 * emp_cov[i, j]**2 * Y[i, i] * Y[j, j])) / (2 * delta(Y, i, j) * emp_cov[i, j])
                        ct = - delta(Y, i, j) * X_former[i, j] ** 2 - 2 * X_former[i, j] * Y[i, j] + 1
                        #print(ct)
                        if ct > 0:
                            if phi(0, i, j, X_former, emp_cov, self.alpha) < phi(m, i, j, X_former, emp_cov, self.alpha):
                                X[i, j] = X[j, i] = 0
                                #print(1)
                            elif phi(0, i, j, X_former, emp_cov, self.alpha) == phi(m, i, j, X_former, emp_cov, self.alpha):
                                X[i, j] = X[j, i] = m * np.sign(abs(X_former[i, j]))
                                #print(2)
                            else:
                                X[i, j] = X[j, i] = m
                                #print(3)
                        else:
                            X[i, j] = X[j, i] = m
                            #print(4)
            Y = np.linalg.inv(X)
            #print(X)
            #print(Y)
            iteration += 1
            print(np.linalg.norm(X - X_former, ord=np.inf))
            if np.linalg.norm(X - X_former) < tol:
                break
        print(iteration)

        self.prec = X












