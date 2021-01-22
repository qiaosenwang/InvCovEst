'''
    ISTA
'''
import numpy as np


def loss(emp_cov, precision):
    loss = np.sum(emp_cov * precision) - np.linalg.slogdet(precision)[1]
    return loss


def grad(emp_cov, precision):
    grad = - np.linalg.inv(precision) + emp_cov
    return grad


def soft_threshold(x, t):
    p = x.shape[0]

    y = x.copy()

    for i in range(1, p):
        for j in range(0, i):
            if x[i, j] > t:
                y[i, j] = y[j, i] = x[i, j] - t
            elif x[i, j] < -t:
                y[i, j] = y[j, i] = x[i, j] + t
            else:
                y[i, j] = y[j, i] = 0.

    return y


class ProxG:

    def __init__(self):
        pass

    def fit_ISTA(self, emp_cov, alpha):
        max_iter = 600
        step_size = 0.2
        tol = 1e-8
        p = emp_cov.shape[0]
        prec = np.linalg.pinv(emp_cov) + 1e-1 * np.eye(p)


        for iteration in range(1, max_iter+1):
            prec_former = prec.copy()
            temp = prec - step_size * grad(emp_cov, prec)
            prec = soft_threshold(temp, alpha * step_size)
            if np.linalg.norm(prec - prec_former, np.inf) < tol:
                print(iteration)
                break

        return prec





