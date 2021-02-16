'''
    ISTA
'''
import numpy as np

def l0_penal_off_diag(x):
    p = x.shape[0]
    count = 0
    for i in range(1, p):
        for j in range(0, i):
            if x[i, j] != 0:
                count += 1
    return 2 * count


def l1_penal(x):

    return np.abs(x).sum()


def l1_penal_off_diag(x):

    return np.abs(x).sum() - np.abs(np.diag(x)).sum()


def loss(emp_cov, x, alpha):
    loss = np.sum(emp_cov * x) - np.linalg.slogdet(x)[1] + alpha * l0_penal_off_diag(x)
    return loss


def grad(emp_cov, x):
    grad = - np.linalg.inv(x) + emp_cov
    return grad


def hard_threshold(x, t):
    p = x.shape[0]

    y = x.copy()
    s = np.sqrt(2 * t)

    for i in range(1, p):
        for j in range(0, i):
            if x[i, j] > s:
                y[i, j] = y[j, i] = x[i, j]
            elif x[i, j] < -s:
                y[i, j] = y[j, i] = x[i, j]
            else:
                y[i, j] = y[j, i] = 0.

    return y


def prox_map(emp_cov, x, step_size, alpha):

    return hard_threshold(x - step_size * grad(emp_cov, x), step_size * alpha)

def dual_gap(emp_cov, x, alpha):

    p = x.shape[0]
    return np.sum(emp_cov * x) + alpha * l1_penal_off_diag(x) - p



class ProxGrad_l0:

    def __init__(self):
        pass

    def fit_ISTA(self, emp_cov, alpha, step_size=0.25, tol=1e-8, dg=None, F=None):
        max_iter = 600
        p = emp_cov.shape[0]
        prec = np.linalg.pinv(emp_cov) + 1e-1 * np.eye(p)

        iteration = 0
        if dg != None:
            dg.append(max(dual_gap(emp_cov, prec, alpha), 0))
        if F != None:
            F.append(loss(emp_cov, prec, alpha))
        while iteration < max_iter:
            prec_former = prec.copy()
            temp = prec - step_size * grad(emp_cov, prec)
            prec = hard_threshold(temp, alpha * step_size)
            iteration += 1
            if dg != None:
                dg.append(max(dual_gap(emp_cov, prec, alpha), 0))
            if F != None:
                F.append(loss(emp_cov, prec, alpha))
            if np.linalg.norm(prec_former-prec) < tol:
                break
        print(iteration)

        return prec

    def fit_FISTA(self, emp_cov, alpha, step_size=0.25, tol=1e-8, dg=None, F=None):

        max_iter = 600
        t = 1
        p = emp_cov.shape[0]
        prec = np.linalg.pinv(emp_cov) + 1e-1 * np.eye(p)
        y = prec.copy()

        iteration = 0
        if dg != None:
            dg.append(max(dual_gap(emp_cov, prec, alpha), 0))
        if F != None:
            F.append(loss(emp_cov, prec, alpha))
        while iteration < max_iter:
            prec_former = prec.copy()
            t_former = t
            prec = prox_map(emp_cov, y, step_size, alpha)
            t = (1 + np.sqrt(1 + 4 * (t ** 2))) / 2
            y = prec + (t_former - 1) / t * (prec - prec_former)
            iteration += 1
            #print(dual_gap(emp_cov, prec, alpha))
            if dg != None:
                dg.append(max(dual_gap(emp_cov, prec, alpha), 0))
            if F != None:
                F.append(loss(emp_cov, prec, alpha))
            if np.linalg.norm(prec_former-prec) < tol:
                break
        print(iteration)

        return prec

    def fit_MFISTA(self, emp_cov, alpha, step_size=0.25, tol=1e-4, dg=None, F=None):

        max_iter = 600
        t = 1
        p = emp_cov.shape[0]
        prec = np.linalg.pinv(emp_cov) + 1e-1 * np.eye(p)
        y = prec.copy()

        iteration = 0
        if dg != None:
            dg.append(max(dual_gap(emp_cov, prec, alpha), 0))
        if F != None:
            F.append(loss(emp_cov, prec, alpha))
        while iteration < max_iter:
            prec_former = prec.copy()
            t_former = t
            z = prox_map(emp_cov, y, step_size, alpha)
            if loss(emp_cov, z, alpha) < loss(emp_cov, prec, alpha):
                prec = z
            t = (1 + np.sqrt(1 + 4 * (t ** 2))) / 2
            y = prec + t_former / t * (z - prec) + (t_former - 1) / t * (prec - prec_former)
            iteration += 1
            #print(iteration)
            #print(dual_gap(emp_cov, prec, alpha))
            if dg != None:
                dg.append(max(dual_gap(emp_cov, prec, alpha), 0))
            if F != None:
                F.append(loss(emp_cov, prec, alpha))
            if dual_gap(emp_cov, prec, alpha) < tol:
                break
        print(iteration)

        return prec






