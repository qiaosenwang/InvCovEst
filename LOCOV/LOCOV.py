'''
    LOCOV
'''
import numpy as np
from functions.penalty import *


def cd_u(u, V_, g0, g, alpha):

    tol = 1e-18
    max_iter = 25
    d = u.shape[0]
    iteration = 0
    while iteration < max_iter:
        u_former = u.copy()
        for i in range(0, d):
            u0 = u.copy()
            u0[i, 0] = 0.0
            #print(u0)
            v_ii = V_[i, i]
            v_i = V_[:, i].reshape(d, 1)
            z = g0 * np.dot(u0.T, v_i) + g[i, 0]
            inner = abs(z) - np.sqrt(2 * g0 * v_ii * alpha)
            if abs(inner) > 1e-5:
                u[i, 0] = - z / (g0 * v_ii)
            else:
                u[i, 0] = 0
        if np.linalg.norm(u - u_former) < tol:
            break
        iteration += 1


def phi(u, w, V_, g, g0, alpha):

    out = np.log(w - np.dot(u.T, np.dot(V_, u))) - 2 * np.dot(g.T, u)
    out += -2 * alpha * L0_penal(u) - g0 * w - alpha
    return out

def wu(u, V_, g0, eps):

    out = np.dot(u.T, np.dot(V_, u))
    if g0 > 0:
        out += 1 / g0
    else:
        out += 1 / eps
    return out




class LOCOV:

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, emp_cov):
        eps = 1e-5
        max_iter = 25
        tol = 1e-15

        p = emp_cov.shape[0]
        #print(p)
        indices = np.arange(p)
        prec = np.eye(p)
        V = np.copy(prec[:p-1, :p-1], order='C')

        iteration = 0
        while iteration < max_iter:
            prec_former = prec.copy()

            for idx in range(p):
                # To keep the contiguous matrix `sub_covariance` equal to
                # covariance_[indices != idx].T[indices != idx]
                # we only need to update 1 column and 1 line when idx changes
                if idx > 0:
                    di = idx - 1
                    V[di] = prec[di][indices != idx]
                    V[:, di] = prec[:, di][indices != idx]
                else:
                    V[:] = prec[:p-1, :p-1]
                #print(V)

                V_ = np.linalg.inv(V)

                u = prec[indices != idx, idx].copy().reshape(p-1, 1)
                u_former = u.copy()
                w_former = prec[idx, idx]
                #G = emp_cov[:p-1, :p-1]
                #print(emp_cov)
                #print(idx)
                g0 = emp_cov[idx, idx]
                g = emp_cov[indices != idx, idx].reshape(p-1, 1)


                if g0 != 0:

                    cd_u(u, V_, g0, g, self.alpha)
                else:
                    print("#")
                    cd_u(u, V_, eps, g, self.alpha)
                w = wu(u, V_, g0, eps)
                #print(u.reshape(p-1))

                if phi(u, w, V_, g, g0, self.alpha) > phi(u_former, w_former, V_, g, g0, self.alpha):
                    prec[indices != idx, idx] = u.reshape(p-1).copy()
                    prec[idx, indices != idx] = u.reshape(p-1).copy()
                    prec[idx, idx] = w
                else:
                    prec[idx, idx] = wu(u_former, V_, g0, eps)

                #print(prec)
            iteration += 1
            #print(np.linalg.norm(prec - prec_former))

            if np.linalg.norm(prec - prec_former) < tol:
                print(iteration)
                break

        self.prec = prec








