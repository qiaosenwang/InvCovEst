'''

'''
import numpy as np

def L0_penal(x):
    y = x.copy()
    y[y == 0] = 0
    y[y != 0] = 1
    return y.sum()

def df(x):
    p = x.shape[0]
    return((L0_penal(x) + p) / 2)

