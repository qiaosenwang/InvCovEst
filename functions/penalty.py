'''

'''
import numpy as np


def L0_penal(x):
    y = x.copy()
    y[y == 0] = 0
    y[y != 0] = 1
    return y.sum()


def L1_penal(x):
    pass


