'''

'''

import numpy as np
from functions.measurements import *
import pandas as pd
from plot_heatmap import heatmap
from ProxGrad import ProxGrad
from cross_val_ProxGrad import cross_val_score_ProxGrad
import matplotlib.pyplot as plt
from plot_network import network

def z_score(data):
    p = data.shape[1]

    for i in range(0, p):
        col = data[:, i].reshape(-1, 1)
        mean = sample_mean(col)
        sd = sample_cov(col)
        data[:, i] = (data[:, i] - mean) / np.sqrt(sd)
    return data



