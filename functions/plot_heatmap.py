

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def heatmap(prec, level=0):

    d = list(prec.shape)[0]
    mask = np.zeros_like(prec)
    font_setting = {'fontsize': 5}

    for i in range(0, d):
        for j in range(0, i+1):
            if abs(prec[i][j]) <= level:
                mask[i][j] = mask[j][i] = True
    anc = prec.copy()
    for i in range(0, d):
        anc[i, i] = 0
    max = np.max([np.max(np.abs(anc)), 1e-1])
    ax = sns.heatmap(prec, annot=prec, annot_kws=font_setting, cmap="RdBu_r", vmin=-1.2*max, vmax=1.2*max,
                square=True, mask=mask)
    ax.set_facecolor('.7')
    plt.show()

