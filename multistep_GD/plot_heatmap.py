

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
    ax = sns.heatmap(prec, annot=prec, annot_kws=font_setting, cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                square=True, mask=mask)
    ax.set_facecolor('.7')
    plt.show()

