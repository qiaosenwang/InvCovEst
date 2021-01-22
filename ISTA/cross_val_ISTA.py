from ISTA import *
from functions.measurements import *

def cross_val_score_ISTA(data, fold=5, alpha=0.01):

    n = data.shape[0]
    m = int(n / fold)
    score = {}
    score['log_lik'] = 0
    score['AIC'] = 0
    score['non_zero'] = 0

    for i in range(1, fold + 1):
        test_index = np.arange((i-1) * m, i * m)
        #print(test_index)
        train_index = np.delete(np.arange(0, n), test_index)
        test_data = data[test_index, :]
        train_data = data[train_index, :]
        test_cov = sample_cov(test_data)
        train_cov = sample_cov(train_data)
        model = ProxG()
        model.fit_ISTA(train_cov, alpha)
        prec = model.fit_ISTA(train_cov, alpha)

        score['log_lik'] += log_likelihood(test_cov, prec) / fold
        score['AIC'] += AIC(test_cov, prec, n - m) / fold
        score['non_zero'] += L0_penal(prec) / fold

    return score