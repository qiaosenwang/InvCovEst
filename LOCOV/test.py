import numpy as np

def main():
    covariance_ = np.array(np.arange(0,16).reshape(4,4))
    n_features = 4
    indices = np.arange(n_features)
    sub_covariance = np.copy(covariance_[1:, 1:], order='C')
    print(covariance_,'\n', sub_covariance, '\n')
    for idx in range(n_features):
        # To keep the contiguous matrix `sub_covariance` equal to
        # covariance_[indices != idx].T[indices != idx]
        # we only need to update 1 column and 1 line when idx changes
        if idx > 0:
            di = idx - 1
            sub_covariance[di] = covariance_[di][indices != idx]
            sub_covariance[:, di] = covariance_[:, di][indices != idx]
        else:
            sub_covariance[:] = covariance_[1:, 1:]
        print(covariance_,'\n', sub_covariance, '\n')

if __name__ == '__main__':
    main()