import numpy as np

class PCA(object):

    def __init__(self, power=.9):
        self.power = power


    def fit(self, X, center=True):
        if center:
            X = X - X.mean(axis=0)

        #Calculate scatter matrix
        #Note here we aren't using var/cov (eigen space is the same)
        M = self._calc_M(X)

        #Calculates eigen values and eigen vectors
        eigen_values, eigen_vectors = np.linalg.eig(M)

        #Get idx of eigen values which satify power input
        component_index = self._power_calc(eigen_values)

        self.eigen_vals = eigen_values[component_index]
        self.eigen_vects = eigen_vectors.T[component_index].T
        self.reduced_dim_mat = X.dot(self.eigen_vects)

    def _calc_M(self, X):
        #calculate variance/covariance matrix
        return X.T.dot(X) / (X.shape[0] - 1)

    def _power_calc(self, eigen_values):

        #get largest to smallest eigven vals
        high_to_low_idx = np.argsort(eigen_values)[::-1]
        total_eigs = np.sum(eigen_values)
        total = 0

        #Determine how many components to keep
        for idx, sort_idx in enumerate(high_to_low_idx):
            total += eigen_values[sort_idx]
            if total/total_eigs < self.power:
                continue
            return high_to_low_idx[:idx+1]

    def


if __name__ == '__main__':
    pass
