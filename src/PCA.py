import numpy as np

class PCA(object):

    def __init__(self):
        self.power = power


    def fit(self, X, center=True):
        if center:
            X = X - X.mean(axis=0)

        #Calculate scatter matrix
        #Note here we aren't using var/cov (eigen space is the same)
        M = self._calc_M(X)

        #Calculates eigen values and eigen vectors
        eigen_values, eigen_vectors = np.linalg.eig(M)

        self._sort_eigen_everything(eigen_values, eigen_vectors)

        #Get idx of eigen values which satify power input
        component_index = self._power_calc(eigen_values)

    def _calc_M(self, X):
        #calculate variance/covariance matrix
        return X.T.dot(X) / (X.shape[0] - 1)

    def _sort_eigen_everything(self, eigen_vals, eigen_vects):

        #Sort eigen values by index
         = np.argsort(eigen_vals)[::-1]

        #sort eigen values from high to low
        self.eigen_values = eigen_vals[eigen_indices]

        #for comparison to SKLearn
        self.singular_values = np.sqrt(self.eigen_values)

        #Calc explained variance
        cumulative_eig_vals = np.cumsum(sorted_vals)
        self.explained_variance = sorted_vals / cumulative_eig_vals

        #Sort eigen vectors
        self.eigen_vects = eigen_vectors.T[eigen_indices].T


if __name__ == '__main__':
    pass
