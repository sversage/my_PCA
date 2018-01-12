import numpy as np
import matplotlib.pyplot as plt

class PCA(object):

    def __init__(self):
        #Attributes
        self.M = None
        self.eigen_vects = None
        self.eigen_values = None
        self.singular_values = None

    @property
    def reduced_mat(self):
        if len(self.eigen_vects) and len(self.eigen_values):
            return self.X.dot(self.eigen_vects)
        else:
            return "Please fit the object"

    def fit(self, X):

        #Calculate the cov/var matrix
        self.X = X
        self.M = np.cov(X, rowvar=False, ddof=1)

        #Calculates eigen values and eigen vectors
        eigen_values, eigen_vectors = np.linalg.eigh(self.M)

        self._sort_eigen_everything(eigen_values, eigen_vectors)

        self._calc_explained_variance()

    def _sort_eigen_everything(self, eigen_vals, eigen_vects):

        #Sort eigen values & vectors by index
        idx = np.argsort(eigen_vals)[::-1]
        self.eigen_vectors = eigen_vects[:,idx]
        self.eigen_values = eigen_vals[idx]

        #for comparison to SKLearn
        self.singular_values = self.eigen_values ** 2

    def _calc_explained_variance(self):

        #Calc explained variance
        cumulative_eig_vals = np.cumsum(self.eigen_values)
        self.explained_variance = self.eigen_values / cumulative_eig_vals

    def plot_explained_variance(self):
        plt.plot(self.explained_variance)
        plt.title('Explained Variance')
        plt.show()

    def plot_2d_embedding(self):
        plt.scatter(np.dot(self.X, self.eigen_vectors[:,1]),
                 np.dot(self.X, self.eigen_vectors[:,2]))
        plt.title('2D Embedding')
        plt.show()

if __name__ == '__main__':
    x = np.array([[0.5, 0.8, 1.5, -2.4],
                  [-1.9, -8.7, 0.02, 4.9],
                  [5.5,6.1, -8.1,3.0]])
    mpca = PCA()
    mpca.fit(x)
    mpca.plot_explained_variance()
    mpca.plot_2d_embedding()
