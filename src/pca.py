import numpy as np 
import pandas as pd 
from numpy.linalg import norm 

class Eigendigits():
    
    def __init__(self, n_pcs):
        """Constructs an Eigendigits object.
        
        Args:
            n_pcs: number of principal components
        """
        self.n_pcs = n_pcs # scalar
        self.mean = None # p vector
        self.explained_variance_ratio = None # scalar
        self.components = None # n_pcs x p matrix


    def __str__(self):
        return f"Eigendigits object\nNumber of principal components: {self.n_pcs}\nExplained variance ratio: {self.explained_variance_ratio}"

    def fit(self, X):
        """
        fit takes as input X, and assigns using this the attributes of the 
        Eigendigits object. 
        """
        self.n = X.shape[0]
        # initialising the sample covariance matrix 
        self.sample_cov_matrix = np.zeros((X.shape[1],X.shape[1]))

        
        self.mean = X.mean(axis=0)

        # calculating the sample covariance matrix 
        for i in range(self.n):

            X_centred = X[i].reshape(-1,1) - self.mean.reshape(-1,1)
            self.sample_cov_matrix += X_centred @ X_centred.transpose()
        self.sample_cov_matrix = (1/(self.n-1)) * (self.sample_cov_matrix)


        # numpy.linalg.eigh func computes the eigenvalues and eigenvectors of a cov matrix
        eigenvalues, eigenvectors = np.linalg.eigh(self.sample_cov_matrix)

        # order the eigenvalues 
        order = np.argsort(eigenvalues)[::-1]
        
        top_indicies = order[:self.n_pcs] # selects the first n elements 

        # obtain the n_pcs top eigenvalues and eigen vectors
        self.eigenvalues_n_pcs = eigenvalues[top_indicies]

        # eigenvectors are columns not rows
        self.eigenvectors_n_pcs = eigenvectors[:, top_indicies]

        # obtain the top eigen vectors 
        self.components = self.eigenvectors_n_pcs.T

        self.explained_variance_ratio = self.eigenvalues_n_pcs.sum() / eigenvalues.sum()


    def scores(self, Z):
        """
        method which takes as input a matrix and for each row(image)
        computes how far along each of the principal components it is
        """
        self.Z = Z 

        # Z: m x p 
        # mu = p

        # tells us how far along each of the principal components our images are
        scores = (self.Z - self.mean) @ self.eigenvectors_n_pcs

        return scores 


    def project(self, Z):
        self.Z = Z 

        # map the scores back to the original dimensions with the projections as rows 
        return self.scores(self.Z) @ self.eigenvectors_n_pcs.T + self.mean


    def classify(self, X_train, y_train, x_test): 
        """
        row one image/row, predict y using the L2-norm

        basically compare the l2-norm of the image with every image in the training set
        and see which are closest together in the new dimensionally reduced shape
        """
        #norms = np.linalg.norm(self.scores(x_test - X_train), ord=2, axis=1)
        norms = np.linalg.norm((self.scores(x_test) - self.scores(X_train)), ord=2, axis=1)
        idx = np.argmin(norms)
        return y_train[idx]