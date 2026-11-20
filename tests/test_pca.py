import numpy as np
import pandas as pd
from src.pca import Eigendigits

def test_components_shape():
    n_pcs = 12
    obj = Eigendigits(n_pcs)
    data = pd.read_csv('data/mnist.csv')
    X = np.array(data.drop("digit", axis=1).copy())
    obj.fit(X)
    np.testing.assert_array_equal(obj.components.shape, (n_pcs, 784))

def test_scores_shape():

    n_pcs = 15 
    obj = Eigendigits(n_pcs)
    data = pd.read_csv('data/mnist.csv')
    X = np.array(data.drop("digit", axis=1).copy())
    X_train = X[:4000]
    X_test = X[2000:]
    m = X_test.shape[0]
    obj.fit(X_train)
    scores = obj.scores(X_test)

    np.testing.assert_array_equal(scores.shape, (m, n_pcs))



def test_cov_matrix_equal():
    # create an Eigendigits object from the pca file
    obj = Eigendigits(10)

    X = np.array([[1,2],[-3,4]])

    # fit the Eigendigits object to the data 
    obj.fit(X)

    sample_cov_matrix = np.array([[8,-4],[-4,2]])
    expected = sample_cov_matrix 
    actual = obj.sample_cov_matrix

    np.testing.assert_array_equal(actual, expected)

def test_eigenvalues_equal():

    obj = Eigendigits(2)

    X = np.array([[5,-2],
              [1,2],
              [3,4],
              ])

    obj.fit(X)

    actual_eigenvalues = [11.474, 1.859]

    calculated_eigenvalues = obj.eigenvalues_n_pcs

    np.testing.assert_allclose(calculated_eigenvalues, 
                               actual_eigenvalues,
                               rtol=0.001)

def test_eigenvectors_equal():

    obj = Eigendigits(2)

    X = np.array([[5,-2],
              [1,2],
              [3,4],
              ])

    obj.fit(X)

    expected_eigenvectors = np.array([[-0.47185793, -0.8816746 ],
                                    [ 0.8816746 , -0.47185793]])

    actual_eigenvectors = obj.eigenvectors_n_pcs 

    np.testing.assert_allclose(expected_eigenvectors,
                               actual_eigenvectors,
                               rtol=0.001)

def test_project_back_to_original():

    data = pd.read_csv('data/mnist.csv')
    X = np.array(data.drop("digit", axis=1).copy())

    n_pcs = X.shape[1] # max number of principal components (all columns)
    obj = Eigendigits(n_pcs)
    obj.fit(X)

    np.testing.assert_allclose(obj.project(X),
                               X,
                               atol=0.0000000001)