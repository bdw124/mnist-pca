PCA from Frist Principles: Eigendigits on MNIST 

Principal Component Analysis (PCA) implemented from scratch in NumPy, then used to build a nearest-neighbour digit classifier in a reduced space. 

Please click on pca_visualisations.ipynb to see the results :) 

**Headline Result**: 96% test accuracy using 54 principal components instead of all 784 pixels (about 14x less data per image).  

**What I Learned**: 
## Dimension Reduction can match or beat the full data!
Explained variance is not accuracy. PCA ranks directions by variance, not by how well it seperates digits. The first components may capture broard shapes such as loops and strokes, whilst later components may capture handwriting style, positioning and pixel noise (overfitting). 

**What I would do differently**: 
- instead of picking number of components by elbow method, select with cross-validation on the training set. 
- Benchmark against a reference. 

- Eigendigits class (located in pca.py) fits PCA via the cov matrix and provides a score, projection and classification methods. 
- Unit tests (located in tests/test_pca.py) check the covariance matrix and eigendecomposition against hand-computed values and output shapes. 
- Notebooks: pca_explained.ipynb derives PCA from the variance-optimisation problem. pca_visualisations.ipynb shows reconstructions and the accuracy-vs-components curve. 

**To run:**
pip install numpy pandas matplotlib scikit-learn pytest
-- place MNIST as data/mnist.csv (column "digit" + 784 pixel columns)
pytest tests/
jupyter notebook pca_visualisations.ipynb