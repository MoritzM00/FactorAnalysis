"""
Exploratory Factor Analysis.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from utils import smc, standardize


class FactorAnalysis(BaseEstimator, TransformerMixin):
    """
    The linear factor analysis model.

    TODO: docstring

    Parameters
    ----------
    n_factors : int
        The number of factors.
    """

    def __init__(self, n_factors, method="paf", max_iter=50):
        self.n_factors = n_factors
        self.method = method
        self.max_iter = max_iter

    def fit(self, X, y=None):
        """
        Fits the factor analysis model to the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, only for API consistency

        Returns
        -------
        self : FactorAnalysis
            The fitted model.
        """
        X = check_array(X, copy=True)
        self.n_samples_, self.n_features_ = X.shape

        # standardize data
        Z, self.mean_, self.std_ = standardize(X)

        # calculate initial correlation matrix
        corr = np.dot(Z.T, Z) / (self.n_samples_ - 1)
        self.corr_ = corr.copy()

        if self.method == "paf":
            self._fit_principal_axis()
        else:
            raise ValueError(f"Method {self.method} is not supported.")

        if self.n_factors > 1:
            # update loading signs to match column sums
            # this is to ensure that signs align with package factor_analyzer
            signs = np.sign(self.loadings_.sum(0))
            signs[(signs == 0)] = 1
            self.loadings_ = np.dot(self.loadings_, np.diag(signs))
        return self

    def transform(self, X):
        """
        Transforms the data using the factors.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        X_new : array_like, shape (n_samples, n_factors)
            The transformed samples.
        """
        pass

    def _fit_principal_axis(self):
        corr = self.corr_.copy()
        print()
        print(corr)

        # using squared multiple correlations as initial estimate
        # for communalities
        squared_multiple_corr = smc(corr)

        # replace the diagonal "ones" with the estimated communalities
        np.fill_diagonal(corr, squared_multiple_corr)

        sum_of_communalities = squared_multiple_corr.sum()
        error = sum_of_communalities
        error_threshold = 0.001
        i = 0
        while i < self.max_iter and error > error_threshold:
            # perform eigenvalue decomposition on the reduced correlation matrix
            eigenvalues, eigenvectors = np.linalg.eigh(corr)

            # sort the eigenvectors by eigenvalues from largest to smallest
            idx = eigenvalues.argsort()[::-1][: self.n_factors]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # update the loadings and calculate reproduced correlation matrix
            loadings = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))
            corr = np.dot(loadings, loadings.T)

            # the new estimate for the communalities is the diagonal
            # of the reproduced correlation matrix
            new_communalities = np.diag(corr)

            error = np.abs(new_communalities.sum() - sum_of_communalities)
            i += 1

        self.loadings_ = loadings
        self.communalities_ = new_communalities
        self.specific_variances_ = 1 - new_communalities
