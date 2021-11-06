"""
Exploratory Factor Analysis.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
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
    method : str, default='paf'
        The fitting method, currently only (iterated) principal axis factoring (PAF)
        is supported.
    max_iter : int, default=50
        The maximum number of iterations. Set it to 1 if you do not want
        the iterated PAF.
    loadings_ : array_like, shape (n_features, n_factors)
        The factor loading matrix.
    communalities_ : array_like, shape (n_features,)
        The communalities (or common variance) of the variables. This is the part
        of the variance of the variables that can be explained by the factors.
    specific_variances_ : array_like, shape (n_features,)
        The specific variances for each variable. It is the part of the variance,
        that cannot be explained by the factors and is unique to each variable.
        Therefore it is also known as the 'uniqueness'.
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

        # using squared multiple correlations as initial estimate
        # for communalities
        squared_multiple_corr = smc(corr)

        # replace the diagonal "ones" with the estimated communalities
        np.fill_diagonal(corr, squared_multiple_corr)

        sum_of_communalities = squared_multiple_corr.sum()
        error = sum_of_communalities
        error_threshold = 0.001
        for _ in range(self.max_iter):
            if error < error_threshold:
                break
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
        else:
            raise ConvergenceWarning(
                "Iterated principal axis factoring did not converge. "
                "Consider increasing the `max_iter` parameter of the model."
            )

        self.loadings_ = loadings
        self.communalities_ = new_communalities
        self.specific_variances_ = 1 - new_communalities

    def get_covariance(self):
        """
        Returns the reproduced (model) covariance matrix.

        Returns
        -------
        cov : array_like, shape (n_features, n_features)
            The model covariance matrix
        """
        return np.dot(self.loadings_, self.loadings_.T) + np.diag(
            self.specific_variances_
        )

    def summary(self, verbose=False):
        """
        Returns a dataframe that contains the loading matrix, the communalities
        and the specific variances.

        Returns
        -------
        df : DataFrame
            The summary dataframe.
        """
        column_names = [f"Factor {i}" for i in range(1, self.n_factors + 1)] + [
            "Communalities",
            "Specific variances",
        ]
        idx = [f"X{i}" for i in range(1, self.n_features_ + 1)]
        df = pd.DataFrame(
            data=np.concatenate(
                (
                    self.loadings_,
                    self.communalities_.reshape(-1, 1),
                    self.specific_variances_.reshape(-1, 1),
                ),
                axis=1,
            ),
            columns=column_names,
            index=idx,
        )
        if verbose:
            print(f"Number of samples: {self.n_samples_}")
            print(f"Number of features: {self.n_features_} \n")
            print("Summary of estimated paramters: \n")
            print(df)
        return df
