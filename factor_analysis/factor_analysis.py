"""
Exploratory Factor Analysis.
"""

import numpy as np
import pandas as pd
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

    def __init__(
        self,
        n_factors,
        method="paf",
        max_iter=50,
        is_corr_mtx=False,
        feature_names=None,
    ):
        self.n_factors = n_factors
        self.method = method
        self.max_iter = max_iter
        self.is_corr_mtx = is_corr_mtx
        self.features_names = feature_names

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

        if self.is_corr_mtx:
            self.corr_ = X.copy()
            self.n_features_ = X.shape[0]
        else:
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
                self.converged_ = True
                break
            # perform eigenvalue decomposition on the reduced correlation matrix
            eigenvalues, eigenvectors = np.linalg.eigh(corr)

            # sort the eigenvectors by eigenvalues from largest to smallest
            idx = eigenvalues.argsort()[::-1][: self.n_factors]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # update the loadings and calculate reproduced correlation matrix (R_hat)
            loadings = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))
            R_hat = np.dot(loadings, loadings.T)

            # the new estimate for the communalities is the diagonal
            # of the reproduced correlation matrix
            new_communalities = np.diag(R_hat)

            # update communalities in the correlation matrix
            np.fill_diagonal(corr, new_communalities)

            error = np.abs(new_communalities.sum() - sum_of_communalities)
        else:
            self.converged_ = False

        self.loadings_ = loadings
        self.communalities_ = new_communalities
        self.specific_variances_ = 1 - self.communalities_

        # proportion of variance explained by each factor
        self.var_explained_ = eigenvalues / np.trace(corr)
        # cumulative variance explained
        self.cum_var_explained_ = eigenvalues.sum() / self.n_features_

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
        factors = [f"Factor {i}" for i in range(1, self.n_factors + 1)]
        column_names = factors + [
            "Communalities",
            "Specific variances",
        ]
        if self.features_names is not None:
            idx = self.features_names.copy()
        else:
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

        var_df = pd.DataFrame(
            self.var_explained_.reshape(-1, 1),
            index=factors,
            columns=["Proportion of variance explained"],
        )
        # calculate difference between correlation matrix and reproduced corr mtx
        diff = np.sum(np.abs(self.corr_ - self.get_covariance()))
        if verbose:
            print("Summary of estimated paramters: \n")
            print(df, "\n")
            print(var_df)
            print(
                f"\nCumulative variance explained: {self.cum_var_explained_ * 100:.2f}%"
            )
            print(
                f"Absolute difference between reproduced \n"
                f"and empirical correlation matrix: {diff:.4f}"
            )
        return df
