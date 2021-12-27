"""
Exploratory Factor Analysis.
"""

import warnings

import numpy as np
import pandas as pd
from factor_analyzer import Rotator
from factor_analyzer.rotator import ORTHOGONAL_ROTATIONS
from numpy.linalg import LinAlgError
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_array, check_is_fitted

from factor_analysis.utils import smc, standardize


class FactorAnalysis(BaseEstimator, TransformerMixin):
    """
    The linear factor analysis model.

    TODO: docstring

    Parameters
    ----------
    n_factors : int
        The number of factors.
    method : str, default='paf'
        The fitting method. Currently (iterated) principal axis factoring and
        principal components method is implemented.
        For PAF use method='paf'
        For Principal Comp. use method='pc'
    rotation : str, default=None
        Sets the factor rotation method. If you do not want to rotate the factors
        after factor extraction, leave it at default=None.
    max_iter : int, default=50
        The maximum number of iterations. Ignored if method='pc'.
        Set it to 1 if you do not want the iterated PAF.
    is_corr_mtx : bool, default=False
        If True, the passed data `X` is assumed to be the correlation matrix.
    use_smc : bool, default=True
        If true, use squared multiple correlations as initial estimate
        for the communalities.
    heywood_handling : str, default='continue'
        Change the behavior in PAF, when a Heywood case is encountered.
        If `heywood_handling` is set to 'continue', then the communalities
        that are greater or equal to 1.0 are set to .99 and iteration will
        continue. Otherwise, if `heywood_handling` is set to 'stop', iteration
        will stop and the calculated laodings should be discarded.

    Attributes
    ----------
    loadings_ : ndarray, shape (n_features, n_factors)
        The factor loading matrix.
    communalities_ : ndarray, shape (n_features,)
        The communalities (or common variance) of the variables. This is the part
        of the variance of the variables that can be explained by the factors.
    specific_variances_ : ndarray, shape (n_features,)
        The specific variances for each variable. It is the part of the variance,
        that cannot be explained by the factors and is unique to each variable.
        Therefore it is also known as the 'uniqueness' of a variable.
    corr_ : ndarray, shape (n_features, n_features)
        The empirical correlation matrix of the data.
    eigenvalues_ : ndarray, shape (n_factors,)
        The eigenvalues of the selected factors.
    n_samples_ : int
        The number of samples. Only available if `is_corr_mtx` is equal to False.
    n_features_ : int
        The number of features.
    n_iter_ : int
        The number of iterations needed, until convergence criterion was fulfilled.
    feature_names_in_ : ndarray, shape (n_features,)
        The feature names seen during fit. If `X` is a DataFrame, then it will
        use the column names as feature names, if all columns are strings.
    """

    def __init__(
        self,
        n_factors=2,
        method="paf",
        rotation=None,
        max_iter=50,
        is_corr_mtx=False,
        use_smc=True,
        heywood_handling="continue",
    ):
        self.n_factors = n_factors
        self.method = method
        self.rotation = rotation
        self.max_iter = max_iter
        self.is_corr_mtx = is_corr_mtx
        self.use_smc = use_smc
        self.heywood_handling = heywood_handling

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
        X = self._validate_input(X)

        if self.is_corr_mtx:
            self.corr_ = X.copy()
            self.n_features_ = X.shape[0]
        else:
            self.n_samples_, self.n_features_ = X.shape

            # standardize data
            Z, *_ = standardize(X)

            # calculate initial correlation matrix
            corr = np.dot(Z.T, Z) / (self.n_samples_ - 1)
            self.corr_ = corr.copy()

        if self.method == "paf":
            if self.use_smc:
                try:
                    start = smc(self.corr_)
                except LinAlgError:
                    # use maximum absolute correlation in each row
                    start = np.max(
                        np.abs(self.corr_ - np.eye(self.n_features_)), axis=0
                    )
            else:
                start = np.repeat(1, self.n_features_)
            self._fit_principal_axis(start_estimate=start)
        else:
            self._fit_principal_component()

        # calculate class attributes after fitting

        # sum of the squared loadings for each variable (each row)
        # are the final communalities
        squared_loadings = self.loadings_ ** 2
        self.communalities_ = np.sum(squared_loadings, axis=1)
        self.specific_variances_ = 1 - self.communalities_

        # the eigenvalue of a factor is the sum of the squared loadings
        # in each column
        self.eigenvalues_ = np.sum(squared_loadings, axis=0)
        # proportion of variance explained by each factor
        self.var_explained_ = self.eigenvalues_ / self.n_features_

        if self.rotation is not None:
            self.loadings_ = Rotator(method=self.rotation).fit_transform(self.loadings_)

        if self.n_factors > 1:
            # update loading signs to match column sums
            # this is to ensure that signs align with package factor_analyzer
            signs = np.sign(self.loadings_.sum(0))
            signs[(signs == 0)] = 1
            self.loadings_ = np.dot(self.loadings_, np.diag(signs))
        return self

    def transform(self, X):
        """
        Computes the factors scores using Thompson's regression method.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        F : array_like, shape (n_samples, n_factors)
            The factor scores
        """
        check_is_fitted(self)

        X = check_array(X, copy=True)
        Z, *_ = standardize(X)

        # Regression method
        inv_corr = np.linalg.inv(np.cov(Z, rowvar=False))
        F = np.linalg.multi_dot([Z, inv_corr, self.loadings_])
        return F

    def _fit_principal_axis(self, start_estimate):
        """
        Fits the factor model using the principal axis method.

        This method replaces the diagonal of the correlation matrix
        with the given start estimate and performs a iterated
        eigenvalue decomposition until a convergence criterion
        or the maximum number of iterations is reached.
        """
        corr = self.corr_.copy()

        # replace the diagonal "ones" with the initial estimate of the communalities
        np.fill_diagonal(corr, start_estimate)

        old_sum = start_estimate.sum()
        error = old_sum
        error_threshold = 0.001
        for i in range(self.max_iter):
            if error < error_threshold:
                self.n_iter_ = i
                break
            # perform eigenvalue decomposition on the reduced correlation matrix
            eigenvalues, eigenvectors = np.linalg.eigh(corr)

            # sort the eigenvectors by eigenvalues from largest to smallest
            eigenvalues = eigenvalues[::-1][: self.n_factors]
            eigenvectors = eigenvectors[:, ::-1][:, : self.n_factors]

            if np.any(eigenvalues < 0):
                raise ValueError(
                    "Fit using the PAF algorithm with SMC as starting value "
                    "failed. Try again with `use_smc=False`."
                )

            # update the loadings
            loadings = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))

            # the new estimate for the communalities is the sum of squares over
            # each row in the loading matrix
            comm = np.sum(loadings ** 2, axis=1)
            if np.any(comm >= 1):
                if self.heywood_handling == "continue":
                    # strip Heywood cases to .999 before iteration continues
                    comm = np.where(comm >= 1.0, 0.999, comm)
                else:
                    warnings.warn(
                        f"Heywood Case in Iteration {i} found. "
                        "Based on the parameter 'heywood_handling' "
                        "the iteration stopped. Try again with "
                        "either heywood_handling='continue' or "
                        f"with max_iter={i - 1}."
                    )
                    self.n_iter_ = i
                    break

            # update communalities in the correlation matrix
            np.fill_diagonal(corr, comm)

            # update error variables
            new_sum = comm.sum()
            error = np.abs(new_sum - old_sum)
            old_sum = new_sum
        else:
            self.n_iter_ = self.max_iter
            if self.max_iter != 1:
                warnings.warn(
                    "PAF algorithm did not converge. Consider increasing the `max_iter` parameter",
                    ConvergenceWarning,
                )

        self.loadings_ = loadings

    def _fit_principal_component(self):
        """
        Fit the factor analysis model using the principal component method.
        (Not principal component analysis)

        This is equivalent to using _fit_principal_axis with starting value 1 and
        max_iter = 1.
        """
        corr = self.corr_.copy()

        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        # sort descending
        eigenvalues = eigenvalues[::-1][: self.n_factors]
        eigenvectors = eigenvectors[:, ::-1][:, : self.n_factors]

        self.loadings_ = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))

    def get_reprod_corr(self, reduced=False):
        """
        Returns the reproduced correlation matrix.

        Parameters
        ----------
        reduced : bool, default=False
            If reduced is set to True, then return the
            reduced correlation matrix, i.e. the diagonal elements
            of the model correlation matrix are the estimated
            communalities.

        Returns
        -------
        reproduced_corr : array_like, shape (n_features, n_features)
            The reproduced correlation matrix.
        """
        reproduced_corr = np.dot(self.loadings_, self.loadings_.T)
        if not reduced:
            reproduced_corr += np.diag(self.specific_variances_)
        return reproduced_corr

    def print_summary(self, file=None, force_full_print=True, precision=4):
        """
        Prints a summary of the estimated parameters of the
        factor analysis model including:
         - loadings
         - communalities
         - specific variances
         - Root mean squared error of residuals

         If the fitting method was principal components (PC) then the following is reported as well:

         - eigenvalues of the factors
         - % of variance explained by each factor
         - cumulative % explained

        Parameters
        ----------
        file : File, default=None
            Optional. If specified, then the output is written to the
            given file. The user has to take care of opening and closing
            the file. Surround the 'print_summary(..., file=f)' statement with a
            ``with open(file, 'a') as f:`` and then pass f as argument to this method.
        force_full_print : bool, default=True
            If True, then it prints the dataframes with no column or row
            width limitations. Otherwise it will print dots if the width
            of the dataframe is too big.
        precision : int, default=4
            Can be used to specify the precision for printing floats.
        """
        check_is_fitted(self)
        factors = [
            f"{'R' if self.rotation is not None else ''}F{i}"
            for i in range(1, self.n_factors + 1)
        ]
        column_names = factors + ["Communality", "Specific Variance"]
        if hasattr(self, "feature_names_in_"):
            idx = self.feature_names_in_
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
        factor_info = pd.DataFrame(
            data={
                "Eigenvalue": self.eigenvalues_,
                "Variance Explained": self.var_explained_,
                "Cumulative Variance": np.cumsum(self.var_explained_),
            },
            index=factors,
        )
        option_precision = ["display.precision", precision]
        option_full_print = [
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            2000,
            "display.max_colwidth",
            None,
        ]
        options = (
            option_precision + option_full_print
            if force_full_print
            else option_precision
        )

        # define custom print function, if the user wants to
        # write the output to a file
        def my_print(*args):
            if file:
                # write output to specified file
                print(*args, file=file)
            else:
                # else just print it to std.out
                print(*args)

        with pd.option_context(*options):
            # print all parameters of the estimator
            with config_context(print_changed_only=False):
                my_print(f"Call fit on {self}")
            my_print(
                f"Number of samples: {self.n_samples_ if not self.is_corr_mtx else 'NA'}"
            )
            my_print(f"Number of features: {self.n_features_}")
            my_print("Summary of estimated parameters: \n")
            my_print(df, "\n")
            if self.method == "pc":
                my_print(factor_info, "\n")
            if self.method == "paf" and self.max_iter > 1:
                my_print(f"Number of iterations: {self.n_iter_}")
            my_print(f"Root mean squared error of residuals: {self.get_rmse():.4f}")

    def get_rmse(self):
        """
        Calculate the root mean squared error of residuals, which is defined as
        the square root of the sum of squared deviations of the empirical
        correlation matrix and the reproduced correlation matrix.
        Because the diagonal elements are estimated exactly and the matrizes
        are symmetric, only calculate the error for the upper- (or lower-)
        triangular part of the matrizes.
        Therefore the correction term t = p(p - 1) / 2, where p is the number of
        features.

        Returns
        -------
        rmse : float
            The root mean squared error of residuals.
        """
        R = self.corr_.copy()
        R_hat = self.get_reprod_corr()
        # 2 * t = number of off-diagonal elements
        t = self.n_features_ * (self.n_features_ - 1) / 2

        sum_of_squared_residuals = 0
        # iterate over the upper-triangular part of both matrizes
        # and calculate the sum of squared residuals
        for i in range(1, self.n_features_):
            for j in range(i + 1, self.n_features_):
                sum_of_squared_residuals += (R[i][j] - R_hat[i][j]) ** 2
        return np.sqrt(sum_of_squared_residuals / t)

    def _validate_input(self, X):
        """
        Validates the input for correct specification. Sets the feature_names_in_
        attribute if X was a DataFrame with string columns
        """
        if isinstance(X, pd.DataFrame) and X.columns.inferred_type == "string":
            # set the feature_names_in attribute
            # only if the column names are all strings
            # use the column names as feature names
            self.feature_names_in_ = X.columns.copy()

        X = check_array(X, copy=True)

        if self.method is None or not isinstance(self.method, str):
            raise ValueError(f"Unsupported method specified: {self.method}")
        self.method = self.method.lower()
        POSSIBLE_METHODS = ["paf", "pc"]
        if self.method not in POSSIBLE_METHODS:
            raise ValueError(
                f"Method {self.method} is currently not supported."
                f"It has to be one of {POSSIBLE_METHODS}."
            )
        if self.heywood_handling not in ["continue", "stop"]:
            raise ValueError("heywood_handling must be either 'continue' or 'stop'.")
        if self.max_iter < 1:
            raise ValueError(
                f"max_iter has to be an integer greater or equal to 1, "
                f"but got {self.max_iter} instead."
            )
        if self.n_factors < 1:
            raise ValueError(
                f"n_factor must be an integer greater or equal to 1, "
                f"but got {self.n_factors} instead."
            )
        if self.rotation not in [None] + ORTHOGONAL_ROTATIONS:
            raise ValueError(
                f"Rotation method must be None"
                f" or one of the following orthogonal rotations:"
                f" {ORTHOGONAL_ROTATIONS}"
            )
        self.rotation = self.rotation.lower() if self.rotation else None
        return X
