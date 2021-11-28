"""
Utils for factor analysis class.
"""

import numpy as np


def smc(corr):
    """
    Calculates the squared multiple correlations.

    Parameters
    ----------
    corr : array_like, shape (p, p)
        The correlation matrix.

    Returns
    -------
    smc : array_like, shape (p, 1)
        The squared multiple correlations
    """
    inv_corr = np.linalg.inv(corr)
    return 1 - 1 / np.diag(inv_corr)


def standardize(X):
    """
    Standardizes X.


    Parameters
    ----------
    X : array_like
        The data to be standardized.

    Returns
    -------
    Z : ndarray
        The standardized data.
    mean : float
        The mean of the data.
    std : float
        The standard deviation of the data.

    """
    mean = np.mean(X, axis=0)
    std = X.std(axis=0)
    Z = (X - mean) / std

    return Z, mean, std


def scree_plot(eigenvalues, axes):
    """
    Plots the scree plot of the eigenvalues onto the given axes

    Parameters
    ----------
    eigenvalues : array_like
        The eigenvalues to plot.
    axes : matplotlib.axes.Axes
        The Axes to plot onto.

    Returns
    -------
    None

    """
    x = np.arange(len(eigenvalues)) + 1
    axes.plot(x, eigenvalues, "bo-", linewidth=2)
    axes.axhline(1, c="g")
    axes.set_title("Scree Plot")
    axes.set_xlabel("Factor")
    axes.set_ylabel("Eigenvalue")
