import numpy as np
from matplotlib import pyplot as plt


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


def plot_loadings_heatmap(X, methods, figsize=(10, 8), fa_params=None):
    """
    Plot the loadings heatmap of multiple unfitted FactorAnalysis instances.
    The Instances will be fitted using the data `X`.

    This is useful for comparing different variants of factor analysis models for
    different parameters. The n_factors parameter has to be set, other parameters
    of the factor analysis instances can be set using the other_params dict.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data that is used in the fit method.
    methods : array_like of tuples (str, FactorAnalysis)
        This contains the variants of factor analysis instances. Each element
        is a tuple (str, FactorAnalysis) where the first element is a short description
        and the second element is the unfitted FactorAnalysis instance.
        You only need to specify the type of algorithm used in each instance.
        The other parameters can be set using the fa_params dict containing
        key-value pairs, e.g. `fa_params={'n_factors': 2}`.
    figsize : tuple of int
        The size of the figure.
    fa_params : dict
        This is used to batch-set the parameters of the factor analysis instances.

    Returns
    -------
    None
    """
    if fa_params is None:
        fa_params = {}
    fig, axes = plt.subplots(ncols=len(methods), figsize=figsize)
    if len(methods) == 1:
        # make it iterable
        axes = [axes]
    for ax, (method, fa) in zip(axes, methods):
        fa.set_params(**fa_params)
        fa.fit(X)
        feature_names = getattr(
            fa, "feature_names_in_", [f"X{i}" for i in range(1, X.shape[1])]
        )
        loadings = fa.loadings_
        vmax = np.abs(loadings).max()
        ax.imshow(loadings, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
        ax.set_yticks(np.arange(len(feature_names)))
        if ax.get_subplotspec().is_first_col():
            ax.set_yticklabels(feature_names)
        else:
            ax.set_yticklabels([])
        ax.set_title(str(method))
        ax.set_xticks(range(fa.n_factors))
        ax.set_xticklabels([f"Factor {i}" for i in range(1, fa.n_factors + 1)])
    fig.suptitle("Factorloadingsmatrix")
    plt.tight_layout()
    plt.show()
