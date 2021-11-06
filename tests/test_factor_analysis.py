import factor_analyzer
import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import FactorAnalysis as FA

from factor_analysis import FactorAnalysis

X = load_iris().data


def test_iris():
    fa = FactorAnalysis(n_factors=2).fit(X)

    fa_fa = factor_analyzer.FactorAnalyzer(
        n_factors=2, rotation=None, method="principal", svd_method="lapack"
    )
    fa_fa.fit(X)
    fa_fa.loadings_ = np.delete(fa_fa.loadings_, obj=[2, 3], axis=1)
    print(fa_fa.loadings_)

    print(fa.loadings_ - fa_fa.loadings_)


def test_iris2():
    # compare with sklearn
    sk_fa = FA(n_components=2, svd_method="lapack")
    sk_fa.fit(X)
    print(sk_fa.components_)


def test_iris3():
    fa_fa = factor_analyzer.FactorAnalyzer(
        n_factors=2, rotation=None, method="principal", svd_method="lapack"
    )
    fa_fa.fit(X)
    print(fa_fa.loadings_)
