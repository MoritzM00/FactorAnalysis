import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from factor_analysis import FactorAnalysis


def test_book_example_two_factors(get_app_ex_data, get_app_ex_loadings):
    data = get_app_ex_data
    corr = data.corr()
    loadings = get_app_ex_loadings
    communalities = [0.968, 0.526, 0.953, 0.991, 0.981]
    fa = FactorAnalysis(n_factors=2, is_corr_mtx=True).fit(corr)
    assert fa.loadings_.shape == (data.shape[1], 2)

    assert_allclose(fa.loadings_, loadings, atol=1e-3)
    assert_allclose(fa.communalities_, communalities, atol=1e-3)


def test_fit_using_corr_mtx(get_app_ex_loadings):
    loadings = get_app_ex_loadings
    # Table 7.3, p. 421 Backhaus Multivariate Analysis (2021)
    R = np.array(
        [
            [1, 0.712, 0.961, 0.109, 0.044],
            [0.712, 1, 0.704, 0.138, 0.067],
            [0.961, 0.704, 1, 0.078, 0.024],
            [0.109, 0.138, 0.078, 1, 0.983],
            [0.044, 0.067, 0.024, 0.983, 1],
        ]
    )
    fa = FactorAnalysis(n_factors=2, is_corr_mtx=True)
    fa.feature_names_in_ = ["Milky", "Melting", "Artificial", "Fruity", "Refreshing"]
    fa.fit(R)

    assert_allclose(fa.loadings_, loadings, atol=1e-3)


@pytest.fixture(scope="session")
def get_app_ex_data():
    # Table 7.3 on p.419, Backhaus Multivariate Analysis
    pth = os.path.join(
        os.getcwd(), "tests", "data", "application_example_backhaus_2021.csv"
    )
    data = pd.read_csv(pth, sep=";", header=None)
    return data


@pytest.fixture
def get_app_ex_loadings():
    # these loadings are taken from Backhaus 2021: Multivariate Analysis, p.419
    loadings = np.array(
        [
            [0.943, -0.280],
            [0.707, -0.162],
            [0.928, -0.302],
            [0.389, 0.916],
            [0.323, 0.936],
        ]
    )
    return loadings
