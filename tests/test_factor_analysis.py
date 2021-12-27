import os

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator

from factor_analysis import FactorAnalysis


@pytest.mark.skip
def test_estimator_check():
    return check_estimator(FactorAnalysis())


@pytest.mark.parametrize("n_factors", [2, 3])
@pytest.mark.skip
def test_iris(n_factors):
    X = load_iris(as_frame=True).data
    fa = FactorAnalysis(n_factors=n_factors).fit(X)
    fa.print_summary()


def test_book_example_two_factors(get_app_ex_data, get_app_ex_loadings):
    data = get_app_ex_data
    corr = data.corr()
    loadings = get_app_ex_loadings
    communalities = [0.968, 0.526, 0.953, 0.991, 0.981]
    fa = FactorAnalysis(n_factors=2, is_corr_mtx=True).fit(corr)
    assert_allclose(fa.loadings_, loadings, atol=1e-2)
    assert_allclose(fa.communalities_, communalities, atol=1e-2)


@pytest.mark.parametrize("rotation", ["varimax", "oblimax"])
def test_women_dataset(rotation):
    pth = os.path.join(os.getcwd(), "data", "women_track_records.csv")
    df = pd.read_csv(pth)

    # first column is the country, so we drop it
    df.drop(columns=["COUNTRY"], inplace=True)
    fa = FactorAnalysis(n_factors=2, rotation=rotation).fit(df)
    fa.print_summary()
    # TODO: assert something


def test_fit_using_corr_mtx(get_app_ex_loadings):
    loadings = get_app_ex_loadings
    R = np.array(
        [
            [1, 0.712, 0.961, 0.109, 0.044],
            [0.712, 1, 0.704, 0.138, 0.067],
            [0.961, 0.704, 1, 0.078, 0.024],
            [0.109, 0.138, 0.078, 1, 0.983],
            [0.044, 0.067, 0.024, 0.983, 1],
        ]
    )
    print(np.max(np.abs(R - np.eye(5)), axis=0))
    fa = FactorAnalysis(n_factors=2, is_corr_mtx=True)
    fa.feature_names_in_ = ["Milky", "Melting", "Artificial", "Fruity", "Refreshing"]
    fa.fit(R)

    assert_allclose(fa.loadings_, loadings, atol=1e-2)


def test_optimal_corr_mtx():
    R = [
        [1, 1, 0, 0, 0.05],
        [1, 1, 0, 0.05, 0.05],
        [0.05, 0.05, 1, 1, 0.9],
        [0.05, 0, 1, 1, 0.9],
        [0, 0.05, 0.9, 0.9, 1],
    ]
    fa = FactorAnalysis(
        n_factors=2, is_corr_mtx=True, method="paf", initial_comm="ones"
    ).fit(R)
    fa.print_summary()


@pytest.mark.parametrize("n_factors", [1, 2, 3, 4])
@pytest.mark.skip("no assertions")
def test_coastal_waves(n_factors):
    pth = os.path.join(os.getcwd(), "data", "coastal_waves_data.csv")
    df = pd.read_csv(pth, sep=",")
    df.replace(-99.90, np.nan, inplace=True)
    df.drop("Date/Time", axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # fa = FactorAnalysis(n_factors=n_factors).fit(df)
    # fa.summary()


@pytest.fixture(scope="session")
def get_app_ex_data():
    # Table 7.2 on p.387, Backhaus Multivariate Analysis
    pth = os.path.join(os.getcwd(), "data", "application_example_backhaus_2021.csv")
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
