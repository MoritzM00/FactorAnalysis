import factor_analyzer
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator

from factor_analysis import FactorAnalysis

pd.options.display.max_columns = 50
pd.set_option("expand_frame_repr", True)
pd.set_option("precision", 4)


@pytest.mark.skip
def test_estimator_check():
    return check_estimator(FactorAnalysis())


@pytest.mark.parametrize("n_factors", [2, 3])
def test_iris(n_factors):
    X = load_iris(as_frame=True).data

    fa = FactorAnalysis(n_factors=n_factors, rotation="varimax").fit(X)
    fa2 = factor_analyzer.FactorAnalyzer(
        n_factors=n_factors, rotation="varimax", method="minres"
    )
    fa2.fit(X)

    fa.summary()


def test_book_example():
    data = pd.read_csv(
        r".\data\application_example_backhaus_2021.CSV", sep=";", header=None
    )
    assert data.shape == (30, 5)
    fa = FactorAnalysis(n_factors=2).fit(data)
    fa.summary()


@pytest.mark.parametrize("rotation", ["varimax", "promax"])
def test_women_dataset(rotation):
    df = pd.read_csv(r".\data\women_track_records.csv")

    # first column is the country, so we drop it
    df.drop(columns=["COUNTRY"], inplace=True)
    fa = FactorAnalysis(n_factors=2, rotation=rotation).fit(df)
    fa.summary()


def test_fit_using_corr_mtx():
    R = np.array(
        [
            [1, 0.712, 0.961, 0.109, 0.044],
            [0.712, 1, 0.704, 0.138, 0.067],
            [0.961, 0.704, 1, 0.078, 0.024],
            [0.109, 0.138, 0.078, 1, 0.983],
            [0.044, 0.067, 0.024, 0.983, 1],
        ]
    )
    fa = FactorAnalysis(n_factors=2, is_corr_mtx=True, max_iter=50)
    fa.feature_names_in_ = ["Milky", "Melting", "Artificial", "Fruity", "Refreshing"]
    fa.fit(R)
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
    print()
    fa.summary()
    assert_allclose(fa.loadings_, loadings, atol=1e-3)


@pytest.mark.parametrize("n_factors", [1, 2, 3, 4])
def test_coastal_waves(n_factors):
    df = pd.read_csv(r".\data\coastal_waves_data.csv", sep=",")
    df.replace(-99.90, np.nan, inplace=True)
    df.drop("Date/Time", axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    fa = FactorAnalysis(n_factors=n_factors).fit(df)
    fa.summary()
