import factor_analyzer
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import load_iris
from sklearn.utils.estimator_checks import check_estimator

from factor_analysis import FactorAnalysis

pd.options.display.max_columns = 10


@pytest.mark.skip
def test_estimator_check():
    return check_estimator(FactorAnalysis())


@pytest.mark.parametrize("n_factors", [1, 2, 3, 4])
@pytest.mark.skip("FactorAnalyzer does not implement iterative paf")
def test_iris(n_factors):
    X = load_iris().data

    fa1 = FactorAnalysis(n_factors=n_factors).fit(X)
    fa2 = factor_analyzer.FactorAnalyzer(
        n_factors=n_factors, method="uls", svd_method="lapack", rotation=None
    ).fit(X)

    assert_allclose(fa1.loadings_, fa2.loadings_[:, :n_factors], atol=1e-2)


def test_book_example():
    data = pd.read_excel(r".\data\application_example_backhaus_2021.xlsx")
    assert data.shape == (29, 5)

    fa = FactorAnalysis(n_factors=2).fit(data)
    fa.summary()
    print(fa.transform(data))


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
    feature_names = ["Milky", "Melting", "Artificial", "Fruity", "Refreshing"]
    fa = FactorAnalysis(
        n_factors=2, is_corr_mtx=True, max_iter=50, feature_names=feature_names
    )
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

    fa = FactorAnalysis(n_factors=n_factors, feature_names=df.columns).fit(df)
    fa.summary()
