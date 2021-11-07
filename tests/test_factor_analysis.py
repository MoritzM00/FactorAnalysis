import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from factor_analysis import FactorAnalysis

X = load_iris().data


def test_iris():
    fa = FactorAnalysis(n_factors=3, method="paf", max_iter=50)
    fa.fit(X)

    # fa_fa = factor_analyzer.FactorAnalyzer(
    #    n_factors=2, rotation=None, method="principal", svd_method="lapack"
    # )
    # fa_fa.fit(X)
    # fa_fa.loadings_ = np.delete(fa_fa.loadings_, obj=[2, 3], axis=1)
    # print(fa_fa.loadings_)

    # print(fa.loadings_ - fa_fa.loadings_)
    fa.summary(verbose=True)


def test_book_example():
    data = pd.read_excel(r".\data\application_example_backhaus_2021.xlsx")
    assert data.shape == (29, 5)

    fa = FactorAnalysis(n_factors=2).fit(data)
    fa.summary()


def test_women_dataset():
    df = pd.read_csv(r".\data\women_track_records.csv")

    # first column is the country, so we drop it
    df.drop(columns=["COUNTRY"], inplace=True)
    X = df.to_numpy()
    fa = FactorAnalysis(n_factors=4).fit(X)
    print(fa.loadings_)


def test_corr_mtx():
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
        n_factors=3, is_corr_mtx=True, max_iter=50, feature_names=feature_names
    )
    fa.fit(R)
    fa.summary()


@pytest.mark.parametrize("n_factors", [1, 2, 3])
def test_coastal_waves(n_factors):
    df = pd.read_csv(r".\data\coastal_waves_data.csv", sep=",")
    df.replace(-99.90, np.nan, inplace=True)
    df.drop("Date/Time", axis=1, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    fa = FactorAnalysis(n_factors=n_factors, feature_names=df.columns).fit(df)
    fa.summary()
