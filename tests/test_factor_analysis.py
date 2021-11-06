import pandas as pd
from sklearn.datasets import load_iris

from factor_analysis import FactorAnalysis

X = load_iris().data


def test_iris():
    fa = FactorAnalysis(n_factors=2)
    fa.fit(X)

    # fa_fa = factor_analyzer.FactorAnalyzer(
    #    n_factors=2, rotation=None, method="principal", svd_method="lapack"
    # )
    # fa_fa.fit(X)
    # fa_fa.loadings_ = np.delete(fa_fa.loadings_, obj=[2, 3], axis=1)
    # print(fa_fa.loadings_)

    # print(fa.loadings_ - fa_fa.loadings_)
    print(fa.loadings_)


def test_book_example():
    data = pd.read_excel(r".\data\application_example_backhaus_2021.xlsx")
    assert data.shape == (29, 5)

    fa = FactorAnalysis(n_factors=2).fit(data)
    print(fa.corr_)
    print(fa.loadings_)


def test_women_dataset():
    df = pd.read_csv(r".\data\women_track_records.csv")

    # first column is the country, so we drop it
    df.drop(columns=["COUNTRY"], inplace=True)
    X = df.to_numpy()
    fa = FactorAnalysis(n_factors=4).fit(X)
    print(fa.loadings_)
