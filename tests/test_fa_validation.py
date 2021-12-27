import numpy as np
import pytest

from factor_analysis import FactorAnalysis

rng = np.random.default_rng(seed=0)


@pytest.fixture
def get_dummy_data():
    return rng.random(size=(30, 7))


def test_invalid_n_factors(get_dummy_data):
    X = get_dummy_data
    with pytest.raises(ValueError):
        FactorAnalysis(n_factors=0).fit(X)

    with pytest.raises(ValueError):
        # n_factors greater than number of features
        FactorAnalysis(n_factors=X.shape[1] + 1).fit(X)


def test_invalid_method(get_dummy_data):
    X = get_dummy_data
    with pytest.raises(ValueError):
        FactorAnalysis(method="foo").fit(X)

    # should not raise an exception
    FactorAnalysis(method="paf").fit(X)
    FactorAnalysis(method="PAF").fit(X)


def test_invalid_heywood_handling(get_dummy_data):
    X = get_dummy_data
    with pytest.raises(ValueError):
        FactorAnalysis(heywood_handling="bar").fit(X)

    FactorAnalysis(heywood_handling="COnTInue").fit(X)


def test_invalid_rotation(get_dummy_data):
    X = get_dummy_data
    with pytest.raises(ValueError):
        FactorAnalysis(rotation="foo").fit(X)
    with pytest.raises(ValueError):
        FactorAnalysis(rotation=7).fit(X)


def test_invalid_max_iter(get_dummy_data):
    X = get_dummy_data
    with pytest.raises(ValueError):
        FactorAnalysis(max_iter=0).fit(X)
    with pytest.raises(ValueError):
        FactorAnalysis(max_iter=9.2).fit(X)


def test_invalid_initial_comm(get_dummy_data):
    X = get_dummy_data
    n_features = X.shape[1]
    with pytest.raises(ValueError):
        FactorAnalysis(initial_comm="foo").fit(X)

    too_long = rng.random(size=n_features + 1)
    good = rng.random(n_features)
    too_short = rng.random(size=n_features - 1)
    with pytest.raises(ValueError):
        FactorAnalysis(initial_comm=too_long).fit(X)
    with pytest.raises(ValueError):
        FactorAnalysis(initial_comm=too_short).fit(X)

    FactorAnalysis(initial_comm=good).fit(X)
