# Factor Analysis

![Tests](https://github.com/MoritzM00/FactorAnalysis/actions/workflows/tests.yml/badge.svg)
![License](https://img.shields.io/github/license/MoritzM00/FactorAnalysis?color=blue)

## Overview

This project implements factor analysis using the principal components (PC) and
the iterated principal axis factoring (PAF) method with optional orthogonal
rotations. The rotations are implemented via the
[factor_analyzer](https://github.com/EducationalTestingService/factor_analyzer) package.

## Usage
First, instantiate a FactorAnalysis object
```python
from factor_analysis import FactorAnalysis
fa = FactorAnalysis(n_factors=2, method="pc", rotation="varimax")
```
Then you can fit the model to a data set by executing
```python
fa.fit(data)
```
and have a look the estimated parameters (e.g. loadings, communalities) with
```python
fa.print_summary()
```
or by directly accessing the fitted attributes, which follow the naming convention
used by the [scikit-learn](https://github.com/scikit-learn/scikit-learn) package
(i.e. the method name ends with a `_`. For example:
```python
fa.loadings_
fa.communalities_
```
To see which parameters the constructor accepts and which attributes are available,
you can have a look at the `FactorAnalysis` class documentation.

The reproduced correlation matrix can be retrieved with the
`get_reprod_corr` method.

Lastly, you can compute the factor scores with
```python
fa.transform(data)
```
where `data` must be the same data used in the fit method.

You can also use the convenience method `fit_transform` to directly compute
the factor scores after fitting.
```python
scores = fa.fit_transform(data)
```


## Dependencies

The required packages for the actual factor analysis class are
- scikit-learn
- numpy
- pandas
- factor_analyzer

In order to use the plotting functions in `factor_analysis.plotting` and
to be able to execute the examples it is required to have
- jupyter
- matplotlib
- seaborn

installed.


To install the required packages, run (after activating a virtual environment)

```bash
pip install -r requirements.txt
```

If you want to run tests and commit to the project, then additionally run

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Contributing

This project uses [Poetry](https://python-poetry.org/ "python-poetry.org") as its dependency manager.
If you do not already have it, install Poetry.
For instructions, follow this link:

https://github.com/python-poetry/poetry/blob/master/README.md#installation


Once you have Poetry installed, clone the repository and execute:

```bash
poetry config virtualenvs.in-project true
poetry install
```

This creates a virtual environment and installs the required packages.
