"""
========================================
Performing exploratory Factor Analysis
on a wines data set.
========================================

This data is the result of a chemical analysis of
wines grown in the same region in Italy but derived
from three different cultivars.
The analysis determined the quantities of 13 constituents
found in each of the three types of wines.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from factor_analysis import FactorAnalysis
from factor_analysis.utils import scree_plot

np.set_printoptions(precision=4, suppress=True)

data = pd.read_csv(r".\data\wine.csv", sep=",", header=None)

target = data[0].copy()
data.drop(columns=0, axis=0, inplace=True)
feature_names = [
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline",
]
data.columns = feature_names

chi2, p_value = FactorAnalysis.calculate_bartlett_sphericity(data)
print(f"Bertlett Sphericity Test: {chi2=}, {p_value=}")
kmos, overall_kmo = FactorAnalysis.calculate_kmo(data)
print("KMO values:")
print(kmos)
print(f"overall KMO: {overall_kmo}")

# plot the correlation matrix as a heatmap
corr = data.corr()
mask = np.triu(np.ones_like(corr))
fig = plt.figure(figsize=(13, 13))
sns.heatmap(data=corr, vmax=1, vmin=-1, cmap="RdBu", mask=mask)
plt.show()

fa = FactorAnalysis(n_factors=12, use_smc=False).fit(data)

fig, ax = plt.subplots(figsize=(6, 4))
scree_plot(fa.eigenvalues_, axis=ax)
plt.show()

# according to Kaiser's Eigenvalue Criterion, it should be enough to only
# keep 3 factors

fa = FactorAnalysis(n_factors=3, method="paf", rotation="varimax").fit(data)
fa.print_summary(force_full_print=True, precision=4)
