"""
========================================
Performing exploratory Factor Analysis
on a car data set.
========================================
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from factor_analysis import FactorAnalysis

df = pd.read_csv(r".\data\car_data.csv", sep=",", header=None)
df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
data = df.select_dtypes(np.number)
data = data.drop(columns=0, axis=0)

feature_names = {
    9: "wheel-base",
    10: "length",
    11: "width",
    12: "height",
    13: "curb-weight",
    16: "engine-size",
    20: "stroke",
    23: "city-mpg",
    24: "highway-mpg",
}
data.rename(feature_names, axis=1, inplace=True)


def plot_corr_heatmap(data):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr))
    sns.heatmap(data=corr, vmax=1, vmin=-1, cmap="RdBu", mask=mask)
    plt.show()


plot_corr_heatmap(data)
fa1 = FactorAnalysis(n_factors=2, rotation="varimax").fit(data)
fa1.print_summary()
print(np.mean(fa1.communalities_))

# in the correlation plot we see, that columns '12' and '20'
# are not significantly correlated to any other variable
# so we might drop it
data.drop(columns=[feature_names[12]], axis=0, inplace=True)
plot_corr_heatmap(data)

fa = FactorAnalysis(
    n_factors=4, method="paf", rotation="varimax", use_smc=True, max_iter=100
).fit(data)
print(np.mean(fa.communalities_))
fa.print_summary()
