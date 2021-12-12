"""
========================================
Performing exploratory Factor Analysis
on the big five personalities dataset
========================================
7. Attribute information:
   1. mcv	mean corpuscular volume
   2. alkphos	alkaline phosphotase
   3. sgpt	alamine aminotransferase
   4. sgot 	aspartate aminotransferase
   5. gammagt	gamma-glutamyl transpeptidase
   6. drinks	number of half-pint equivalents of alcoholic beverages
                drunk per day
   7. selector  field used to split data into two sets
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from factor_analysis import FactorAnalysis

data = pd.read_csv(r".\data\liver_disorder.csv", sep=",", header=None)

# there are some duplicates rows, so we drop them
data.drop_duplicates(inplace=True)

# plot correlation heatmap
corr = data.corr()
mask = np.triu(np.ones_like(corr))
sns.heatmap(data=corr, vmax=1, vmin=-1, cmap="RdBu", mask=mask)
plt.show()

fa = FactorAnalysis(n_factors=2).fit(data)
fa.print_summary()
