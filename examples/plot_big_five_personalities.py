"""
========================================
Performing exploratory Factor Analysis
on the big five personalities dataset
========================================
"""
import numpy as np
import pandas as pd

from factor_analysis import FactorAnalysis

# load and clean data
df = pd.read_csv(r".\data\big_five_personality.csv", sep="\t")
df = df[(df["IPC"] == 1)]  # cleaning repeaters (same IP)
df.dropna(inplace=True)
df.drop(df.columns[50:107], axis=1, inplace=True)
df.drop(df.columns[51:], axis=1, inplace=True)
df.drop("country", axis=1, inplace=True)

# Fit the model
fa = FactorAnalysis(n_factors=5, rotation="varimax")
fa.fit(df)
fa.summary()

# test kmo
print(
    f"KMO for all variables > 0.6?: {np.all(FactorAnalysis.calculate_kmo(df)[0] > 0.6)}"
)
