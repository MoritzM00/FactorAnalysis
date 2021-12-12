"""
========================================
Performing exploratory Factor Analysis
on the big five personalities dataset
========================================
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# load and clean data
df = pd.read_csv(r".\data\big_five_personality.csv", sep="\t")
df = df[(df["IPC"] == 1)]  # cleaning repeaters (same IP)
df.dropna(inplace=True)
df.drop(df.columns[50:], axis=1, inplace=True)
X_train, X_test = train_test_split(df, train_size=0.8, test_size=0.2)

# Fit the model
# fa = FactorAnalysis(n_factors=5, rotation="varimax")
# fa.fit(df)
# fa.summary()

# test kmo
# print(
#    f"KMO for all variables > 0.6?: {np.all(FactorAnalysis.calculate_kmo(df)[0] > 0.6)}"
# )
plt.figure(figsize=(16, 16))


corr = X_train.corr()
mask = np.triu(np.ones_like(corr))

sns.heatmap(data=corr, vmax=1, vmin=-1, cmap="RdBu", mask=mask)
# plt.savefig("corr_heatmap")
plt.show()
