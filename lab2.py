import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from platform import python_version

#from scipy.stats import pearsonr
python_version()


base_data = pd.read_csv("datasets/DSP_4.csv")
print(base_data)
df = pd.read_csv("datasets/DSP_4.csv", sep=";")
print(df)
print(df.columns)

print(df["wiek"].mean())

df_2 = df.fillna(df.mean())
print(df_2)

print(df_2.isnull().any())


# - - - - - - -- - - - - - -- podstawowe statystyki opisowe - - - - -- - - -


print(df_2["wiek"].mean())
print(round(df_2["wiek"].mean(),2))
print(df_2["wiek"].median())
print(df_2["wiek"].max())
print(df_2["wiek"].min())
print(df_2["wiek"].var())


print(df_2.wiek.max() - df_2.wiek.min())

print(df_2.wiek.quantile([0.25, .5,.75]))

print(round(df_2.wiek.std(),2))

print(round(df_2.describe(),2))
print(df_2.info())

print(df_2.wiek.groupby([df_2.objawy]).describe())
print(df_2.corr())
# sns.heatmap(df_2.corr)

plt.figure(figsize=(10,5))