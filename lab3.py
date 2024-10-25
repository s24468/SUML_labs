import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def make_data():
    data = pd.read_csv("datasets/DSP_6.csv")
    data.drop(columns=["Cabin"], inplace=True)
    return data
def show_Data(data):
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap="viridis")
    sns.jointplot(x="Age", y="Fare", data=data, color="red", kind="kde")
    plt.show()

    sns.set_style("whitegrid")
    sns.countplot(x="Survived", data=data)
    plt.show()

    sns.set_style("whitegrid")
    sns.countplot(x="Survived", hue="Sex", data=data)
    plt.show()


if __name__ == '__main__':
    show_Data(make_data())
