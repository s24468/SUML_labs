
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

def zadanie1():
    imported_model = pickle.load(open("our_model.pkl", "rb"))
    X_unknown = np.array([2.78])
    print('y = a*X + b')
    print(imported_model.coef_[0][0] * X_unknown[0] + imported_model.intercept_[0])




X = 21
y = 21.65

# Add row:
# Read the file as DataFrame
df = pd.read_csv('10_points.csv')

# Add values as last row to DF
df.loc[len(df.index)] = [X, y]

# Overwrite old file with new values (reset indexes)
df.to_csv('path2csv', index=False)

# - - - - - - - - - - - - - - - - - -  - - - - - - - - - - - - - - -- - - - -
df = pd.read_csv('10_points.csv')

# Reshape data for modelling
X = df['x'].values.reshape(-1, 1)  # -1 means it calculates the dimension of rows, but has 1 column
y = df['y'].values

# Instantiate the model
our_model = LinearRegression()

# Fit the model
our_model.fit(X, y)

# Export the model
print('...Exporting the model...')
pickle.dump(our_model, open('our_model.pkl', 'wb'))
