import logging

import numpy as np
import pandas as pd
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
logging.basicConfig(filename='logger.txt', level=logging.INFO)
logger = logging.getLogger("")


def predict_value():
    X_unknown = np.array([2.78])
    return imported_model.coef_[0][0] * X_unknown[0] + imported_model.intercept_[0]


def update_and_fit_model(csv_path,x,y):
    df = pd.read_csv(csv_path)
    # Add values as last row in DF and Overwrite
    df.loc[len(df.index)] = [x, y]  # list
    df.to_csv(csv_path, index=False)
    # Read data again
    df = pd.read_csv(csv_path)
    # Reshape data for modeling
    x = df['x'].values.reshape(-1, 1)
    y = df['y'].values.reshape(-1, 1)
    # Instantiate the model
    our_model = LinearRegression()
    # Fit the model
    our_model.fit(x, y)
    # Export the model
    pickle.dump(our_model, open('our_model.pkl', 'wb'))

if __name__ == '__main__':
    imported_model = pickle.load(open("our_model.pkl", "rb"))
    logger.info("predict value")
    logger.info("y:")
    logger.info(predict_value())
    logger.info("update and fit model:")
    logger.info(update_and_fit_model('10_points.csv',11,21.65))
