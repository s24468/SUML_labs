from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile




class Point(BaseModel):
    x: int
    y: Optional[float]
    # data: list  # Lista rekordów, gdzie ka?dy jest s?ownikiem





# async def predict_from_json(request: Point):
#     input_data = pd.DataFrame(request.data)
#
#     input_data['score_to_tuition_ratio'] = input_data['score'] / input_data['tuition']
#
#     input_data['gender'] = input_data['gender'].map({'male': 0, 'female': 1}).fillna(0)
#     input_data = input_data.rename(columns={'gender': 'gender_1'})
#
#     input_data = pd.get_dummies(input_data, columns=['ethnicity'], drop_first=True)
#
#     expected_features = model.feature_names_in_
#     for col in expected_features:
#         if col not in input_data.columns:
#             input_data[col] = 0
#
#     input_data = input_data[expected_features]
#
#     predictions = model.predict(input_data)
#     return {"predictions": predictions.tolist()}