from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd

# Tworzenie aplikacji FastAPI
app = FastAPI()

# Za?adowanie wytrenowanego modelu
try:
    trained_model = joblib.load("score_model.pkl")
except FileNotFoundError:
    trained_model = None

# Klasa do walidacji danych JSON
class PredictionRequest(BaseModel):
    data: list  # Lista rekordów, gdzie ka?dy jest s?ownikiem

@app.post("/predict-json")
async def predict_from_json(request: PredictionRequest):
    """
    Przewiduje wyniki na podstawie danych JSON.
    """
    if trained_model is None:
        return {"error": "Model nie zosta? za?adowany."}

    input_data = pd.DataFrame(request.data)

    try:

        input_data['score_to_tuition_ratio'] = input_data['score'] / input_data['tuition']

        input_data['gender'] = input_data['gender'].map({'male': 0, 'female': 1}).fillna(0)
        input_data = input_data.rename(columns={'gender': 'gender_1'})

        input_data = pd.get_dummies(input_data, columns=['ethnicity'], drop_first=True)

        expected_features = trained_model.feature_names_in_
        for col in expected_features:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[expected_features]

        predictions = trained_model.predict(input_data)
        return {"predictions": predictions.tolist()}

    except Exception as e:
        return {"error": str(e)}


@app.post("/predict-csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Przewiduje wyniki na podstawie pliku CSV.
    """
    if trained_model is None:
        return {"error": "Model nie zosta? za?adowany."}

    # Odczyt CSV
    try:
        input_data = pd.read_csv(file.file)
        predictions = trained_model.predict(input_data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}
