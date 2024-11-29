from http.client import responses

import uvicorn
from fastapi import FastAPI, HTTPException, Form
from models.point import Point
from libs.model import train, predict
from pathlib import Path
from typing import Annotated

# BASE_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath('ml_models')
DATA_DIR = Path(BASE_DIR).joinpath('data')

app = FastAPI()


@app.get("/", tags=["intro"])
async def index():
    return {"message": "Linear Regression ML"}


@app.post("/model/point", tags=["data"], response_model=Point, status_code=200)
async def point(x: Annotated[int, Form()], y: Annotated[int, Form()]):
    return Point(x, y)


@app.post("/model/train", tags=["model"], status_code=200)
async def train_model(data: Point, data_name="10_points", model_name="our_model"):
    data_file = Path(DATA_DIR).joinpath(f"{data_name}.csv")
    model_file = Path(MODEL_DIR).joinpath(f"{model_name}.pkl")
    data = data.model_dump()
    x = data["x"]
    y = data["y"]
    train(x, y, data_file, model_file)

    response_object = {"model_fit": "ok", "model_save": "ok"}
    return response_object


@app.post("/model/predict", tags=["model"], response_model=Point, status_code=200)
async def getPrediction(data: Point, model_name="our_model"):
    model_file = Path(MODEL_DIR).joinpath(f"{model_name}.pkl")
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    data = data.model_dump()
    x = data["x"]
    y_pred = predict(x=x, ml_model=model_file)
    data["y"] = y_pred
    response_object = {"x": x, "y": y_pred[0][0]}
    return response_object

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8008)
