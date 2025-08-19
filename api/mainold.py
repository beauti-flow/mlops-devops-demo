# api/main.py
from fastapi import FastAPI
import joblib
import numpy as np
####
app = FastAPI()
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "ML model API is running!"}

@app.post("/predict")
def predict(features: list[float]):
    prediction = model.predict([np.array(features)])
    return {"prediction": prediction.tolist()}
