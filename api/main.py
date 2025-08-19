from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
with open("ml/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(x: float):
    pred = model.predict(np.array([[x]]))
    return {"prediction": float(pred[0])}
