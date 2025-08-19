import os
import pickle
import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()

MODEL_PATH = "ml/model.pkl"

def load_model():
    """Try to load the model, retrain if missing or invalid."""
    if not os.path.exists(MODEL_PATH):
        print("⚠️ Model file not found. Training a new model...")
        retrain_model()

    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load model ({e}). Retraining...")
        retrain_model()
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

def retrain_model():
    """Run the training script to generate a new model.pkl."""
    result = subprocess.run(
        ["python", "ml/train.py"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("❌ Training failed:")
        print(result.stderr)
        raise RuntimeError("Training failed. Check train.py.")
    print("✅ Model retrained successfully.")

# Load model at startup
model = load_model()

@app.get("/")
def root():
    return {"message": "API is running and model is loaded!"}

@app.post("/predict")
def predict(x: float):
    try:
        # example: simple model with .predict method
        prediction = model.predict([[x]])
        return {"input": x, "prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
