from fastapi import FastAPI
from pydantic import BaseModel
from inference import InferenceModel
import os

MODEL_1_DIR = "models/baseline_model"
MODEL_2_DIR = "models/threshold_model"

FALSE_LABEL_ID = 0  # LIKELY_FALSE

class Claim(BaseModel):
    text: str

app = FastAPI() # Initialize the FastAPI app

models = {
    "baseline": InferenceModel(MODEL_1_DIR, "baseline", FALSE_LABEL_ID),
    "thresholded": InferenceModel(MODEL_2_DIR, "thresholded", FALSE_LABEL_ID),
}

@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Prediction endpoint
@app.post("/predict_all")
def predict(claim: Claim):
    """
    claim: Claim  means the incoming JSON will be parsed into Claim(text="...")
    """
    results = {}

    for name, model in models.items():
        results[name] = model.predict(claim.text)

    return results