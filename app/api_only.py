from fastapi import FastAPI
from pydantic import BaseModel
from inference import InferenceModel
import os

PULL_HF = True  # set True if loading model from Hugging Face
USE_TOKEN = False  # set to True if loading from private repo
FALSE_LABEL_ID = 0  # LIKELY_FALSE

MODEL_1_DIR = "app/models/baseline" if not PULL_HF else "dpmendez/environmental-misinformation-baseline"
MODEL_2_DIR = "app/models/threshold" if not PULL_HF else "dpmendez/environmental-misinformation-threshold"

app = FastAPI() # Initialize the FastAPI app

class Claim(BaseModel):
    text: str

models = {
    "baseline": InferenceModel(MODEL_1_DIR, "baseline", FALSE_LABEL_ID, PULL_HF, USE_TOKEN),
    "thresholded": InferenceModel(MODEL_2_DIR, "thresholded", FALSE_LABEL_ID, PULL_HF, USE_TOKEN),
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