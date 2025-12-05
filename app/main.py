from fastapi import FastAPI
from pydantic import BaseModel
from inference import InferenceModel
import os

MODEL_PATH = "models/threshold_model"
FALSE_LABEL_ID = 0  # likely_false

class Claim(BaseModel):
    text: str

app = FastAPI() # Initialize the FastAPI app
model = InferenceModel(MODEL_PATH, FALSE_LABEL_ID) # Load trained model

@app.get("/")
def read_root():
    return {"message": "API is running!"}

# Prediction endpoint
@app.post("/predict")
def predict(claim: Claim):
    """
    claim: Claim  means the incoming JSON will be parsed into Claim(text="...")
    """
    result = model.predict(claim.text)
    return result
