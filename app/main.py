from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

### Memory logging setup
import psutil, os, threading, time

def log_memory():
    process = psutil.Process(os.getpid())
    while True:
        mem_mb = process.memory_info().rss / (1024 ** 2) # Convert to MB
        print(f"[Memory usage] {mem_mb:.2f} MB")
        time.sleep(5)

threading.Thread(target=log_memory, daemon=True).start()

app = FastAPI() # Initialize the FastAPI app

# Serve static assets (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

# Request schema
class Claim(BaseModel):
    text: str
    model: str = "thresholded" # default model

# Home page
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from api_only import predict as backend_predict_all
# Prediction API
@app.post("/predict_all")
def predict_all(payload: Claim):

    # Send a Claim object to match api_only FastAPI behavior
    backend_output = backend_predict_all(Claim(text=payload.text))
    model_name = payload.model  # received from dropdown

    if model_name not in backend_output:
        return {"error": f"Model '{model_name}' not found. Available: {list(backend_output.keys())}"}
    
    selected = backend_output[model_name]
    
    formatted = {
        "results": [
            {
                "model": selected["model"],
                "pred_label": selected["pred_label"],
                "confidence": selected["confidence"],
                "threshold": selected["threshold"],
                "probabilities": selected["probabilities"],
            }
        ]
    }

    return formatted

# Run app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)