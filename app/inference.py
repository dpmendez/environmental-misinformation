import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

# Apply global settings before loading any models/tensors
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float32)

class InferenceModel:
    def __init__(self, model_dir, model_name, false_label_id, pull_hf=True, use_token=False):
        """
        model_dir should contain:
            - config.json
            - pytorch_model.bin
            - tokenizer files
            - threshold.json / threshold_app.json for app only (mean)
            - label_map.json
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        if pull_hf:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=use_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, use_auth_token=use_token)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)

        self.model.to(self.device)
        self.model.eval()

        # Optional threshold
        # threshold_path = os.path.join(model_dir, "threshold.json")
        threshold_path = os.path.join(model_dir, "threshold_app.json")
        if pull_hf:
            try:
                # threshold_path = hf_hub_download(repo_id=model_dir, filename="threshold.json", use_auth_token=use_token)
                threshold_path = hf_hub_download(repo_id=model_dir, filename="threshold_app.json", use_auth_token=use_token)
            except:
                threshold_path = None

        if threshold_path and os.path.exists(threshold_path):
            with open(threshold_path) as f:
                self.threshold = json.load(f).get("best_threshold")
        else:
            self.threshold = None # model may not use threshold

        print(f"[Loaded Model] {model_name} | threshold={self.threshold}")
        
        # Load label mappings
        label_map_path = os.path.join(model_dir, "label_map.json")
        if pull_hf:
            label_map_path = hf_hub_download(repo_id=model_dir, filename="label_map.json", use_auth_token=use_token)

        with open(label_map_path) as f:
            maps = json.load(f)
        self.label2id = {k: int(v) for k, v in maps["label2id"].items()}
        self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        self.false_label_id = int(false_label_id)

    def predict(self, claim, max_length=256):
        """
        Predict for a single claim (string).
        """
        inputs = self.tokenizer(claim, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Default argmax prediction
        pred_id = int(probs.argmax())
        
        # Apply threshold only if present (binary case)
        if self.threshold is not None and len(probs) == 2:
            if probs[self.false_label_id] >= self.threshold:
                pred_id = self.false_label_id
            else:
                pred_id = 1 - self.false_label_id
                
        return {
            "text": claim,
            "model": self.model_name,
            "pred_id": pred_id,
            "pred_label": self.id2label[pred_id],
            "confidence": round(float(probs[pred_id])*100,2),
            "probabilities": {
                self.id2label[i]: float(probs[i])
                for i in range(len(probs))
            },
            "threshold": self.threshold
        }