import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class InferenceModel:
    def __init__(self, model_dir, false_label_id):
        """
        model_dir should contain:
            - config.json
            - pytorch_model.bin
            - tokenizer files
            - threshold.json
            - label_map.json
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Load threshold
        with open(os.path.join(model_dir, "threshold.json")) as f:
            self.threshold = json.load(f)["best_threshold"]

        with open(os.path.join(model_dir, "label_map.json")) as f:
            maps = json.load(f)
        self.label2id = maps["label2id"]
        self.id2label = maps["id2label"]
        self.false_id = self.label2id["LIKELY_FALSE"]

    def predict(self, claim, max_length=256):
        """
        Predict for a single claim (string).
        """
        inputs = self.tokenizer(claim, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        p_false = probs[self.false_label_id]

        # Apply threshold
        if p_false >= self.threshold:
            pred_id = self.false_label_id
        else:
            pred_id = 1 - self.false_label_id  # only valid for binary

        return {
            "pred_id": int(pred_id),
            "pred_label": self.id2label[pred_id],
            "probabilities": {self.id2label[i]: float(probs[i]) for i in range(len(probs))},
            "threshold": self.threshold,
            "text": claim
        }