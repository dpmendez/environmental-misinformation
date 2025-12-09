import pandas as pd
import numpy as np 
import kagglehub
import os
import re
import string

from datasets import load_dataset
from evaluate import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, balanced_accuracy_score, classification_report

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

def map_sf_label(label):
    """
    Map Science Feedback labels to unified scheme: SUPPORTS, REFUTES, NEUTRAL, DISPUTED.
    """
    if not isinstance(label, str):
        return 'DISPUTED'  # fallback for missing or unexpected values

    label_lower = label.lower()

    # High confidence / correct -> SUPPORTS
    if 'high' in label_lower or 'accurate' in label_lower or 'correct' in label_lower:
        return 'SUPPORTS'

    # low confidence / incorrect -> REFUTES
    elif 'low' in label_lower or 'inaccurate' in label_lower or 'incorrect' in label_lower \
         or 'misleading' in label_lower or 'unsupported' in label_lower or 'flawed' in label_lower:
        return 'REFUTES'

    # neutral / context unclear -> NEUTRAL
    elif 'neutral' in label_lower or 'lacks context' in label_lower or 'imprecise' in label_lower:
        return 'NEUTRAL'

    # anything else -> DISPUTED
    else:
        return 'DISPUTED'

def clean_text(text):

    text = text.lower() # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) # no urls
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces

    return text

# Split into training, validation and test sets
def split_data(texts, labels):
	# 70, 15, 15 split
	train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
	val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

	return {'x_train': train_texts, 'y_train': train_labels,
			'x_test': test_texts, 'y_test': test_labels,
			'x_val': val_texts, 'y_val': val_labels}


# Split into training, validation and test sets
def split_data_extras(texts, labels, extras):
	# 70, 15, 15 split
	train_texts, temp_texts, train_labels, temp_labels, train_extras, temp_extras = train_test_split(texts, labels, extras, test_size=0.3, random_state=42)
	val_texts, test_texts, val_labels, test_labels, val_extras, test_extras = train_test_split(temp_texts, temp_labels, temp_extras, test_size=0.5, random_state=42)

	return {'x_train': train_texts, 'y_train': train_labels, 'z_train': train_extras,
			'x_test': test_texts, 'y_test': test_labels, 'z_test': test_extras,
			'x_val': val_texts, 'y_val': val_labels, 'z_val': val_extras}


# Get model features from LR, RF, XGB
def get_feature_importance(this_model, top_n=10):
    clf = this_model.named_steps["classifier"]
    feature_names = this_model.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [f.replace("text__", "") for f in feature_names]

    # Logistic Regression → coefficients
    if hasattr(clf, "coef_"):
        feature_weights_list = []
        for i, label in enumerate(clf.classes_):
            df_i = pd.DataFrame({
                "Feature": feature_names,
                "Weight": clf.coef_[i],
                "Label": label
            })
            feature_weights_list.append(df_i)
        feature_weights = pd.concat(feature_weights_list)

        # Top positive + negative
        top_features = []
        for label in clf.classes_:
            df_label = feature_weights[feature_weights["Label"] == label]
            top_pos = df_label.sort_values("Weight", ascending=False).head(top_n)
            top_neg = df_label.sort_values("Weight").head(top_n)
            top_features.append(pd.concat([top_pos, top_neg]))
        return pd.concat(top_features)

    # RF / XGB → feature importances
    elif hasattr(clf, "feature_importances_"):
        feature_weights = pd.DataFrame({
            "Feature": feature_names,
            "Weight": clf.feature_importances_
        }).sort_values("Weight", ascending=False)
        return feature_weights.head(top_n)

    else:
        raise ValueError("Model does not support feature importance")

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.

    Accepts either an (logits, labels) tuple, an EvalPrediction-like object
    with .predictions and .label_ids, or (preds, labels) where preds may be
    either logits/probabilities (2D) or already-decoded label ids (1D).
    Returns a dict of scalar metrics and a classification_report dict under
    the key 'classification_report'.
    """
    # Unpack EvalPrediction or tuple
    if hasattr(eval_pred, "predictions") and hasattr(eval_pred, "label_ids"):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        preds, labels = eval_pred

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    # If preds are logits/probabilities (2D), convert to label ids
    if preds.ndim == 2:
        pred_ids = np.argmax(preds, axis=1)
    elif preds.ndim == 1:
        # already predicted label ids
        pred_ids = preds.astype(int)
    else:
        raise ValueError(f"Unsupported prediction shape: {preds.shape}")

    false_id = 0

    metrics = {
        "accuracy": accuracy_score(labels, pred_ids),
        "balanced_accuracy": balanced_accuracy_score(labels, pred_ids),
        "f1_weighted": f1_score(labels, pred_ids, average="weighted"),
        "f1_false": f1_score(labels, pred_ids, pos_label=false_id),
        "recall_false": recall_score(labels, pred_ids, pos_label=false_id),
        "precision_false": precision_score(labels, pred_ids, pos_label=false_id)
    }

    # # Add sklearn classification report (as a dict) for per-class breakdown
    # try:
    #     cls_report = classification_report(labels, pred_ids, output_dict=True, zero_division=0)
    #     metrics["classification_report"] = cls_report
    # except Exception:
    #     # if something goes wrong building the report, skip it but keep scalar metrics
    #     pass

    return metrics
    
# Create a PyTorch Dataset
class NewsDataset(Dataset):

	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		item = {key: val[idx] for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item
