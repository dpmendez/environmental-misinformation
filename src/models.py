from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# from xgboost import XGBClassifier

from transformers import Trainer
import torch
import torch.nn as nn

def train_classic_model(x_train, y_train,
                        model_type="logreg",
                        ngram_range=(1,2),
                        max_features=10000,
                        class_weight="balanced"):
    
    preprocessor = ColumnTransformer([
        ("transform_text", TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words="english"), "text")
    ])
    
    if model_type == "logreg":
        classifier = LogisticRegression(max_iter=1000, class_weight=class_weight)
    elif model_type == "rf":
        classifier = RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=42)
    elif model_type == "svc":
        classifier = LinearSVC(class_weight=class_weight, random_state=42)
#    elif model_type == "xgb":
#        classifier = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
#                                   subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
    else:
        raise ValueError("Choose from: 'logreg', 'rf', 'svc', 'xgb'")
    
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    clf.fit(x_train, y_train)
    return clf

# Weighted Trainer to handle label imbalance
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store class weights as tensor but defer device placement until compute_loss
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Accept either 'label' or 'labels' keys coming from the dataset / data collator.
        # Map 'label' -> 'labels' so the model receives the expected argument name.
        if "label" in inputs and "labels" not in inputs:
            inputs["labels"] = inputs.pop("label")
        elif "labels" in inputs:
            # already present, nothing to do
            pass
        else:
            raise KeyError("Expected 'label' or 'labels' key in inputs but not found. Ensure dataset set_format includes the label column.")

        # Move tensors to the model device to avoid device-mismatch errors
        device = model.device
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        labels = inputs.get("labels")

        # Forward pass
        outputs = model(**{k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask", "token_type_ids", "labels")})
        logits = outputs.get("logits")

        # Place class weights on correct device if provided
        if self.class_weights is not None:
            weight = self.class_weights.to(device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        if labels is None:
            # If no labels provided, return zero loss and outputs (this scenario should not happen during training)
            loss = torch.tensor(0.0, device=device)
        else:
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss