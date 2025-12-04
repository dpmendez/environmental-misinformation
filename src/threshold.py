import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

def find_optimal_threshold(model, tokenizer, val_dataset, false_label_id,
                           harmful_cost=1.0, benign_cost=0.3, 
                           batch_size=16, device=None):
    """
    Compute the optimal probability threshold for 'likely_false'.

    Parameters
    ----------
    model : HF model
    tokenizer : tokenizer
    val_dataset : Dataset (tokenized and set_format)
    false_label_id : int
        Numeric ID of the "likely_false" class.
    harmful_cost : float
        Cost of FP (= predict true when false).
    benign_cost : float
        Cost of FN (= predict false when true).
    batch_size : int
    device : torch.device

    Returns
    -------
    best_threshold : float
    fig : Plotly figure
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # ------------------------------------------
    # Get probabilities
    # ------------------------------------------
    all_probs = []
    all_labels = []

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    with torch.no_grad():
        for batch in val_loader:
            # Move only the input tensors to device (labels may be named 'label' or 'labels')
            inputs = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, false_label_id]

            # Accept either 'label' or 'labels' keys in the batch (depending on data collator/trainer behavior)
            if "label" in batch:
                labels_tensor = batch["label"]
            elif "labels" in batch:
                labels_tensor = batch["labels"]
            else:
                raise KeyError("No 'label' or 'labels' key found in batch. Ensure val_dataset set_format includes the label column named 'label'.")

            # Convert true labels to binary: 1 if equals false_label_id (likely_false), else 0
            true_binary = (labels_tensor == false_label_id).to(torch.long)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(true_binary.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ------------------------------------------
    # Evaluate thresholds
    # ------------------------------------------
    thresholds = np.linspace(0, 1, 101)
    harmful_errors = [] # fp
    benign_errors = []  # fn
    total_harm = []

    for t in thresholds:
        # preds = (all_probs > t).astype(int) # likely_false
        preds = np.where(all_probs >= t, 0, 1)

        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
        
        harmful_errors.append(fp)   # predicted TRUE when actually FALSE
        benign_errors.append(fn)    # predicted FALSE when actually TRUE
        
        harm = fn*harmful_cost + fp*benign_cost
        total_harm.append(harm)

    # ------------------------------------------
    # Pick optimal threshold
    # ------------------------------------------
    best_idx = np.argmin(total_harm)
    best_threshold = thresholds[best_idx]

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=harmful_errors, name="Harmful (FN)"))
    fig.add_trace(go.Scatter(x=thresholds, y=benign_errors, name="Benign (FP)"))
    fig.add_trace(go.Scatter(x=thresholds, y=total_harm, name="Weighted Harm", line=dict(width=4)))

    fig.add_vline(x=best_threshold, line_dash="dash", line_color="black",
                  annotation_text=f"Best = {best_threshold:.2f}")

    fig.update_layout(
        title="Threshold Optimization — Harm Curve",
        xaxis_title="Threshold for predicting 'likely_false'",
        yaxis_title="Error Count / Harm",
        template="plotly_white"
    )

    fig.show()

    return best_threshold, fig

def classify_with_threshold(model, tokenizer, text, threshold, false_label_id, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]

    p_false = probs[false_label_id].item()

    if p_false > threshold:
        prediction = "LIKELY_FALSE"
    else:
        prediction = "LIKELY_TRUE"

    return {
        "prediction": prediction,
        "prob_false": p_false,
        "threshold": threshold
    }
