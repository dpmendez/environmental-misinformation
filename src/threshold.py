import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

cost_FP = 5.0     # very harmful to miss false claim
cost_FN = 1.0     # moderately harmful to block true claim
benefit_TN = 3.0  # good to catch harmful claim
benefit_TP = 1.0  # good to allow true claim


def get_harm(fn, fp, tp, tn):
    return (
            fn * cost_FN +
            fp * cost_FP -
            tp * benefit_TP -
            tn * benefit_TN
        )
    
def find_optimal_threshold_from_scores(
    y_true,
    scores,
    false_label_id=0,
    return_curve=True
):
    """
    y_true: array-like of shape (n_samples,), values {0,1}
    scores: array-like of shape (n_samples,)
            Higher score = more likely class `false_label_id`
    """

    thresholds=np.linspace(0, 1, 101)

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    harmful_errors = []
    benign_errors = []
    total_harm = []

    for t in thresholds:
        preds = np.where(scores >= t, false_label_id, 1 - false_label_id)

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        harmful_errors.append(fp)
        benign_errors.append(fn)
        harm = get_harm(fn, fp, tp, tn)
        total_harm.append(harm)

    best_idx = np.argmin(total_harm)
    best_threshold = thresholds[best_idx]

    if return_curve:
        return {
            "best_threshold": best_threshold,
            "thresholds": thresholds,
            "harm": total_harm,
            "fp": harmful_errors,
            "fn": benign_errors,
        }

    return best_threshold


def find_optimal_threshold(model, tokenizer,
                           val_dataset, false_label_id=0,
                           batch_size=20, device=None):
    """
    Find the optimal probability threshold for predicting class `false_label_id`
    (likely_false = 0).

    - The model outputs probabilities for each class.
    - We use P(likely_false) to threshold.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # ------------------------------------------
    # Collect probabilities and labels
    # ------------------------------------------
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ["input_ids", "attention_mask"]
            }

            outputs = model(**inputs)
            logits = outputs.logits

            # Probability of class 0 (likely_false)
            p_false = torch.softmax(logits, dim=1)[:, false_label_id]

            # Labels should be 0 or 1 directly
            if "label" in batch:
                labels = batch["label"]
            elif "labels" in batch:
                labels = batch["labels"]
            else:
                raise KeyError("No label column found in the batch.")

            all_probs.extend(p_false.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ------------------------------------------
    # Threshold evaluation
    # ------------------------------------------
    thresholds = np.linspace(0, 1, 101)

    harmful_errors = []  # FP
    benign_errors = []   # FN
    total_harm = []

    for t in thresholds:

        # Predict likely_false (0) when P(false) >= t
        preds = np.where(all_probs >= t, 0, 1)

        tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

        harmful_errors.append(fp)
        benign_errors.append(fn)
        harm = get_harm(fn, fp, tp, tn)        
        total_harm.append(harm)

    # ------------------------------------------
    # Find best threshold
    # ------------------------------------------
    best_idx = np.argmin(total_harm)
    best_threshold = thresholds[best_idx]

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=harmful_errors, name="Harmful (FN)"))
    fig.add_trace(go.Scatter(x=thresholds, y=benign_errors, name="Benign (FP)"))
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=total_harm,
        name="Weighted Harm",
        line=dict(width=4)
    ))

    fig.add_vline(
        x=best_threshold,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Best = {best_threshold:.2f}"
    )

    fig.update_layout(
        title="Threshold Optimization — Harm Curve",
        xaxis_title="Threshold for predicting 'likely_false' (class 0)",
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
