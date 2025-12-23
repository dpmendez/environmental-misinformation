import numpy as np

def get_false_class_scores(model, X, model_type):
    """
    Returns a 1D array of continuous scores for the 'false' class.
    Higher = more likely false.
    """

    if model_type in ["log", "rf", "xgb", "log"]:
        if not hasattr(model, "predict_proba"):
            raise ValueError("Model does not support predict_proba")
        probs = model.predict_proba(X)
        return probs[:, 0]  # P(likely_fase) assuming 'false' is class 0

    elif model_type == "svc":
        if not hasattr(model, "decision_function"):
            raise ValueError("SVC does not support decision_function")
        scores = model.decision_function(X)
        return scores  # not probabilities, but valid scores

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
