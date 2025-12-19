import os
import json
import numpy as np
import matplotlib.pyplot as plt


def save_and_show(fig_path, dpi=200):
    plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()


def load_all_models(results_root):
    models = []

    for root, dirs, files in os.walk(results_root):
        if "roc.npz" in files and "metrics.json" in files:
            roc = np.load(os.path.join(root, "roc.npz"))
            with open(os.path.join(root, "metrics.json")) as f:
                metrics = json.load(f)

            models.append({
                "name": metrics["model_name"],
                "type": metrics["model_type"],
                "fpr": roc["fpr"],
                "tpr": roc["tpr"] ,
                "auc": roc["auc"].item(),
                **metrics
            })
    
    return models