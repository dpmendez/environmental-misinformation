import argparse
import matplotlib.pyplot as plt
from collect_results import load_all_models, save_and_show

BEST_MODEL_NAMES = [
    "log",
    "distilbert",
]

def plot_rocs(models, title, output_path=None):
    plt.figure(figsize=(7,6))

    for m in models:
        plt.plot(
            m["fpr"],
            m["tpr"],
            label=f'{m["name"]} (AUC={m["auc"]:.3f})'
        )
    
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()

    if output_path:
        save_and_show(output_path)
    else:
        plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-threshold", action="store_true", help="Plot only thresholded models")
    args = parser.parse_args()

    apply_threshold=args.apply_threshold
    suffix = "_threshold" if apply_threshold else ""

    models = load_all_models("../notebooks/results/")

    if args.apply_threshold:
        models = [
            m for m in models
            if m["threshold"]==True
        ]
    else:
        models = [
            m for m in models
            if m["threshold"]==False
        ]

    classic = [m for m in models if m["type"] == "classic"]
    transformer = [m for m in models if m["type"] == "transformer"]
    best_models = [m for m in models if m["name"] in BEST_MODEL_NAMES]

    plot_rocs(classic, "ROC – Classic ML Models", "figures/roc_classic_models.png")
    plot_rocs(transformer, "ROC – Transformer Models", "figures/roc_transformer_models.png")
    plot_rocs(models, "ROC – All Models", "figures/roc_all_models.png")
    plot_rocs(best_models, "ROC – Best Classic and Transformer Models", "figures/roc_best_models.png")

if __name__ == "__main__":
    main()