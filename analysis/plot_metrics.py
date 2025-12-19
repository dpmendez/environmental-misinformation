import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collect_results import load_all_models, save_and_show

def plot_single_model_metrics(df_long, model_name):
    subset = df_long[df_long["model_name"] == model_name]

    plt.figure(figsize=(6, 4))
    plt.bar(subset["metric"], subset["value"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"Metrics – {model_name}")
    plt.tight_layout()


def plot_models_single_metric(df_long, metric):
    subset = df_long[df_long["metric"] == metric]

    plt.figure(figsize=(8, 4))
    plt.bar(subset["model_name"], subset["value"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric.capitalize()} by Model")
    plt.tight_layout()


def plot_all_models_metrics(df_long, output_path=None):
    title="Model Metrics Comparison"
    metrics = df_long["metric"].unique()
    models = df_long["model_name"].unique()

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    plt.figure(figsize=(9, 5))

    for i, model in enumerate(models):
        vals = (
            df_long[df_long["model_name"] == model]
            .set_index("metric")
            .loc[metrics]["value"]
            .values
        )
        plt.bar(x + i * width, vals, width, label=model)
    
    plt.xticks(x + width * (len(models) - 1) / 2, metrics)
    plt.ylabel("Score")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9)
    plt.tight_layout()

    if output_path:
        save_and_show(output_path)
    else:
        plt.show()
    

models = load_all_models("../notebooks/results/")

df = pd.DataFrame(models)[[
    "model_name",
    "model_type",
    "accuracy",
    "balanced_accuracy",
    "precision_false",
    "recall_false",
    "f1_false",
    "f1_weighted",
    "auc"
]]

# melt for plotting
df_long = df.melt(
    id_vars=["model_name", "model_type"], #columns that identify the entity (stay the same)
    value_vars=["precision_false", "recall_false", "f1_false", "f1_weighted", "auc"], #columns i want to stack into rows
    var_name="metric", #name of the new column that will hold the old column names
    value_name="value" #name of the new column that holds the values.
)

df.to_csv("results/metrics_summary.csv", index=False)
df_long.to_csv("results/metrics_long.csv", index=False)

# for metric in ["precision_false", "recall_false", "f1_false", "f1_weighted", "auc"]:
#     plot_models_single_metric(df_long, metric)

for model in df_long["model_name"].unique():
    plot_single_model_metrics(df_long, model)

plot_all_models_metrics(df_long, "figures/metrics_all_models.png")
