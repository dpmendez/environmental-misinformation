import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collect_results import load_all_models, save_and_show

METRICS = [
    "balanced_accuracy",
    "precision_false",
    "recall_false",
    "f1_false",
    "auc"
]

DELTA_METRICS = [
    "balanced_accuracy",
    "precision_false",
    "recall_false",
    "f1_false",
]

def plot_single_model_metrics(df_long, model_name, apply_threshold=True):
    subset = df_long[
        (df_long["model_name"] == model_name) &
        (df_long["threshold_applied"] == apply_threshold)]

    if subset.empty:
        print(f"[WARN] No data for {model_name}, threshold={apply_threshold}")
        return
    
    plt.figure(figsize=(6, 4))
    plt.bar(subset["metric"], subset["value"])
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"Metrics – {model_name} (threshold={apply_threshold})")
    plt.tight_layout()

    suffix = "_threshold" if apply_threshold else ""
    save_and_show(f"figures/{model_name}_metrics{suffix}.png")


def plot_models_single_metric(df_long, metric, apply_threshold=True):
    subset = df_long[
        (df_long["metric"] == metric) & 
        (df_long["threshold_applied"] == apply_threshold)]

    if df_plot.empty:
        print("[WARN] No data to plot")
        return
    
    plt.figure(figsize=(8, 4))
    plt.bar(subset["model_name"], subset["value"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"{metric} by Model (threshold={apply_threshold})")
    plt.tight_layout()


def plot_all_models_metrics(df_long, output_path=None, apply_threshold=True):
    title="Model Metrics Comparison"

    df_plot = df_long[df_long["threshold_applied"] == apply_threshold]
    
    metrics = df_plot["metric"].unique()
    models = df_plot["model_name"].unique()

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    plt.figure(figsize=(9, 5))

    for i, model in enumerate(models):
        vals = (
            df_plot[df_plot["model_name"] == model]
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
    

def compute_metric_deltas(df_long):
    """
    Returns a tidy dataframe with:
    model_name | metric | delta
    where delta = value(threshold=True) - value(threshold=False)
    """

    df_use = df_long[df_long["metric"].isin(DELTA_METRICS)]

    # Separate threshold vs no-threshold
    df_t = df_use[df_use["threshold_applied"] == True]
    df_nt = df_use[df_use["threshold_applied"] == False]

    # Pivot so metrics become rows, values aligned
    df_t_p = df_t.pivot_table(
        index=["model_name", "metric"],
        values="value"
    )

    df_nt_p = df_nt.pivot_table(
        index=["model_name", "metric"],
        values="value"
    )

    # Align & subtract
    delta = df_t_p - df_nt_p
    delta = delta.reset_index().rename(columns={"value": "delta"})

    return delta


def plot_metric_deltas(df_deltas, output_path=None):
    plt.figure(figsize=(9, 5))

    metrics = df_deltas["metric"].unique()
    models = df_deltas["model_name"].unique()

    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        vals = (
            df_deltas[df_deltas["model_name"] == model]
            .set_index("metric")
            .loc[metrics]["delta"]
            .values
        )
        plt.bar(x + i * width, vals, width, label=model)

    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x + width * (len(models) - 1) / 2, metrics, rotation=30)
    plt.ylabel("Δ metric (threshold − no threshold)")
    plt.title("Effect of Applying Decision Threshold")
    plt.legend(fontsize=9)
    plt.tight_layout()

    if output_path:
        save_and_show(output_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply-threshold", action="store_true", help="Apply threshold")
    args = parser.parse_args()

    apply_threshold=args.apply_threshold
    suffix = "_threshold" if apply_threshold else ""

    models = load_all_models("../notebooks/results/")

    df = pd.DataFrame(models)[["model_name", "model_type", "threshold_applied"] + METRICS]
    df_metrics = df.copy()
    df_metrics = df_metrics.sort_values(by=["model_type", "model_name", "threshold_applied"])
    df_metrics.to_csv("results/metrics_summary.csv", index=False)

    # melt for plotting
    df_long = df.melt(
        id_vars=["model_name", "model_type", "threshold_applied"], #columns that identify the entity (stay the same)
        value_vars=["precision_false", "recall_false", "f1_false", "balanced_accuracy", "auc"], #columns i want to stack into rows
        var_name="metric", #name of the new column that will hold the old column names
        value_name="value" #name of the new column that holds the values.
    )

    for model in df_long["model_name"].unique():
        plot_single_model_metrics(df_long, model, apply_threshold)

    plot_all_models_metrics(df_long, f"figures/metrics_all_models{suffix}.png", apply_threshold)

    deltas = compute_metric_deltas(df_long)
    df_deltas = deltas.reset_index()
    df_deltas = df_deltas.sort_values(by=["model_name"])
    df_deltas.to_csv("results/metrics_deltas.csv", index=False)
    
    plot_metric_deltas(deltas, output_path="figures/metrics_deltas.png")

if __name__ == "__main__":
    main()