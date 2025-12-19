"""
Evaluation script for saved models.

Supports:
 - Transformer (Hugging Face) models saved with `trainer.save_model(...)` or `model.save_pretrained(...)` + `tokenizer.save_pretrained(...)`.
 - Classic sklearn Pipeline objects saved with `joblib.dump(...)`.

Outputs:
 - Prints metrics from `src.utils.compute_metrics`
 - Optionally computes/saves ROC curve and AUC

Usage examples:
 python app/eval.py --model-path ./results/best_model --model-type transformer --test-csv ./data/test_data.csv --label-map ./results/best_model/label_map.json --threshold ./results/best_model/threshold.json --roc-out ./results/best_model/roc.png

 python app/eval.py --model-path ./results/baseline_model/baseline_pipeline.joblib --model-type sklearn --test-csv ./data/test_data.csv --label-col label --roc-out ./results/baseline_model/roc.png

"""
import os
import json
import argparse
from typing import Optional

import numpy as np
import pandas as pd

from src.utils import compute_metrics

# Transformer imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

# Sklearn imports
import joblib
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from src.viz import plotly_confusion_matrix
import glob


def load_label_map(path: Optional[str]):
    if path is None:
        return None
    with open(path, "r") as f:
        data = json.load(f)
    # accept either {'label2id':..., 'id2label':...} or a flat mapping
    if isinstance(data, dict) and "label2id" in data:
        return data["label2id"], data.get("id2label")
    return data, None


def eval_transformer(model_path, test_texts, y_true, label2id=None, threshold_path=None, device="cpu", false_label_id: Optional[int]=None):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Tokenize in batches to avoid OOM
    batch_size = 64
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_texts = list(test_texts[i:i+batch_size])
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()
            logits_list.append(logits)

    logits = np.vstack(logits_list)
    probs = softmax(logits, axis=1)

    # If threshold provided, use it for a binary decision on the specified "false" class
    if threshold_path is not None:
        try:
            with open(threshold_path, "r") as f:
                tdata = json.load(f)
            best_threshold = tdata.get("best_threshold")
        except Exception:
            best_threshold = None
    else:
        best_threshold = None

    # Decide predictions
    if best_threshold is not None:
        # determine false_label_id precedence: explicit arg > label2id mapping > default 0
        if false_label_id is None:
            if isinstance(label2id, dict):
                if "LIKELY_FALSE" in label2id:
                    false_label_id = label2id["LIKELY_FALSE"]
                else:
                    false_label_id = 0
            else:
                false_label_id = 0

        # probability for the specified false class
        p_false = probs[:, false_label_id]
        # if p_false >= threshold, predict false_label_id, otherwise fall back to argmax over classes
        fallback = np.argmax(probs, axis=1)
        y_pred = np.where(p_false >= best_threshold, false_label_id, fallback)
    else:
        y_pred = np.argmax(probs, axis=1)

    return logits, probs, y_pred


def eval_sklearn(model_path, test_texts, y_true, text_col: str = "text"):
    """Load a sklearn pipeline and predict.

    If test_texts is a 1D list/array of strings, convert to a DataFrame with column `text_col`
    because many pipelines use ColumnTransformer expecting named columns.
    """
    clf = joblib.load(model_path)

    # Prepare input X for the sklearn pipeline
    if isinstance(test_texts, (list, tuple, np.ndarray)):
        # convert to DataFrame with a single column matching training
        X = pd.DataFrame({text_col: list(test_texts)})
    elif isinstance(test_texts, pd.DataFrame):
        X = test_texts
    else:
        # try to coerce
        X = pd.DataFrame({text_col: list(test_texts)})

    try:
        probs = clf.predict_proba(X)
        # For binary, take probs[:,1] as positive class; but need to know ordering of classes
        y_pred = clf.predict(X)
        return clf, probs, y_pred
    except Exception:
        # No probability support; fallback to decision_function if available
        try:
            scores = clf.decision_function(X)
            # decision_function can return shape (n_samples,) for binary or (n_samples, n_classes)
            if scores.ndim == 1:
                # binary scoring
                probs = None
                y_pred = (scores > 0).astype(int)
            else:
                # multiclass scores -> argmax
                probs = None
                y_pred = np.argmax(scores, axis=1)
            return clf, probs, y_pred
        except Exception:
            # last resort: only predict
            y_pred = clf.predict(X)
            return clf, None, y_pred


def compute_and_maybe_plot_roc(y_true, probs, output_path: Optional[str], labels=None):
    # y_true: array-like of shape (n_samples,) with integer labels
    # probs: None or array of shape (n_samples, n_classes)
    if probs is None:
        print("No probability scores available; cannot compute ROC/AUC.")
        return None

    n_classes = probs.shape[1]
    if n_classes == 2:
        # binary
        # choose class 1 as positive by default; allow caller to override via `labels` or
        # by passing a specific positive_label through the labels parameter if it's an int.
        # If labels is an int, treat it as the positive class id to compute ROC for.
        positive_label = None
        if isinstance(labels, int):
            positive_label = labels

        if positive_label is None:
            pos_idx = 1
        else:
            pos_idx = positive_label

        y_score = probs[:, pos_idx]
        auc_score = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            plt.figure()
            plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_score:.3f})")
            plt.plot([0,1],[0,1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig(output_path)
            plt.close()

            roc_data_path = output_path.replace(".png", ".npz")
            np.savez(roc_data_path, fpr=fpr, tpr=tpr, auc=auc_score)
            print(f"Saved ROC image to {output_path}")
            print(f"Saved ROC data to {roc_data_path}")
        return auc_score
    else:
        # multiclass: use one-vs-rest macro-average
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        try:
            auc_score = roc_auc_score(y_true_bin, probs, average='macro', multi_class='ovr')
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # plot macro-average by aggregating per-class curves (optional simplified plot)
                plt.figure()
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
                    plt.plot(fpr, tpr, label=f"class {i}")
                plt.plot([0,1],[0,1], linestyle='--', color='gray')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'Multiclass ROC Curves (macro AUC={auc_score:.3f})')
                plt.legend(loc='lower right')
                plt.savefig(output_path)
                plt.close()
            
                roc_data_path = output_path.replace(".png", ".npz")
                np.savez(roc_data_path, fpr=fpr, tpr=tpr, auc=auc_score)
                print(f"Saved ROC image to {output_path}")
                print(f"Saved ROC data to {roc_data_path}")

            return auc_score
        except Exception as e:
            print("Failed to compute multiclass ROC/AUC:", e)
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Path to saved model (transformer directory or sklearn joblib file)")
    parser.add_argument("--model-type", choices=["transformer", "classic"], required=True)
    parser.add_argument("--model-id", default=None, help="Optional model id in lower case")
    parser.add_argument("--test-csv", required=True, help="CSV file with test data")
    parser.add_argument("--text-col", default="text", help="Column name for text in CSV")
    parser.add_argument("--label-col", default="label", help="Column name for label in CSV (integers expected or will be mapped)")
    parser.add_argument("--label-map", default=None, help="Optional JSON file with label2id mapping (saved from training)")
    parser.add_argument("--threshold", default=None, help="Optional JSON file with best_threshold from thresholding step")
    parser.add_argument("--false-label-id", type=int, default=None, help="Optional integer id of the 'false' class used when applying thresholding")
    parser.add_argument("--roc-out", default=None, help="Optional path to save ROC plot PNG")
    parser.add_argument("--device", default="cpu", help="torch device to use (cpu or cuda)")
    args = parser.parse_args()
    # Normalize model directory: if model-path is a directory use it, otherwise use its parent dir
    model_path_abs = os.path.abspath(args.model_path)
    if os.path.isdir(model_path_abs):
        model_dir = model_path_abs
    else:
        model_dir = os.path.dirname(model_path_abs) or os.getcwd()

    # Decide which directory to use for default files. For transformer models we
    # should look inside the exact `--model-path` directory the user passed
    # (e.g. `.../electra-base-discriminator`) rather than the parent directory.
    if args.model_type == "transformer":
        model_dir_for_defaults = model_path_abs
    else:
        model_dir_for_defaults = model_dir

    # If user didn't provide explicit roc output, metrics, threshold, or label_map paths,
    # default them to files inside the chosen model directory
    if args.roc_out is None:
        args.roc_out = os.path.join(model_dir_for_defaults, "roc.png")
    # metrics will be saved in model_dir_for_defaults/metrics.json
    metrics_out = os.path.join(model_dir_for_defaults, "metrics.json")

    # For transformer models, default threshold and label_map to files in model_dir_for_defaults
    if args.model_type == "transformer":
        if args.threshold is None:
            args.threshold = os.path.join(model_dir_for_defaults, "threshold.json")
        if args.label_map is None:
            args.label_map = os.path.join(model_dir_for_defaults, "label_map.json")

    # Use default evaluation batch size inside eval_transformer (64)

    # Initialize label maps (may be populated below if --label-map provided)
    label2id = None
    id2label = None

    df = pd.read_csv(args.test_csv).dropna(subset=[args.text_col, args.label_col])
    texts = df[args.text_col].astype(str).tolist()
    labels_raw = df[args.label_col].tolist()

    # try to coerce labels to ints if possible
    try:
        y_true = np.array(labels_raw, dtype=int)
    except Exception:
        # if labels are strings and label_map provided, map them
        if args.label_map:
            if not os.path.exists(args.label_map):
                raise FileNotFoundError(
                    f"Label map not found at {args.label_map}.\n"
                    "If you trained a transformer, pass --model-path pointing to the model directory (which should contain label_map.json),\n"
                    "or provide --label-map with the full path to the file."
                )
            label2id, id2label = load_label_map(args.label_map)
            # normalize id2label keys to ints (JSON stores keys as strings)
            if id2label is not None:
                try:
                    id2label = {int(k): v for k, v in id2label.items()}
                except Exception:
                    # leave as-is if conversion fails
                    pass
            y_true = np.array([label2id.get(l, -1) for l in labels_raw])
            if (y_true == -1).any():
                raise ValueError("Some labels in CSV not found in provided label_map")
        else:
            raise ValueError("Labels are non-integer and no --label-map was provided to map them to integer ids")

    if args.model_type == "transformer":
        label2id, id2label = (None, None)
        if args.label_map:
            if not os.path.exists(args.label_map):
                raise FileNotFoundError(
                    f"Label map not found at {args.label_map}.\n"
                    "Provide --label-map pointing to the saved label_map.json or ensure --model-path (for transformer) points to the model directory."
                )
            label2id, id2label = load_label_map(args.label_map)
            if id2label is not None:
                try:
                    id2label = {int(k): v for k, v in id2label.items()}
                except Exception:
                    pass
        logits, probs, y_pred = eval_transformer(
            args.model_path,
            texts,
            y_true,
            label2id=label2id,
            threshold_path=args.threshold,
            device=args.device,
            false_label_id=args.false_label_id,
        )
        # compute_metrics accepts either logits or 1D preds; here we prefer the thresholded predictions
        # (y_pred) so metrics reflect any manual thresholding applied.
        metrics = compute_metrics((y_pred, y_true))
        metrics_addon = {
            **metrics,
            "model_type" : args.model_type,
            "model_name" : args.model_id
        }
        print(json.dumps(metrics_addon, indent=2))
        # Optionally compute ROC
        auc_score = compute_and_maybe_plot_roc(y_true, probs, args.roc_out)
        if auc_score is not None:
            print(f"ROC AUC: {auc_score:.4f}")

        # Save metrics JSON to model directory
        try:
            def _np_converter(o):
                if isinstance(o, (np.integer, np.floating)):
                    return o.item()
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return str(o)

            with open(metrics_out, "w") as mf:
                json.dump(metrics_addon, mf, indent=2, default=_np_converter)
            print(f"Saved metrics to {metrics_out}")
        except Exception as e:
            print("Failed to save metrics:", e)

        # Create and save a Plotly confusion matrix (HTML + attempt PNG)
        try:
            # labels for plotting: prefer id2label mapping if available
            if id2label:
                labels_plot = [id2label[i] for i in sorted(id2label.keys(), key=int)]
                y_true_plot = [id2label[int(i)] for i in y_true]
                y_pred_plot = [id2label[int(i)] for i in y_pred]
            else:
                labels_plot = None
                y_true_plot = y_true
                y_pred_plot = y_pred

            cm_fig, cm = plotly_confusion_matrix(y_true_plot, y_pred_plot, labels=labels_plot, title="Confusion Matrix")
            cm_html = os.path.join(model_dir, "confusion_matrix.html")
            cm_png = os.path.join(model_dir, "confusion_matrix.png")
            cm_fig.write_html(cm_html)
            np.save(os.path.join(model_dir, "confusion_matrix.npy"), cm)
            print(f"Saved confusion matrix HTML to {cm_html}")
            try:
                # attempt to save PNG (requires kaleido or orca)
                cm_fig.write_image(cm_png)
                print(f"Saved confusion matrix PNG to {cm_png}")
            except Exception as e:
                print("Could not save PNG (kaleido/orca may be missing):", e)
        except Exception as e:
            print("Failed to create/save confusion matrix:", e)

    else:  # sklearn
        # If the user passed a directory for the sklearn model, attempt to find a
        # serialized pipeline inside it. This makes `--model-path` behave like the
        # transformer case where a directory is provided. We search for common
        # extensions and pick the first candidate deterministically.
        # Always perform auto-discovery for sklearn model files. If the user
        # passed a file path, search its parent directory; if they passed a
        # directory, search that directory directly.
        provided = args.model_path
        if os.path.isdir(provided):
            search_dir = provided
        else:
            search_dir = os.path.dirname(os.path.abspath(provided)) or os.getcwd()

        pattern_list = ["*.joblib", "*.pkl", "*.sav", "*.model"]
        candidates = []
        for pat in pattern_list:
            found = glob.glob(os.path.join(search_dir, pat))
            if found:
                candidates.extend(sorted(found))

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No sklearn model file (*.joblib, *.pkl, *.sav, *.model) found in {search_dir}. Please ensure the model file exists."
            )
        if len(candidates) > 1:
            print(f"Multiple sklearn model files found in {search_dir}; using the first: {candidates[0]}")

        model_path_for_sklearn = candidates[0]
        clf_obj, probs, y_pred = eval_sklearn(model_path_for_sklearn, texts, y_true, text_col=args.text_col)
        metrics = compute_metrics((y_pred, y_true))
        metrics_addon = {
            **metrics,
            "model_type" : args.model_type,
            "model_name" : args.model_id
        }
        print(json.dumps(metrics_addon, indent=2))
        auc_score = compute_and_maybe_plot_roc(y_true, probs, args.roc_out)
        if auc_score is not None:
            print(f"ROC AUC: {auc_score:.4f}")

        # Save metrics JSON to model directory
        try:
            def _np_converter(o):
                if isinstance(o, (np.integer, np.floating)):
                    return o.item()
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return str(o)

            with open(metrics_out, "w") as mf:
                json.dump(metrics_addon, mf, indent=2, default=_np_converter)
            print(f"Saved metrics to {metrics_out}")
        except Exception as e:
            print("Failed to save metrics:", e)

        # Create and save a Plotly confusion matrix for the sklearn model too
        try:
            labels_plot = None
            y_true_plot = y_true
            y_pred_plot = y_pred

            # If user provided a label_map JSON, use it
            if args.label_map:
                if os.path.exists(args.label_map):
                    l2id, id2label_local = load_label_map(args.label_map)
                    if id2label_local is not None:
                        try:
                            id2label_local = {int(k): v for k, v in id2label_local.items()}
                        except Exception:
                            pass
                        labels_plot = [id2label_local[i] for i in sorted(id2label_local.keys(), key=int)]
                        y_true_plot = [id2label_local[int(i)] for i in y_true]
                        y_pred_plot = [id2label_local[int(i)] for i in y_pred]
            # Otherwise, try to use classifier.classes_ if present
            if labels_plot is None and clf_obj is not None and hasattr(clf_obj, "classes_"):
                classes = list(clf_obj.classes_)
                # If predictions are integer indices, map them into class labels
                try:
                    yp = np.array(y_pred)
                    yt = np.array(y_true)
                    if np.issubdtype(yp.dtype, np.integer) and yp.max() < len(classes):
                        y_pred_plot = [classes[int(i)] for i in yp]
                    if np.issubdtype(yt.dtype, np.integer) and yt.max() < len(classes):
                        y_true_plot = [classes[int(i)] for i in yt]
                    labels_plot = classes
                except Exception:
                    # fallback: leave as-is
                    pass

            cm_fig, cm = plotly_confusion_matrix(y_true_plot, y_pred_plot, labels=labels_plot, title="Confusion Matrix")
            cm_html = os.path.join(model_dir, "confusion_matrix_sklearn.html")
            cm_png = os.path.join(model_dir, "confusion_matrix_sklearn.png")
            cm_fig.write_html(cm_html)
            np.save(os.path.join(model_dir, "confusion_matrix.npy"), cm)
            print(f"Saved sklearn confusion matrix HTML to {cm_html}")
            try:
                cm_fig.write_image(cm_png)
                print(f"Saved sklearn confusion matrix PNG to {cm_png}")
            except Exception as e:
                print("Could not save PNG for sklearn confusion matrix (kaleido/orca may be missing):", e)
        except Exception as e:
            print("Failed to create/save sklearn confusion matrix:", e)


if __name__ == "__main__":
    main()