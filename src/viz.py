import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

color_discrete_map = {
    "Fake": "#d95f5f",  # muted red
    "Real": "#5f9fd9"   # muted blue
}


def plotly_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix"):
    # Infer labels if not provided
    if labels is None:
        labels = np.unique(y_true)
        
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Normalize safely (avoid division by zero)
    cm_percent = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        if row_sum > 0:
            cm_percent[i] = cm[i] / row_sum * 100
        else:
            cm_percent[i] = 0  # if row sum is 0, leave zeros
    
    # Annotated text
    z_text = [[f"{int(cm[i][j])}\n({cm_percent[i][j]:.1f}%)"
               for j in range(len(labels))]
              for i in range(len(labels))]
    
    fig = ff.create_annotated_heatmap(
        z=cm_percent,
        x=labels,
        y=labels,
        annotation_text=z_text,
        # colorscale=[ [0, '#d95f5f'], [1, '#5f9fd9'] ],  # low = red, high = blue
        colorscale="Blues",
        showscale=True
    )
    
    fig.update_layout(
        title_text=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis=dict(autorange="reversed")
    )
    
    fig.show()


def plot_feature_importance(feature_weights, title="Feature Importance"):
    fig = px.bar(
        feature_weights.sort_values("Weight", ascending=True),
        x="Weight", y="Feature", color="Weight",
        title=title, orientation="h"
    )
    fig.show()


def plot_wordcloud(feature_weights, model_type="logreg"):
    if model_type == "logreg":
        # Logistic Regression: positive vs negative weights
        word_freq = dict(zip(feature_weights["Feature"], abs(feature_weights["Weight"])))
        colors = {f: ("green" if w > 0 else "red") for f, w in zip(feature_weights["Feature"], feature_weights["Weight"])}

        def color_func(word, *args, **kwargs):
            return colors.get(word, "black")

        wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        plt.imshow(wc.recolor(color_func=color_func), interpolation="bilinear")
        plt.axis("off")
        plt.show()

    else:
        # RF / XGB: only positive importances
        word_freq = dict(zip(feature_weights["Feature"], feature_weights["Weight"]))
        wc = WordCloud(width=800, height=400, background_color="white", colormap="Blues").generate_from_frequencies(word_freq)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()


def plot_wordcloud_by_label(feature_weights, label):
    """
    feature_weights: output of get_feature_importance (for LogReg, includes Label column)
    label: class name (e.g., "FAKE" or "REAL")
    """
    df_label = feature_weights[feature_weights["Label"] == label]

    # Create frequency dict
    word_freq = dict(zip(df_label["Feature"], abs(df_label["Weight"])))
    colors = {f: ("green" if w > 0 else "red") 
              for f, w in zip(df_label["Feature"], df_label["Weight"])}

    def color_func(word, *args, **kwargs):
        return colors.get(word, "black")

    # Generate
    wc = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    plt.imshow(wc.recolor(color_func=color_func), interpolation="bilinear")
    plt.title(f"Word Cloud for class: {label}")
    plt.axis("off")
    plt.show()


