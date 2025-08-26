import plotly.figure_factory as ff
import numpy as np
from sklearn.metrics import confusion_matrix

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
        colorscale="RdYlGn_r",
        showscale=True
    )
    
    fig.update_layout(
        title_text=title,
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        yaxis=dict(autorange="reversed")
    )
    
    fig.show()
