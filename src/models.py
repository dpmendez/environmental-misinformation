from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import precision_recall_fscore_support

def train_classic_model(x_train, y_train,
                        model_type="logreg",
                        ngram_range=(1,2),
                        max_features=10000,
                        class_weight="balanced"):
    
    preprocessor = ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, stop_words="english"), "clean_text")
    ])
    
    if model_type == "logreg":
        classifier = LogisticRegression(max_iter=1000, class_weight=class_weight)
    elif model_type == "rf":
        classifier = RandomForestClassifier(n_estimators=200, class_weight=class_weight, random_state=42)
    elif model_type == "xgb":
        classifier = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
    else:
        raise ValueError("Choose from: 'logreg', 'rf', 'xgb'")
    
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    clf.fit(x_train, y_train)
    return clf
