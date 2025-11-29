from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from utils import *

def prepare_data(source="climatefever", combine=True):
    """
    Prepare train/val/test CSVs from either Climate-FEVER or Science Feedback.
    """
    # ----- Load & clean -----
    if combine:
        cf_df = download_climatefever_data()
        cf_df["clean_text"] = cf_df["text"].apply(clean_text)

        sf_df = pd.read_csv("../data/raw/climate_feedback_articles.csv")
        sf_df["unified_label"] = sf_df["verdict"].apply(map_sf_label)
        sf_df = sf_df.dropna(subset=["claim", "unified_label"])
        sf_df["clean_text"] = sf_df["claim"].apply(clean_text)
        sf_df = sf_df.rename(columns={"unified_label": "label", "claim": "text"})

        df = pd.concat(
            [cf_df[["clean_text", "label"]].rename(columns={"clean_text": "text"}),
             sf_df[["text", "label"]]],
            ignore_index=True
        )
        tag = ""
    else:
        if source == "climatefever":
            df = download_climatefever_data()
            df["clean_text"] = df["text"].apply(clean_text)
            df = df.rename(columns={"clean_text": "text"})
            df = df.loc[:, ~df.columns.duplicated()]  # remove duplicate column names
            tag = "_cf"
        elif source == "sciencefeedback":
            df = pd.read_csv("../data/raw/climate_feedback_articles.csv")
            df["unified_label"] = df["verdict"].apply(map_sf_label)
            df = df.dropna(subset=["claim", "unified_label"])
            df["clean_text"] = df["claim"].apply(clean_text)
            df = df.rename(columns={"unified_label": "label", "clean_text": "text"})
            tag = "_sf"
        else:
            raise ValueError("Unknown source. Choose 'climatefever' or 'sciencefeedback'")

    print("Label distribution:")
    print(df["label"].value_counts())
    
    # Collapse labels into two as there are too few examples
    df["label"] = df["label"].replace({
        "SUPPORTS": "LIKELY_TRUE",
        "NEUTRAL": "LIKELY_TRUE",
        "REFUTES": "LIKELY_FALSE",
        "DISPUTED": "LIKELY_FALSE"
    })

    # ----- Split -----
    X_train, X_temp, y_train, y_temp = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    # ----- Combine into DataFrames -----
    train_df = pd.DataFrame({"text": X_train, "label": y_train})
    val_df   = pd.DataFrame({"text": X_val, "label": y_val})
    test_df  = pd.DataFrame({"text": X_test, "label": y_test})

    # ----- Save -----
    os.makedirs("../data", exist_ok=True)
    train_df.to_csv(f"../data/train_data{tag}.csv", index=False)
    val_df.to_csv(f"../data/val_data{tag}.csv", index=False)
    test_df.to_csv(f"../data/test_data{tag}.csv", index=False)

    print(f"Saved train/val/test CSVs for {source} with tag '{tag}'")
    print("Train:", train_df.shape, " Val:", val_df.shape, " Test:", test_df.shape)
    print(train_df["label"].value_counts())

    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_data(source="sciencefeedback")