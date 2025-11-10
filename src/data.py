from utils import *
import os
import pandas as pd

def prepare_data(source="climatefever", combine=False):
    """
    Prepare train/val/test CSVs from either Climate-FEVER or Science Feedback.
    source: "climatefever" or "sciencefeedback"
    combine: If True, combine both datasets into one (no source tag in filenames)
    """
    if combine:
        cf_df = download_climatefever_data()
        cf_df["clean_text"] = cf_df["text"].apply(clean_text)
        
        sf_df = pd.read_csv("raw/climate_feedback_articles.csv")
        sf_df["unified_label"] = sf_df["verdict"].apply(map_sf_label)
        sf_df = sf_df.dropna(subset=["claim", "unified_label"])
        sf_df["clean_text"] = sf_df["claim"].apply(clean_text)
        sf_df = sf_df.rename(columns={"unified_label": "label", "claim": "text"})
        
        df = pd.concat([cf_df[["clean_text", "label"]].rename(columns={"clean_text":"text"}), 
                        sf_df[["text", "label"]]], ignore_index=True)
        tag = ""
    else:
        if source == "climatefever":
            df = download_climatefever_data()
            df["clean_text"] = df["text"].apply(clean_text)
            df = df.rename(columns={"clean_text": "text"})
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

    # split data
    data_split = split_data(df['text'], df['label'])

    # convert to DataFrames
    train_df = pd.DataFrame({"text": data_split['x_train'], "label": data_split['y_train']})
    val_df   = pd.DataFrame({"text": data_split['x_val'],   "label": data_split['y_val']})
    test_df  = pd.DataFrame({"text": data_split['x_test'],  "label": data_split['y_test']})

    # save
    os.makedirs("../data", exist_ok=True)
    train_df.to_csv(f"../data/train_data{tag}.csv", index=False)
    val_df.to_csv(f"../data/val_data{tag}.csv", index=False)
    test_df.to_csv(f"../data/test_data{tag}.csv", index=False)

    print(f"Saved train/val/test CSVs for {source} with tag '{tag}'")
    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_data(source="sciencefeedback")
    # or to combine both datasets:
    # prepare_data(combine=True)