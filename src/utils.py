import pandas as pd
import numpy as np 
import kagglehub
import os
import re
import string

from datasets import load_dataset
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


def download_climatefever_data():

	# Load Climate-FEVER
	# Climate-FEVER has: claim(text) and claim_label(label)
	cf = load_dataset("tdiggelm/climate_fever")
	cf_full = pd.DataFrame(cf["test"])
	cf_full = cf_full[["claim", "claim_label"]].rename(columns={
		"claim": "text",
		"claim_label": "label"
	})

	# Normalize labels
	# Climate-FEVER: SUPPORTS/REFUTES/NEUTRAL/DISPUTED
	# Let's map labels to an unified scheme
	label_map = {
	    0 : "SUPPORTS",
	    1 : "REFUTES",
	    2 : "NEUTRAL",
	    3 : "DISPUTED",
	    "SUPPORTS": "SUPPORTS",
	    "REFUTES": "REFUTES",
	    "NOT_ENOUGH_INFO": "NEUTRAL"
	}

	cf_full["label"] = cf_full["label"].map(label_map)

	# # Combine datasets
	# df = pd.concat([cf_full, cm_full], ignore_index=True)
	df = cf_full.copy()

	# Some print-outs
	print("Climate-FEVER shape:", cf_full.shape)
	print("Combined shape:", df.shape)
	print("\nLabel distribution:\n", df["label"].value_counts())

	print(df.head())
	print(df.tail())

	return df


def download_kaggle_data():

	path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
	print("Path to dataset files: ", path)

	real_df = pd.read_csv(os.path.join(path, "True.csv"))
	fake_df = pd.read_csv(os.path.join(path, "Fake.csv"))

	# Give labels
	real_df["label"] = 1
	fake_df["label"] = 0

	df = pd.concat([real_df, fake_df], axis=0).reset_index(drop=True)

	return df


def preprocess_kaggle_subject(df, include_title=True):

	print("Custom subject preprocessing")

	df.loc[df["subject"] == 'politicsNews', "subject"] = 'politics'
	df.loc[df["subject"] == 'worldnews', "subject"] = 'world'
	df.loc[df["subject"] == 'Government News', "subject"] = 'government'
	df.loc[df["subject"] == 'US_News', "subject"] = 'usa'
	df.loc[df["subject"] == 'left-news', "subject"] = 'left'
	df.loc[df["subject"] == 'Middle-east', "subject"] = 'middle-east'
	df.loc[df["subject"] == 'News', "subject"] = 'news'
	
	print('unique subjects: ', df['subject'].unique())
	print('df dtypes: ', df.dtypes)

	df.dropna(subset=['title', 'text'], inplace=True)

	if include_title:
		df["clean_text"] = (df["title"] + " " + df["text"]).apply(clean_text)
	else:
		df["clean_text"] = df["text"].apply(clean_text)

	# Remove rows where clean_text is empty
	df = df[df['clean_text'].str.strip() != ""]

	return df 


def clean_text(text):

    text = text.lower() # lowercase
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE) # no urls
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
    text = re.sub(r'\d+', '', text) # remove numbers
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces

    return text


# Split into training, validation and test sets
def split_data(texts, labels):
	# 70, 15, 15 split
	train_texts, temp_texts, train_labels, temp_labels = train_test_split(texts, labels, test_size=0.3, random_state=42)
	val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

	return {'x_train': train_texts, 'y_train': train_labels,
			'x_test': test_texts, 'y_test': test_labels,
			'x_val': val_texts, 'y_val': val_labels}


# Split into training, validation and test sets
def split_data_extras(texts, labels, extras):
	# 70, 15, 15 split
	train_texts, temp_texts, train_labels, temp_labels, train_extras, temp_extras = train_test_split(texts, labels, extras, test_size=0.3, random_state=42)
	val_texts, test_texts, val_labels, test_labels, val_extras, test_extras = train_test_split(temp_texts, temp_labels, temp_extras, test_size=0.5, random_state=42)

	return {'x_train': train_texts, 'y_train': train_labels, 'z_train': train_extras,
			'x_test': test_texts, 'y_test': test_labels, 'z_test': test_extras,
			'x_val': val_texts, 'y_val': val_labels, 'z_val': val_extras}


# Get model features from LR, RF, XGB
def get_feature_importance(this_model, top_n=10):
    clf = this_model.named_steps["classifier"]
    feature_names = this_model.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [f.replace("text__", "") for f in feature_names]

    # Logistic Regression → coefficients
    if hasattr(clf, "coef_"):
        feature_weights_list = []
        for i, label in enumerate(clf.classes_):
            df_i = pd.DataFrame({
                "Feature": feature_names,
                "Weight": clf.coef_[i],
                "Label": label
            })
            feature_weights_list.append(df_i)
        feature_weights = pd.concat(feature_weights_list)

        # Top positive + negative
        top_features = []
        for label in clf.classes_:
            df_label = feature_weights[feature_weights["Label"] == label]
            top_pos = df_label.sort_values("Weight", ascending=False).head(top_n)
            top_neg = df_label.sort_values("Weight").head(top_n)
            top_features.append(pd.concat([top_pos, top_neg]))
        return pd.concat(top_features)

    # RF / XGB → feature importances
    elif hasattr(clf, "feature_importances_"):
        feature_weights = pd.DataFrame({
            "Feature": feature_names,
            "Weight": clf.feature_importances_
        }).sort_values("Weight", ascending=False)
        return feature_weights.head(top_n)

    else:
        raise ValueError("Model does not support feature importance")


# Create a PyTorch Dataset
class NewsDataset(Dataset):

	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		item = {key: val[idx] for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item


